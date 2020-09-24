import os
import time
import numpy as np
import tensorflow as tf

from base_networks.load_model import load_model
from misc.learning_rate_schedule import StepDecay, CosineDecay


class MoCo(object):
    def __init__(self, t_params, strategy):
        # global parameters
        self.cur_tf_ver = t_params['cur_tf_ver']
        self.name = t_params['name']
        self.moco_version = t_params['moco_version']
        self.aug_fn = t_params['aug_fn']
        self.res = t_params['res']
        self.use_tf_function = t_params['use_tf_function']
        self.model_base_dir = t_params['model_base_dir']
        self.batch_size = t_params['batch_size_per_replica']
        self.global_batch_size = t_params['global_batch_size']
        self.epochs = t_params['epochs']
        self.n_images = t_params['n_images']
        self.max_steps = int(np.ceil(self.n_images / self.global_batch_size))
        self.epochs_per_step = self.global_batch_size / (self.n_images / self.epochs)
        self.reached_max_steps = False
        self.print_step = 100
        self.save_step = 500
        self.n_replica = self.global_batch_size // self.batch_size
        self.dist_loss_scaler = 1.0 / self.global_batch_size
        self.dist_wreg_scaler = 1.0 / self.n_replica

        # moco parameters
        self.base_encoder = t_params['base_encoder']
        self.dim = t_params['network_params']['dim']
        self.K = t_params['network_params']['K']
        self.m = t_params['network_params']['m']
        self.T = t_params['network_params']['T']

        # create queue
        self.queue, self.queue_ptr = self._setup_queue()

        # model, optimizer, and checkpoint must be created under `strategy.scope`
        with strategy.scope():
            # create the encoders and clone(q -> k)
            self.encoder_q = load_model('encoder_q', self.base_encoder, t_params['network_params'], trainable=True)
            self.encoder_k = load_model('encoder_k', self.base_encoder, t_params['network_params'], trainable=True)
            for qw, kw in zip(self.encoder_q.weights, self.encoder_k.weights):
                assert qw.shape == kw.shape
                assert qw.name == kw.name
                # don't copy ema variables
                if 'moving' in qw.name:
                    continue
                kw.assign(qw)

            # create optimizer
            if self.moco_version == 1:
                self.lr_schedule_fn = StepDecay(t_params['learning_rate']['initial_lr'],
                                                t_params['learning_rate']['lr_decay'],
                                                t_params['n_images'],
                                                t_params['epochs'],
                                                t_params['learning_rate']['lr_decay_boundaries'],
                                                t_params['global_batch_size'])
            else:
                self.lr_schedule_fn = CosineDecay(t_params['learning_rate']['initial_lr'], self.max_steps)
            self.optimizer = tf.keras.optimizers.SGD(self.lr_schedule_fn, momentum=0.9, nesterov=False)
            # self.optimizer = tf.keras.optimizers.SGD(self.lr_schedule_fn, momentum=0.9, nesterov=True)

            # setup saving locations (object based savings)
            self.ckpt_dir = os.path.join(self.model_base_dir, self.name)
            self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer,
                                            encoder_q=self.encoder_q,
                                            encoder_k=self.encoder_k,
                                            queue_ptr=self.queue_ptr,
                                            queue=self.queue)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=2)

            # try to restore
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print('Restored from {}'.format(self.manager.latest_checkpoint))

                # check if already trained in this resolution
                restored_step = self.optimizer.iterations.numpy()
                if restored_step >= self.max_steps:
                    print('Already reached max steps {}/{}'.format(restored_step, self.max_steps))
                    self.reached_max_steps = True
                    return
            else:
                print('Not restoring from saved checkpoint')
        return

    def _setup_queue(self):
        with tf.device('GPU:0'):
            # queue pointer
            queue_ptr_init = tf.zeros(shape=[], dtype=tf.int64)
            queue_ptr = tf.Variable(queue_ptr_init, trainable=False)

            # actual queue
            queue_init = tf.math.l2_normalize(tf.random.normal([self.K, self.dim]), axis=1)
            queue = tf.Variable(queue_init, trainable=False)
        return queue, queue_ptr

    @tf.function
    def _batch_shuffle(self, im_k, strategy):
        collected_im_k = tf.concat(strategy.experimental_local_results(im_k), axis=0)
        global_batch_size = tf.shape(collected_im_k)[0]

        # create shuffled index for global batch size
        indices = tf.range(global_batch_size)
        shuffled_idx = tf.random.shuffle(indices)

        # shuffle
        shuffled_data = tf.gather(collected_im_k, indices=shuffled_idx)
        return shuffled_data, shuffled_idx

    @tf.function
    def _batch_unshuffle(self, k_shuffled, shuffled_idx, strategy):
        collected_k_shuffled = tf.concat(strategy.experimental_local_results(k_shuffled), axis=0)
        output_shape = tf.shape(collected_k_shuffled)           # [GN, C]
        shuffled_idx = tf.expand_dims(shuffled_idx, axis=1)     # [GN, 1]
        unshuffled_im_k = tf.scatter_nd(indices=shuffled_idx, updates=collected_k_shuffled, shape=output_shape)
        return unshuffled_im_k

    @tf.function
    def _dequeue_and_enqueue(self, keys):
        # keys: [GN, C]
        end_queue_ptr = self.queue_ptr + self.global_batch_size
        indices = tf.range(self.queue_ptr, end_queue_ptr, dtype=tf.int64)   # [GN,  ]
        indices = tf.expand_dims(indices, axis=1)                           # [GN, 1]

        # update to new values
        updated_queue = tf.tensor_scatter_nd_update(tensor=self.queue, indices=indices, updates=keys)
        updated_queue_ptr = end_queue_ptr % self.K

        # update queue
        self.queue.assign(updated_queue)

        # update pointer
        self.queue_ptr.assign(updated_queue_ptr)
        return

    def forward_encoder_k(self, inputs):
        shuffled_im_k = inputs[0]
        global_batch_size = tf.shape(shuffled_im_k)[0]  # GN

        # inputs are already shuffled, just take shuffled data respect to their index will suffice
        all_idx = tf.range(global_batch_size)
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
        this_start = replica_id * self.batch_size
        this_end = this_start + self.batch_size
        this_idx = all_idx[this_start:this_end]     # [N, ]

        # get this replica's im_k
        im_k = tf.gather(shuffled_im_k, indices=this_idx)   # [N, res, res, 3]

        # run augmentation on gpu
        if self.aug_fn is not None:
            im_k = self.aug_fn(im_k, self.res)

        # compute query features
        k = self.encoder_k(im_k, training=True)  # keys: NxC
        k = tf.math.l2_normalize(k, axis=1)
        return k

    def train_step(self, inputs):
        im_q, all_k, compute_accuracy = inputs

        # run augmentation on gpu
        if self.aug_fn is not None:
            im_q = self.aug_fn(im_q, self.res)

        # get appropriate keys
        all_idx = tf.range(self.global_batch_size)
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
        this_start = replica_id * self.batch_size
        this_end = this_start + self.batch_size
        this_idx = all_idx[this_start:this_end]
        k = tf.gather(all_k, indices=this_idx)

        t_var = self.encoder_q.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([t_var])

            # compute query features
            q = self.encoder_q(im_q, training=True)  # queries: NxC
            q = tf.math.l2_normalize(q, axis=1)

            # compute logits: Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = tf.expand_dims(tf.einsum('nc,nc->n', q, k), axis=-1)
            # negative logits: NxK
            l_neg = tf.einsum('nc,kc->nk', q, self.queue)

            # logits: Nx(1+K)
            logits = tf.concat([l_pos, l_neg], axis=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = tf.zeros(self.batch_size, dtype=tf.int64)  # [N, ]
            c_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)  # [N, ]

            # tf.argmax takes too much time...
            if compute_accuracy:
                accuracy = tf.cast(tf.equal(tf.math.argmax(logits, axis=1), labels), tf.float32)
            else:
                accuracy = 0.0

            # weight regularization loss
            # l2_w_reg = tf.reduce_sum(self.encoder_q.losses)
            l2_w_reg = tf.nn.scale_regularization_loss(self.encoder_q.losses)

            # scale to global batch scale
            accuracy = tf.reduce_sum(accuracy) * self.dist_loss_scaler
            c_loss = tf.reduce_sum(c_loss) * self.dist_loss_scaler
            # l2_w_reg = l2_w_reg * self.dist_wreg_scaler
            loss = c_loss + l2_w_reg

        gradients = tape.gradient(loss, t_var)
        self.optimizer.apply_gradients(zip(gradients, t_var))
        return loss, c_loss, l2_w_reg, accuracy

    def train(self, dist_dataset, strategy):
        def dist_encode_key(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_result = strategy.experimental_run_v2(fn=self.forward_encoder_k, args=(inputs,))
            else:
                per_replica_result = strategy.run(fn=self.forward_encoder_k, args=(inputs,))
            return per_replica_result

        def dist_train_step(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_out = strategy.experimental_run_v2(fn=self.train_step, args=(inputs,))
            else:
                per_replica_out = strategy.run(fn=self.train_step, args=(inputs,))
            mean_loss0 = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_out[0], axis=None)
            mean_loss1 = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_out[1], axis=None)
            mean_loss2 = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_out[2], axis=None)
            mean_acc = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_out[3], axis=None)
            return mean_loss0, mean_loss1, mean_loss2, mean_acc

        if self.use_tf_function:
            dist_encode_key = tf.function(dist_encode_key)
            dist_train_step = tf.function(dist_train_step)

        if self.reached_max_steps:
            return

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

        # start actual training
        print(f'max_steps: {self.max_steps}')
        t_start = time.time()
        for im_q, im_k in dist_dataset:
            compute_accuracy = True if (self.optimizer.iterations + 1) % self.print_step == 0 else False

            # shuffle data on global scale
            shuffled_im_k, shuffled_idx = self._batch_shuffle(im_k, strategy)

            # run on encoder_k to collect shuffled keys
            k_shuffled = dist_encode_key((shuffled_im_k, ))

            # unshuffle and merge all
            k_unshuffled = self._batch_unshuffle(k_shuffled, shuffled_idx, strategy)

            # train step: update queue encoder
            mean_loss, mean_c_loss, mean_l2_reg, mean_accuracy = dist_train_step((im_q, k_unshuffled, compute_accuracy))

            # update key encoder
            self.encoder_k.momentum_update(self.encoder_q, self.m)

            # update queue
            self._dequeue_and_enqueue(k_unshuffled)

            # get current step
            step = self.optimizer.iterations.numpy()

            # print every self.print_steps == on compute_accuracy
            if compute_accuracy:
                elapsed = time.time() - t_start

                c_lr = self.optimizer.learning_rate(self.optimizer.iterations)
                c_ep = self.epochs_per_step * step

                # save to tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('accuracy', mean_accuracy, step=step)
                    tf.summary.scalar('epochs', c_ep, step=step)
                    tf.summary.scalar('learning_rate', c_lr, step=step)
                    tf.summary.scalar('info_NCE', mean_c_loss, step=step)
                    tf.summary.scalar('w_l2_reg', mean_l2_reg, step=step)
                    tf.summary.scalar('total_loss', mean_loss, step=step)
                    # tf.summary.histogram('queue_0', self.queue[0, :], step=step)
                    # tf.summary.histogram('queue_-1', self.queue[-1, :], step=step)

                logs_h = '[step/epoch/lr: {}/{:.3f}/{:.3f} in {:.2f}s]: '.format(step, c_ep, c_lr.numpy(), elapsed)
                logs_b = 'loss {:.3f}, c_loss {:.3f}, l2_reg {:.3f}, acc {:.3f}'.format(mean_loss.numpy(),
                                                                                        mean_c_loss.numpy(),
                                                                                        mean_l2_reg.numpy(),
                                                                                        mean_accuracy.numpy())
                print(f'{logs_h}{logs_b}')

                # reset timer
                t_start = time.time()

            # save every self.save_step
            if step % self.save_step == 0:
                self.manager.save(checkpoint_number=step)

            # check exit status
            if step >= self.max_steps:
                break

        # save last checkpoint
        step = self.optimizer.iterations.numpy()
        self.manager.save(checkpoint_number=step)
        return
