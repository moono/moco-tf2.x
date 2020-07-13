import os
import argparse
import numpy as np
import tensorflow as tf

from pprint import pprint as pp
from misc.utils import str_to_bool
from misc.tf_utils import allow_memory_growth, split_gpu_for_testing
from datasets.imagenet import get_dataset as get_imagenet_dataset

from base_networks.load_model import load_model


class CosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, max_step_size, name=None):
        super(CosineDecay, self).__init__()
        self.initial_learning_rate = initial_lr
        self.max_step_size = max_step_size
        self.name = name

    def __call__(self, step):
        initial_lr = tf.convert_to_tensor(self.initial_learning_rate)
        pi = tf.convert_to_tensor(np.pi)
        global_step_recomp = tf.cast(step, initial_lr.dtype)
        return initial_lr * 0.5 * (1.0 + tf.cos(global_step_recomp / self.max_step_size * pi))

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'max_step_size': self.max_step_size,
            'name': self.name,
        }


def constant_learning_rate_decay(global_batch_size, n_images, initial_lr, decay, epoch_decay):
    assert isinstance(global_batch_size, int) and isinstance(n_images, int)
    assert isinstance(initial_lr, float) and isinstance(decay, float)
    if epoch_decay is None:
        return initial_lr

    assert isinstance(epoch_decay, list) and isinstance(all(epoch_decay), int)

    images_per_step = n_images / global_batch_size
    boundaries_steps = [int(val * images_per_step) for val in epoch_decay]
    values = [initial_lr] + [initial_lr * (decay ** (p + 1)) for p in range(len(boundaries_steps))]

    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries_steps, values)
    return schedule


class MoCo(object):
    def __init__(self, t_params, strategy):
        # global parameters
        self.name = t_params['name']
        self.moco_version = t_params['moco_version']
        self.use_tf_function = t_params['use_tf_function']
        self.model_base_dir = t_params['model_base_dir']
        self.batch_size = t_params['batch_size_per_replica']
        self.global_batch_size = t_params['global_batch_size']
        self.epochs = t_params['epochs']
        self.n_images = t_params['n_images']
        self.max_steps = int(np.ceil(self.n_images / self.global_batch_size))
        self.epochs_per_step = self.global_batch_size / (self.n_images / self.epochs)
        self.reached_max_steps = False
        self.summary_step = 10
        self.accuracy_step = 50
        self.print_step = 100
        self.save_step = 500
        self.n_replica = self.global_batch_size // self.batch_size
        self.profile_dir = os.path.join(self.model_base_dir, self.name)

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
                self.lr_schedule_fn = constant_learning_rate_decay(self.global_batch_size,
                                                                   t_params['n_images'],
                                                                   t_params['learning_rate']['initial_lr'],
                                                                   t_params['learning_rate']['lr_decay'],
                                                                   t_params['learning_rate']['lr_decay_boundaries'])
            else:
                self.lr_schedule_fn = CosineDecay(t_params['learning_rate']['initial_lr'], self.max_steps)
            self.optimizer = tf.keras.optimizers.SGD(self.lr_schedule_fn, momentum=0.9, nesterov=True)
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

    def _batch_shuffle(self, im_k, strategy):
        collected_im_k = tf.concat(strategy.experimental_local_results(im_k), axis=0)
        global_batch_size = tf.shape(collected_im_k)[0]

        # create shuffled index for global batch size
        indices = tf.range(global_batch_size)
        shuffled_idx = tf.random.shuffle(indices)

        # shuffle
        shuffled_data = tf.gather(collected_im_k, indices=shuffled_idx)
        return shuffled_data, shuffled_idx

    def _batch_unshuffle(self, k_shuffled, shuffled_idx, strategy):
        collected_k_shuffled = tf.concat(strategy.experimental_local_results(k_shuffled), axis=0)
        output_shape = tf.shape(collected_k_shuffled)  # [GN, C]
        shuffled_idx = tf.expand_dims(shuffled_idx, axis=1)  # [GN, 1]
        unshuffled_im_k = tf.scatter_nd(indices=shuffled_idx, updates=collected_k_shuffled, shape=output_shape)
        return unshuffled_im_k

    def _dequeue_and_enqueue(self, keys):
        # keys: [GN, C]
        end_queue_ptr = self.queue_ptr + self.global_batch_size
        indices = tf.range(self.queue_ptr, end_queue_ptr, dtype=tf.int64)  # [GN,  ]
        indices = tf.expand_dims(indices, axis=1)  # [GN, 1]

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
        this_idx = all_idx[this_start:this_end]  # [N, ]

        # get this replica's im_k
        im_k = tf.gather(shuffled_im_k, indices=this_idx)  # [N, res, res, 3]

        # compute query features
        k = self.encoder_k(im_k, training=True)  # keys: NxC
        k = tf.math.l2_normalize(k, axis=1)
        return k

    def train_step(self, inputs):
        im_q, all_k = inputs

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

            # weight regularization loss
            l2_w_reg = tf.reduce_sum(self.encoder_q.losses)

            # scale to global batch scale
            c_loss = tf.reduce_sum(c_loss) * (1.0 / self.global_batch_size)
            loss = c_loss + l2_w_reg

        gradients = tape.gradient(loss, t_var)
        self.optimizer.apply_gradients(zip(gradients, t_var))
        return loss, c_loss, l2_w_reg, logits

    def train(self, dist_dataset, strategy):
        def dist_encode_key(inputs):
            # per_replica_result = strategy.experimental_run_v2(fn=self.forward_encoder_k, args=(inputs,))
            per_replica_result = strategy.run(fn=self.forward_encoder_k, args=(inputs,))
            return per_replica_result

        def dist_train_step(inputs):
            # per_replica_losses = strategy.experimental_run_v2(fn=self.train_step, args=(inputs,))
            per_replica_out = strategy.run(fn=self.train_step, args=(inputs,))
            mean_loss0 = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_out[0], axis=None)
            mean_loss1 = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_out[1], axis=None)
            mean_loss2 = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_out[2], axis=None)
            out_logits = tf.concat(strategy.experimental_local_results(per_replica_out[3]), axis=0)
            return mean_loss0, mean_loss1, mean_loss2, out_logits

        if self.use_tf_function:
            dist_encode_key = tf.function(dist_encode_key)
            dist_train_step = tf.function(dist_train_step)

        profile_started = False

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.profile_dir)

        # start actual training
        print('Profiling warm up...')
        for im_q, im_k in dist_dataset:
            # shuffle data on global scale
            shuffled_im_k, shuffled_idx = self._batch_shuffle(im_k, strategy)

            # run on encoder_k to collect shuffled keys
            k_shuffled = dist_encode_key((shuffled_im_k,))

            # unshuffle and merge all
            k_unshuffled = self._batch_unshuffle(k_shuffled, shuffled_idx, strategy)

            # train step: update queue encoder
            mean_loss, mean_c_loss, mean_l2_reg, logits = dist_train_step((im_q, k_unshuffled))

            # update key encoder
            self.encoder_k.momentum_update(self.encoder_q, self.m)

            # update queue
            self._dequeue_and_enqueue(k_unshuffled)

            # get current step
            step = self.optimizer.iterations.numpy()
            c_lr = self.optimizer.learning_rate(self.optimizer.iterations)
            c_ep = self.epochs_per_step * step

            # compute accuracy (tf.argmax takes too much time...)
            if step % self.accuracy_step == 0:
                labels = tf.zeros(shape=(tf.shape(logits)[0]), dtype=tf.int64)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(logits, axis=1), labels), tf.float32))
                tf.summary.scalar('accuracy', accuracy, step=step)

            # save to tensorboard
            if step % self.summary_step == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('epochs', c_ep, step=step)
                    tf.summary.scalar('learning_rate', c_lr, step=step)
                    tf.summary.scalar('info_NCE', mean_c_loss, step=step)
                    tf.summary.scalar('w_l2_reg', mean_l2_reg, step=step)
                    tf.summary.scalar('total_loss', mean_loss, step=step)
                    tf.summary.histogram('queue_0', self.queue[0, :], step=step)
                    tf.summary.histogram('queue_-1', self.queue[-1, :], step=step)

            # profile
            if step > 100 and not profile_started:
                print('Profiling started')
                tf.profiler.experimental.start(self.profile_dir)
                profile_started = True

            if step > 200 and profile_started:
                print('Profiling ended')
                tf.profiler.experimental.stop(self.profile_dir)
                break
        return


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--debug_split_gpu', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_tf_function', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--name', default='profile2', type=str)
    parser.add_argument('--tfds_data_dir', default='/mnt/vision-nas/data-sets/tensorflow_datasets', type=str)
    parser.add_argument('--model_base_dir', default='./models', type=str)
    parser.add_argument('--moco_version', default=2, type=int)
    parser.add_argument('--batch_size_per_replica', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    args = vars(parser.parse_args())

    # check args
    assert args['moco_version'] in [1, 2]

    # GPU environment settings
    if args['allow_memory_growth']:
        allow_memory_growth()
    if args['debug_split_gpu']:
        split_gpu_for_testing()

    # default values
    dataset_n_images = {'train': 1281167, 'validation': 50000}
    w_decay = 0.0001
    if args['moco_version'] == 1:
        moco_params = {
            'base_encoder': 'resnet50',
            'network_params': {
                'input_shape': [224, 224, 3],
                'dim': 128,
                'K': 65536,
                'm': 0.999,
                'T': 0.07,
                'mlp': False,
                'w_decay': w_decay,
            },
            'learning_rate': {
                'schedule': 'step',
                'initial_lr': 0.03,
                'lr_decay': 0.1,
                'lr_decay_boundaries': [120, 160],
            }
        }
    else:
        moco_params = {
            'base_encoder': 'resnet50',
            'network_params': {
                'input_shape': [224, 224, 3],
                'dim': 128,
                'K': 65536,
                'm': 0.999,
                'T': 0.2,
                'mlp': True,
                'w_decay': w_decay,
            },
            'learning_rate': {
                'schedule': 'cos',
                'initial_lr': 0.03,
            }
        }
    res = moco_params['network_params']['input_shape'][0]

    # prepare distribute training
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = args['batch_size_per_replica'] * strategy.num_replicas_in_sync

    # training parameters
    training_parameters = {
        # global params
        'name': f'{args["name"]}_moco_v{args["moco_version"]}',
        'use_tf_function': args['use_tf_function'],
        'model_base_dir': args['model_base_dir'],

        # moco params
        'moco_version': args['moco_version'],
        **moco_params,

        # training params
        'n_images': dataset_n_images['train'] * args['epochs'],
        'epochs': args['epochs'],
        'batch_size_per_replica': args['batch_size_per_replica'],
        'global_batch_size': global_batch_size,
    }

    # print current details
    pp(training_parameters)

    # load dataset
    dataset = get_imagenet_dataset(args['tfds_data_dir'], is_training=True, res=res, moco_ver=args['moco_version'],
                                   batch_size=global_batch_size, epochs=args['epochs'])

    # create MoCo instance
    moco = MoCo(training_parameters, strategy)

    with strategy.scope():
        # distribute dataset
        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        # start training
        print('Training...')
        moco.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
