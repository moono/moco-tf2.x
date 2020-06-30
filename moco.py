import os
import time
import argparse
import numpy as np
import tensorflow as tf

from pprint import pprint as pp
from misc.utils import str_to_bool
from misc.tf_utils import allow_memory_growth, split_gpu_for_testing
from datasets.toy_dataset import get_dataset as get_toy_dataset
from base_networks.load_model import load_model


ALL_MOCO_PARAMS = {
    # debugging
    0: {
        'base_encoder': 'linear',
        'network_params': {'input_shape': [1], 'dim': 4, 'K': 16, 'm': 0.999, 'T': 0.07}
    },
    # MoCo v1
    1: {
        'base_encoder': 'resnet50',
        'network_params': {'input_shape': [224, 224, 3], 'dim': 128, 'K': 65536, 'm': 0.999, 'T': 0.07, 'mlp': False},
    },
    # MoCo v2
    2: {
        'base_encoder': 'resnet50',
        'network_params': {'input_shape': [224, 224, 3], 'dim': 128, 'K': 65536, 'm': 0.999, 'T': 0.2, 'mlp': True}
    }
}


def constant_learning_rate_decay(global_batch_size, n_images, initial_lr, decay, epoch_decay):
    assert isinstance(global_batch_size, int) and isinstance(n_images, int)
    assert isinstance(initial_lr, float) and isinstance(decay, float)
    assert isinstance(epoch_decay, list) and isinstance(all(epoch_decay), int)

    images_per_step = n_images / global_batch_size
    boundaries_steps = [int(val * images_per_step) for val in epoch_decay]
    values = [initial_lr] + [initial_lr * (decay ** (p + 1)) for p in range(len(boundaries_steps))]

    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries_steps, values)
    return schedule


class MoCo(object):
    def __init__(self, t_params):
        assert t_params['base_encoder'] in ['resnet50', 'linear']

        # global parameters
        self.name = t_params['name']
        self.use_tf_function = t_params['use_tf_function']
        self.model_base_dir = t_params['model_base_dir']
        self.batch_size = t_params['batch_size_per_replica']
        self.global_batch_size = t_params['global_batch_size']
        self.n_images = t_params['n_images']
        self.max_steps = int(np.ceil(self.n_images / self.global_batch_size))
        self.reached_max_steps = False
        self.print_step = 50
        self.save_step = 200
        self.image_summary_step = 200

        # moco parameters
        self.base_encoder = t_params['base_encoder']
        self.dim = t_params['network_params']['dim']
        self.K = t_params['network_params']['K']
        self.m = t_params['network_params']['m']
        self.T = t_params['network_params']['T']

        # create the encoders and clone(q -> k)
        self.encoder_q = load_model('encoder_q', self.base_encoder, t_params['network_params'], trainable=True)
        self.encoder_k = load_model('encoder_k', self.base_encoder, t_params['network_params'], trainable=False)
        for qw, kw in zip(self.encoder_q.weights, self.encoder_k.weights):
            kw.assign(qw)

        # create queue
        self.queue, self.queue_ptr = self._setup_queue()

        # create optimizer
        self.lr_schedule = constant_learning_rate_decay(self.global_batch_size,
                                                        t_params['n_images'],
                                                        t_params['initial_lr'],
                                                        t_params['lr_decay'],
                                                        t_params['lr_decay_boundaries'])
        self.optimizer = tf.keras.optimizers.SGD(self.lr_schedule, momentum=0.9, nesterov=True)

        # setup saving locations (object based savings)
        self.ckpt_dir = os.path.join(self.model_base_dir, self.name)
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, encoder_q=self.encoder_q, encoder_k=self.encoder_k)
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
        # queue pointer
        queue_ptr_init = tf.zeros(shape=[], dtype=tf.int64)
        queue_ptr = tf.Variable(queue_ptr_init, trainable=False,
                                synchronization=tf.VariableSynchronization.ON_READ,
                                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        # actual queue
        queue_init = tf.math.l2_normalize(tf.random.normal([self.K, self.dim]), axis=1)
        queue = tf.Variable(queue_init, trainable=False,
                            synchronization=tf.VariableSynchronization.ON_READ,
                            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        return queue, queue_ptr

    def _batch_shuffle(self, all_gathered, strategy):
        # convert to tf.Tensor
        data = tf.concat(strategy.experimental_local_results(all_gathered), axis=0)

        # create shuffled index for global batch size
        all_idx = tf.range(self.global_batch_size)
        shuffled_idx = tf.random.shuffle(all_idx)

        # shuffle
        shuffled_data = tf.gather(data, indices=shuffled_idx)
        return shuffled_data, shuffled_idx

    def _batch_unshuffle(self, all_shuffled, shuffled_idx, strategy):
        all_shuffled = tf.concat(strategy.experimental_local_results(all_shuffled), axis=0)     # [GN, C]
        shuffled_idx = tf.expand_dims(shuffled_idx, axis=1)                                     # [GN, 1]
        output_shape = tf.shape(all_shuffled)                                                   # [GN, C]
        unshuffled_data = tf.scatter_nd(indices=shuffled_idx, updates=all_shuffled, shape=output_shape)
        return unshuffled_data

    def _dequeue_and_enqueue(self, keys):
        # keys: [GN, C]
        end_queue_ptr = self.queue_ptr + self.global_batch_size
        indices = tf.range(self.queue_ptr, end_queue_ptr, dtype=tf.int64)   # [GN,  ]
        indices = tf.expand_dims(indices, axis=1)                           # [GN, 1]

        updated_queue = tf.tensor_scatter_nd_update(tensor=self.queue, indices=indices, updates=keys)
        updated_queue_ptr = end_queue_ptr % self.K

        self.queue.assign(updated_queue)
        self.queue_ptr.assign(updated_queue_ptr)
        return

    def forward_encoder_k(self, inputs):
        all_im_k = inputs[0]

        # inputs are already shuffled, just take shuffled data respect to their index will suffice
        all_idx = tf.range(self.global_batch_size)
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
        this_start = replica_id * self.batch_size
        this_end = this_start + self.batch_size
        this_idx = all_idx[this_start:this_end]

        # get this replica's im_k
        im_k = tf.gather(all_im_k, indices=this_idx)

        # compute query features
        k = self.encoder_k(im_k)  # keys: NxC
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

        with tf.GradientTape() as tape:
            # compute query features
            q = self.encoder_q(im_q)  # queries: NxC
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
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)  # [N, ]

            # scale losses
            loss = tf.reduce_sum(loss) * (1.0 / self.global_batch_size)

        t_var = self.encoder_q.trainable_variables
        gradients = tape.gradient(loss, t_var)
        self.optimizer.apply_gradients(zip(gradients, t_var))
        return loss

    def train(self, dist_dataset, strategy):
        def dist_encode_key(inputs):
            per_replica_result = strategy.experimental_run_v2(fn=self.forward_encoder_k, args=(inputs,))
            return per_replica_result

        def dist_train_step(inputs):
            per_replica_losses = strategy.experimental_run_v2(fn=self.train_step, args=(inputs,))
            mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            return mean_loss

        if self.use_tf_function:
            dist_encode_key = tf.function(dist_encode_key)
            dist_train_step = tf.function(dist_train_step)

        if self.reached_max_steps:
            return

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

        # loss metrics
        metric_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

        # start actual training
        print(f'max_steps: {self.max_steps}')
        t_start = time.time()
        for im_q, im_k in dist_dataset:
            # train step

            # shuffle data on global scale
            # shuffled_im_k: [GN, ]
            # shuffled_idx: [GN, ]
            shuffled_im_k, shuffled_idx = self._batch_shuffle(im_k, strategy)

            # run on encoder_k to collect shuffled keys
            k_shuffled = dist_encode_key((shuffled_im_k, ))

            # unshuffle and merge all
            k_unshuffled = self._batch_unshuffle(k_shuffled, shuffled_idx, strategy)

            # train step: update queue encoder
            loss = dist_train_step((im_q, k_unshuffled))

            # update key encoder
            self.encoder_k.momentum_update(self.encoder_q, self.m)

            # update queue
            self._dequeue_and_enqueue(k_unshuffled)

            # update metrics
            metric_loss(loss)

            # get current step
            step = self.optimizer.iterations.numpy()

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', metric_loss.result(), step=step)

            # print every self.print_steps
            if step % self.print_step == 0:
                elapsed = time.time() - t_start
                print('[step: {}] elapsed {:.2f}s, loss {:.3f}'.format(step, elapsed, loss.numpy()))

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


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--debug_split_gpu', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_tf_function', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--name', default='debugging', type=str)
    parser.add_argument('--model_base_dir', default='./models', type=str)
    parser.add_argument('--dataset_name', default='toy', type=str)
    parser.add_argument('--dataset_n_images', default=1000000, type=int)
    parser.add_argument('--moco_version', default=0, type=int)
    parser.add_argument('--base_encoder', default='linear', type=str)
    parser.add_argument('--batch_size_per_replica', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--initial_lr', default=0.003, type=float)
    parser.add_argument('--lr_decay_factor', default=0.1, type=float)
    parser.add_argument('--lr_decay_boundaries', nargs='*', type=int)   # takes list of ints
    args = vars(parser.parse_args())

    # default lr_decay_boundaries value
    if args['lr_decay_boundaries'] is None:
        args['lr_decay_boundaries'] = [120, 160]

    # GPU environment settings
    if args['allow_memory_growth']:
        allow_memory_growth()
    if args['debug_split_gpu']:
        split_gpu_for_testing()

    # get MoCo parameters
    assert args['moco_version'] in [0, 1, 2]
    moco_params = ALL_MOCO_PARAMS[args['moco_version']]

    # prepare distribute training
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = args['batch_size_per_replica'] * strategy.num_replicas_in_sync

    # training parameters
    training_parameters = {
        # global params
        'name': args['name'],
        'use_tf_function': args['use_tf_function'],
        'model_base_dir': args['model_base_dir'],

        # moco params
        **moco_params,

        # training params
        'n_images': args['dataset_n_images'],
        'epochs': args['epochs'],
        'initial_lr': args['initial_lr'],
        'lr_decay': args['lr_decay_factor'],
        'lr_decay_boundaries': args['lr_decay_boundaries'],
        'batch_size_per_replica': args['batch_size_per_replica'],
        'global_batch_size': global_batch_size,
    }

    # print current details
    pp(training_parameters)

    # load dataset
    if args['dataset_name']:
        dataset = get_toy_dataset(global_batch_size, args['dataset_n_images'], args['epochs'])
    else:
        dataset = None

    with strategy.scope():
        # create MoCo instance
        moco = MoCo(training_parameters)

        # distribute dataset
        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        # start training
        print('Training...')
        moco.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
