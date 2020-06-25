import argparse
import tensorflow as tf

from misc.utils import str_to_bool
from misc.tf_utils import allow_memory_growth
from base_networks.load_model import load_model


class MoCoTrainer(object):
    def __init__(self, t_params, name):
        assert t_params['base_encoder'] == 'resnet50'

        # global parameters
        self.name = name
        self.use_tf_function = t_params['use_tf_function']
        self.model_base_dir = t_params['model_base_dir']
        self.train_res = t_params['train_res']
        self.global_batch_size = t_params['global_batch_size']
        self.n_replicas = t_params['n_replicas']

        # moco parameters
        self.base_encoder = t_params['base_encoder']
        self.dim = t_params['dim']
        self.res = t_params['res']
        self.K = t_params['K']
        self.m = t_params['m']
        self.T = t_params['T']
        self.mlp = t_params['mlp']

        # create the encoders and clone(q -> k)
        self.encoder_q = load_model(self.base_encoder, self.res, self.dim, self.mlp, trainable=True)
        self.encoder_k = load_model(self.base_encoder, self.res, self.dim, self.mlp, trainable=False)
        for qw, kw in zip(self.encoder_q.weights, self.encoder_k.weights):
            kw.assign(qw)

        # create the queue
        self.queue, self.queue_ptr = self._setup_queue()
        return

    def _setup_queue(self):
        queue_ptr_init = tf.zeros(shape=[], dtype=tf.int64)
        queue_init = tf.math.l2_normalize(tf.random.normal([self.K, self.dim]), axis=1)
        queue = tf.Variable(queue_init, trainable=False,
                            synchronization=tf.VariableSynchronization.ON_READ,
                            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        queue_ptr = tf.Variable(queue_ptr_init, trainable=False,
                                synchronization=tf.VariableSynchronization.ON_READ,
                                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        return queue, queue_ptr

    @tf.function
    def _momentum_update_key_encoder(self):
        for qw, kw in zip(self.encoder_q.weights, self.encoder_k.weights):
            assert qw.shape == kw.shape
            kw.assign(kw * self.m + qw * (1.0 - self.m))
        return

    def _dequeue_and_enqueue(self, keys):
        # need to implement
        return

    def _batch_shuffle(self, x):
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
        batch_size = tf.shape(x)[0]

        all_idx = tf.range(self.global_batch_size)
        shuffle_idx = tf.random.shuffle(all_idx)

        this_start = replica_id * batch_size
        this_end = this_start + batch_size
        this_idx = shuffle_idx[this_start:this_end]

        return

    def _batch_unshuffle(self, x, idx_unshuffle):
        # need to implement
        return

    def forwards(self, inputs):
        # im_q: a batch of query images
        # im_k: a batch of key images
        im_q, im_k = inputs

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = tf.math.l2_normalize(q, axis=1)

        # compute key features
        self._momentum_update_key_encoder()  # update the key encoder

        # shuffle for making use of BN
        im_k, idx_unshuffle = self._batch_shuffle(im_k)

        k = self.encoder_k(im_k)  # keys: NxC
        k = tf.math.l2_normalize(k, axis=1)

        # undo shuffle
        k = self._batch_unshuffle(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = tf.expand_dims(tf.einsum('nc,nc->n', [q, k]), axis=-1)
        # negative logits: NxK
        l_neg = tf.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        return

    def train(self, dist_dataset):
        return


def get_moco_params(args):
    dim = 128
    K = 65536
    m = 0.999
    if args['moco_version'] == 1:
        moco_params = {
            'base_encoder': args['base_encoder'],
            'res': args['train_res'],
            'dim': dim,
            'K': K,
            'm': m,
            'T': 0.07,
            'mlp': False,
        }
    else:
        moco_params = {
            'base_encoder': args['base_encoder'],
            'res': args['train_res'],
            'dim': dim,
            'K': K,
            'm': m,
            'T': 0.2,
            'mlp': True,
        }
    return moco_params


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_tf_function', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--model_base_dir', default='./models', type=str)
    parser.add_argument('--images_dir', default='/mnt/vision-nas/junho/dataset/FFHQ', type=str)
    parser.add_argument('--base_encoder', default='resnet50', type=str)
    parser.add_argument('--train_res', default=224, type=int)
    parser.add_argument('--batch_size_per_replica', default=8, type=int)
    parser.add_argument('--moco_version', default=1, type=int)
    args = vars(parser.parse_args())

    if args['allow_memory_growth']:
        allow_memory_growth()

    # get MoCo specific parameters
    assert args['moco_version'] in [1, 2]
    moco_params = get_moco_params(args)
    name = 'MoCo_v1' if args['moco_version'] == 1 else 'MoCo_v2'

    # prepare distribute training
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = args['batch_size_per_replica'] * strategy.num_replicas_in_sync

    replica_context = tf.distribute.get_replica_context()
    replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
    print()

    # training parameters
    training_parameters = {
        # global params
        'use_tf_function': args['use_tf_function'],
        'model_base_dir': args['model_base_dir'],
        'train_res': args['train_res'],

        # network params
        **moco_params,

        # training params
        'opt': {'learning_rate': 0.002},
        'global_batch_size': global_batch_size,
        'n_replicas': strategy.num_replicas_in_sync,
        'epochs': 200,
    }

    # create moco trainer
    moco_v1 = MoCoTrainer(training_parameters, name=name)
    return


if __name__ == '__main__':
    main()
