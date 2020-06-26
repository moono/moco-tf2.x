import tensorflow as tf


class Linear(tf.keras.models.Model):
    def __init__(self, dim, **kwargs):
        super(Linear, self).__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(dim, use_bias=True)
        return

    @tf.function
    def momentum_update(self, src_net):
        for qw, kw in zip(src_net.weights, self.weights):
            assert qw.shape == kw.shape
            kw.assign(kw * self.m + qw * (1.0 - self.m))
        return

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.dense(x)
        return tf.identity(x)


class MoCoTrainer(object):
    def __init__(self, batch_size, global_batch_size, dim):
        self.use_tf_function = False
        self.K = 1024
        self.m = 0.999
        self.dim = dim
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size

        # create the encoders and clone(q -> k)
        self.encoder_q = Linear(self.dim)
        self.encoder_k = Linear(self.dim)
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

    def forward_encoder_k(self, inputs):
        all_im_k = inputs[0]

        # inputs are already shuffled, just take shuffled data respect to their index will suffice
        all_idx = tf.range(self.global_batch_size)
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
        this_start = replica_id * self.batch_size
        this_end = this_start + self.batch_size
        this_idx = all_idx[this_start:this_end]

        im_k = tf.gather(all_im_k, indices=this_idx)
        # tf.print(f'{replica_id}: {im_k}')

        # compute query features
        k = self.encoder_k(im_k)  # keys: NxC
        k = tf.math.l2_normalize(k, axis=1)

        # tf.print(f'{replica_id}: {k}')
        return k

    def train_step(self, inputs):
        im_q, all_k = inputs

        # get appropriate ks
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

            # # should momentum update here?
            # self._momentum_update_key_encoder()

            tf.print(f'{replica_id}, k: {k}')
            tf.print(f'{replica_id}, q: {q}')
            print(f'{replica_id}, k: {k}')
            print(f'{replica_id}, q: {q}')

            # # compute logits: Einstein sum is more intuitive
            # l_pos = tf.expand_dims(tf.einsum('nc,nc->n', [q, k]), axis=-1)  # positive logits: Nx1
            # l_neg = tf.einsum('nc,kc->nk', [q, self.queue])                 # negative logits: NxK
            # logits = tf.concat([l_pos, l_neg], axis=1)                      # Nx(K+1)
            # tf.print(f'l_pos: {l_pos}')
            # tf.print(f'l_neg: {l_neg}')
            # tf.print(f'logits: {logits}')
        return

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

        # tf.print(f'shuffled_k: {all_shuffled}')
        unshuffled_data = tf.scatter_nd(indices=shuffled_idx, updates=all_shuffled, shape=output_shape)
        return unshuffled_data

    def train(self, dist_dataset, strategy):
        def dist_run_key_encoder(inputs):
            per_replica_result = strategy.experimental_run_v2(fn=self.forward_encoder_k, args=(inputs, ))
            return per_replica_result

        def dist_run_train_step(inputs):
            per_replica_result = strategy.experimental_run_v2(fn=self.train_step, args=(inputs,))
            return per_replica_result

        if self.use_tf_function:
            dist_run_key_encoder = tf.function(dist_run_key_encoder)
            dist_run_train_step = tf.function(dist_run_train_step)

        for im_q, im_k in dist_dataset:
            # shuffle data on global scale
            # shuffled_k: [GN, ]
            # shuffled_idx: [GN, ]
            shuffled_k, shuffled_idx = self._batch_shuffle(im_k, strategy)

            # tf.print(f'shuffled_k: {shuffled_k}')
            # tf.print(f'shuffled_idx: {shuffled_idx}')

            # run on encoder_k to collect shuffled keys
            k_shuffled = dist_run_key_encoder((shuffled_k, ))

            # unshuffle and merge all
            k_unshuffled = self._batch_unshuffle(k_shuffled, shuffled_idx, strategy)
            # tf.print(f'k_unshuffled: {k_unshuffled}')

            out = dist_run_train_step((im_q, k_unshuffled))

        return


def some_random_augmentation(data):
    # just simulation
    return tf.identity(data)


def two_crops(data):
    data_q = tf.identity(data)
    data_k = tf.identity(data)

    data_q = some_random_augmentation(data_q)
    data_k = some_random_augmentation(data_k)
    return data_q, data_k


# def shuffle_fn(data_q, data_k, batch_size):
#     # data_q: [N, 1]
#     # data_k: [N, 1]
#
#     # create shuffled index for global batch size
#     all_idx = tf.range(batch_size)
#     shuffled_idx = tf.random.shuffle(all_idx)
#
#     # shuffle
#     shuffled_data_k = tf.gather(data_k, indices=shuffled_idx)
#
#     shuffled_idx = tf.expand_dims(shuffled_idx, axis=1)
#     return data_q, data_k, shuffled_data_k, shuffled_idx


def get_toy_dataset(global_batch_size):
    # Total data length: T
    # global batch size: N
    T = global_batch_size * 5
    data = tf.range(100, 100 + T)           # [T,  ]
    data = tf.expand_dims(data, axis=1)     # [T, 1]

    dataset = tf.data.Dataset.from_tensor_slices((data, ))
    dataset = dataset.map(map_func=lambda d: two_crops(d), num_parallel_calls=8)
    dataset = dataset.batch(global_batch_size)
    # dataset = dataset.map(map_func=lambda q, k: shuffle_fn(q, k, global_batch_size), num_parallel_calls=8)
    return dataset


def main():
    # shuffled_data = tf.constant([[104], [103], [105], [106], [100], [101], [107], [102]])
    # # shuffled_data = tf.squeeze(shuffled_data, axis=1)
    # shuffled_indx = tf.constant([[4], [3], [5], [6], [0], [1], [7], [2]])
    # shape = tf.constant([8, 1])
    #
    # print(shuffled_data.shape)
    # print(shuffled_indx.shape)
    # print(shape.shape)
    #
    # unshuffled_data = tf.scatter_nd(indices=shuffled_indx, updates=shuffled_data, shape=shape)
    # print(unshuffled_data)

    # batch_size = 4
    # global_batch_size = batch_size * 2
    # dataset = get_toy_dataset(global_batch_size)
    #
    # for q, k in dataset:
    #     # tf.print(f'q: {q}')
    #     # tf.print(f'k: {k}')
    #     # tf.print(f'sk: {sk}')
    #     # tf.print(f'sk_idx: {sk_idx}')
    #     print(q.shape)
    #     print(k.shape)
    #     print()

    # prepare distribute training
    strategy = tf.distribute.MirroredStrategy()

    dim = 4
    batch_size = 4
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    dataset = get_toy_dataset(global_batch_size)

    # create moco trainer
    with strategy.scope():
        moco_trainer = MoCoTrainer(batch_size, global_batch_size, dim)

        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        print('Training...')
        moco_trainer.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()

# shuffled_idx: [1 6 2 4 7 5 3 0]
# 0
# [[ 0.41266525  0.54188955 -0.45740324  0.5717039 ]
#  [ 0.41266528  0.54188955 -0.45740327  0.5717039 ]
#  [ 0.41266522  0.54188955 -0.45740327  0.5717039 ]
#  [ 0.41266525  0.54188955 -0.45740327  0.5717039 ]]
# 1
# [[ 0.41266522  0.5418895  -0.45740324  0.5717039 ]
#  [ 0.41266525  0.5418896  -0.45740327  0.571704  ]
#  [ 0.41266525  0.54188955 -0.45740327  0.57170385]
#  [ 0.41266525  0.54188955 -0.45740324  0.57170385]]
#
# merged_shuffled
# [[ 0.41266525  0.54188955 -0.45740324  0.5717039 ]
#  [ 0.41266528  0.54188955 -0.45740327  0.5717039 ]
#  [ 0.41266522  0.54188955 -0.45740327  0.5717039 ]
#  [ 0.41266525  0.54188955 -0.45740327  0.5717039 ]
#  [ 0.41266522  0.5418895  -0.45740324  0.5717039 ]
#  [ 0.41266525  0.5418896  -0.45740327  0.571704  ]
#  [ 0.41266525  0.54188955 -0.45740327  0.57170385]
#  [ 0.41266525  0.54188955 -0.45740324  0.57170385]]
#
# merged_unshuffled
# [[ 0.41266525  0.54188955 -0.45740324  0.57170385]
#  [ 0.41266525  0.54188955 -0.45740324  0.5717039 ]
#  [ 0.41266522  0.54188955 -0.45740327  0.5717039 ]
#  [ 0.41266525  0.54188955 -0.45740327  0.57170385]
#  [ 0.41266525  0.54188955 -0.45740327  0.5717039 ]
#  [ 0.41266525  0.5418896  -0.45740327  0.571704  ]
#  [ 0.41266528  0.54188955 -0.45740327  0.5717039 ]
#  [ 0.41266522  0.5418895  -0.45740324  0.5717039 ]]
