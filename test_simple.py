import tensorflow as tf


class Linear(tf.keras.models.Model):
    def __init__(self, dim, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.m = 0.999
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
        self.K = 16
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

        # create optimizer
        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.SGD(self.learning_rate, momentum=0.9, nesterov=True)
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

        # get this replica's im_k
        im_k = tf.gather(all_im_k, indices=this_idx)

        # compute query features
        k = self.encoder_k(im_k)  # keys: NxC
        k = tf.math.l2_normalize(k, axis=1)
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

            # should momentum update here?
            # ... not working?

            # compute logits: Einstein sum is more intuitive
            l_pos = tf.expand_dims(tf.einsum('nc,nc->n', q, k), axis=-1)    # positive logits: Nx1
            l_neg = tf.einsum('nc,kc->nk', q, self.queue)                   # negative logits: NxK
            logits = tf.concat([l_pos, l_neg], axis=1)                      # Nx(K+1)

            labels = tf.zeros(self.batch_size, dtype=tf.int64)  # [N, ]
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)     # [N, ]

            # scale losses
            loss = tf.reduce_sum(loss) * (1.0 / self.global_batch_size)

        t_var = self.encoder_q.trainable_variables
        gradients = tape.gradient(loss, t_var)
        self.optimizer.apply_gradients(zip(gradients, t_var))
        return loss

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

    def _dequeue_and_enqueue(self, keys):
        # keys: [GN, C]
        end_queue_ptr = self.queue_ptr + self.global_batch_size
        indices = tf.range(self.queue_ptr, end_queue_ptr, dtype=tf.int64)   # [GN,  ]
        indices = tf.expand_dims(indices, axis=1)                           # [GN, 1]

        tf.print(f'keys: {keys}')
        tf.print(f'indices: {indices}')

        updated_queue = tf.tensor_scatter_nd_update(tensor=self.queue, indices=indices, updates=keys)
        updated_queue_ptr = end_queue_ptr % self.K

        tf.print(f'updated_queue: {updated_queue}')
        tf.print(f'updated_queue_ptr: {updated_queue_ptr}')
        self.queue.assign(updated_queue)
        self.queue_ptr.assign(updated_queue_ptr)
        return

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

            # run on encoder_k to collect shuffled keys
            k_shuffled = dist_run_key_encoder((shuffled_k, ))

            # unshuffle and merge all
            k_unshuffled = self._batch_unshuffle(k_shuffled, shuffled_idx, strategy)

            # train step: update queue encoder
            losses = dist_run_train_step((im_q, k_unshuffled))

            # update key encoder
            self.encoder_k.momentum_update(self.encoder_q)

            # update queue
            self._dequeue_and_enqueue(k_unshuffled)

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

# keys
# [[0.4036725  0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.68100476]
#  [0.40367246 0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.68100476]
#  [0.40367252 0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.68100476]
#  [0.4036725  0.22994013 0.5660464  0.68100476]]
#
# indices
# [[0] [1] [2] [3] [4] [5] [6] [7]]
# updated_queue:
# [[0.4036725  0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.68100476]
#  [0.40367246 0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.68100476]
#  [0.40367252 0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.68100476]
#  [0.4036725  0.22994013 0.5660464  0.68100476]
#  [-0.6740818  -0.7073098   0.11813977 -0.1771143 ]
#  [-0.3537059   0.6475618  -0.45937663  0.4944985 ]
#  [-0.33242273 -0.6241452   0.7024059   0.08101729]
#  [-0.70300645 -0.2730486  -0.4551356  -0.47336876]
#  [-0.60927194 -0.03494551 -0.7881595  -0.0798188 ]
#  [ 0.6838572   0.5050782  -0.2435258  -0.46683028]
#  [-0.564013    0.4755977   0.30972537 -0.59980536]
#  [-0.60397404 -0.43192804 -0.66061807 -0.11062176]]
#
# keys
# [[0.4036725  0.22994016 0.5660464  0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.6810048 ]
#  [0.40367252 0.22994016 0.5660464  0.6810048 ]
#  [0.4036725  0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994013 0.56604636 0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.68100476]
#  [0.4036725  0.22994013 0.5660464  0.68100476]
#  [0.40367252 0.22994015 0.5660464  0.6810047 ]]
#
# indices
# [[ 8] [ 9] [10] [11] [12] [13] [14] [15]]
#
# updated_queue:
# [[0.4036725  0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.68100476]
#  [0.40367246 0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.68100476]
#  [0.40367252 0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.68100476]
#  [0.4036725  0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994016 0.5660464  0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.6810048 ]
#  [0.40367252 0.22994016 0.5660464  0.6810048 ]
#  [0.4036725  0.22994013 0.5660464  0.68100476]
#  [0.4036725  0.22994013 0.56604636 0.68100476]
#  [0.4036725  0.22994015 0.5660464  0.68100476]
#  [0.4036725  0.22994013 0.5660464  0.68100476]
#  [0.40367252 0.22994015 0.5660464  0.6810047 ]]
