import tensorflow as tf


class Linear(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Linear, self).__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(4, use_bias=True)
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
    def __init__(self, batch_size, global_batch_size):
        self.use_tf_function = False
        self.m = 0.999
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size

        self.encoder_q = Linear()
        self.encoder_k = Linear()
        return

    def forward_encoder_q(self, im_q):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = tf.math.l2_normalize(q, axis=1)
        return q

    def forward_encoder_k(self, inputs):
        im_k = inputs[0]
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

        tf.print(f'{replica_id}: {im_k}')

        # compute query features
        k = self.encoder_k(im_k)  # keys: NxC
        k = tf.math.l2_normalize(k, axis=1)

        tf.print(f'{replica_id}: {k}')

        # # inputs are already shuffled,
        # # just take shuffled data respect to their index will suffice
        # all_idx = tf.range(self.global_batch_size)
        # replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
        # this_start = replica_id * self.batch_size
        # this_end = this_start + self.batch_size
        # this_idx = all_idx[this_start:this_end]
        #
        # im_k = tf.gather(all_images, indices=this_idx)
        #
        # # run encoder_k
        # im_k = tf.expand_dims(im_k, axis=1)
        # k = self.encoder_k(im_k)  # keys: NxC
        # k = tf.math.l2_normalize(k, axis=1)
        #
        # tf.print(f'Run {replica_id}: {im_k}')
        return k

    def train_step(self, inputs):
        im_q, im_k, shuffled_im_k = inputs

        im_q = tf.expand_dims(im_q, axis=1)

        with tf.GradientTape() as tape:
            # compute query features
            q = self.forward_encoder_q(im_q)  # queries: NxC

            # # should momentum update here?
            # self._momentum_update_key_encoder()

            # compute key features with shuffled data
            k = self.forward_encoder_k(shuffled_im_k)
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

        for q, k in dist_dataset:
            # shuffle data on global scale (maybe shuffle index is not needed)
            shuffled_k, shuffled_idx = self._batch_shuffle(k, strategy)

            tf.print(f'shuffled_k: {shuffled_k}')
            tf.print(f'shuffled_idx: {shuffled_idx}')

            # run on encoder_k to collect shuffled keys
            k_shuffled = dist_run_key_encoder((shuffled_k, ))

            # merge all
            sk_merged = tf.concat(strategy.experimental_local_results(k_shuffled), axis=0)
            tf.print(f'merged: {sk_merged}')

            # out = dist_run_train_step((q, k, sk))

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

    batch_size = 4
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    dataset = get_toy_dataset(global_batch_size)

    # create moco trainer
    with strategy.scope():
        moco_trainer = MoCoTrainer(batch_size, global_batch_size)

        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        print('Training...')
        moco_trainer.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
