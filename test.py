import tensorflow as tf


class Linear(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Linear, self).__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(4, use_bias=True)
        return

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.dense(x)
        return tf.identity(x)


class MoCoTrainer(object):
    def __init__(self, batch_size, global_batch_size):
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size

        self.encoder_q = Linear()
        self.encoder_k = Linear()
        return

    # def _batch_shuffle(self, shuffle_idx):
    #     replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
    #
    #     this_start = replica_id * self.batch_size
    #     this_end = this_start + self.batch_size
    #     this_idx = shuffle_idx[this_start:this_end]
    #
    #     tf.print(f'Replica {replica_id}: {this_idx}')
    #     # tf.print(f'Replica {replica_id}: {d}')
    #     return this_idx

    def _batch_shuffle(self, all_gathered, strategy):
        # convert to tf.Tensor
        data = tf.concat(strategy.experimental_local_results(all_gathered), axis=0)

        # create shuffled index for global batch size
        all_idx = tf.range(self.global_batch_size)
        shuffled_idx = tf.random.shuffle(all_idx)

        # shuffle
        shuffled_data = tf.gather(data, indices=shuffled_idx)
        return shuffled_data, shuffled_idx

    def forward_encoder_q(self, im_q):
        # compute query features
        im_q = tf.expand_dims(im_q, axis=1)
        q = self.encoder_q(im_q)    # queries: NxC
        q = tf.math.l2_normalize(q, axis=1)
        return q

    def forward_encoder_k(self, all_images):
        # inputs are already shuffled,
        # just take shuffled data respect to their index will suffice
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

        all_idx = tf.range(self.global_batch_size)
        this_start = replica_id * self.batch_size
        this_end = this_start + self.batch_size
        this_idx = all_idx[this_start:this_end]

        im_k = tf.gather(all_images, indices=this_idx)

        # run encoder_k
        im_k = tf.expand_dims(im_k, axis=1)
        k = self.encoder_k(im_k)  # keys: NxC
        k = tf.math.l2_normalize(k, axis=1)

        tf.print(f'Run {replica_id}: {im_k}')
        return k

    def train(self, dist_dataset, strategy):
        def dist_run_query_encoder(data):
            per_replica_result = strategy.experimental_run_v2(fn=self.forward_encoder_q, args=(data,))
            return per_replica_result

        def dist_run_key_encoder(data):
            per_replica_result = strategy.experimental_run_v2(fn=self.forward_encoder_k, args=(data, ))
            return per_replica_result

        for d in dist_dataset:
            qs = dist_run_query_encoder(d)

            # shuffle batch to fool batch normalization
            shuffled_data, shuffled_idx = self._batch_shuffle(d, strategy)
            tf.print(shuffled_data)     # [103 105 101 ... 107 102 100]
            tf.print(shuffled_idx)      # [3 5 1 ... 7 2 0]

            ks = dist_run_key_encoder(shuffled_data)
            # Run 0: [103 105 101 104]
            # Run 1: [106 107 102 100]

            tf.print(strategy.experimental_local_results(ks))   # ([103 105 101 104], [106 107 102 100])

            # all_shuffled_idx = strategy.experimental_local_results(dist_batch_shuffle())
            # all_shuffled_idx = tf.concat(all_shuffled_idx, axis=0)
            # tf.print(all_shuffled_idx)
            #
            # shuffled_data = tf.gather(d, indices=all_shuffled_idx)
            # tf.print(strategy.experimental_local_results(shuffled_data))
        return


def main():
    # prepare distribute training
    strategy = tf.distribute.MirroredStrategy()

    batch_size = 4
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    data = tf.range(100, 100 + global_batch_size * 10)
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(global_batch_size)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    # create moco trainer
    with strategy.scope():
        moco_trainer = MoCoTrainer(batch_size, global_batch_size)

        print('Training...')
        moco_trainer.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
