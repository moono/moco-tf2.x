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
        self.use_tf_function = True
        self.m = 0.999
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size

        self.encoder_q = Linear()
        self.encoder_k = Linear()
        return

    @tf.function
    def _momentum_update_key_encoder(self):
        for qw, kw in zip(self.encoder_q.weights, self.encoder_k.weights):
            assert qw.shape == kw.shape
            kw.assign(kw * self.m + qw * (1.0 - self.m))
        return

    def forward_encoder_q(self, im_q):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = tf.math.l2_normalize(q, axis=1)
        return q

    def forward_encoder_k(self, all_images):
        # inputs are already shuffled,
        # just take shuffled data respect to their index will suffice
        all_idx = tf.range(self.global_batch_size)
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
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

    def train_step(self, inputs):
        im_q, im_k, shuffled_data = inputs

        im_q = tf.expand_dims(im_q, axis=1)

        with tf.GradientTape() as tape:
            # compute query features
            q = self.forward_encoder_q(im_q)  # queries: NxC

            # should momentum update here?
            self._momentum_update_key_encoder()

            # compute key features with shuffled data
            k = self.forward_encoder_k(shuffled_data)
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
        def dist_run_train_step(inputs):
            per_replica_result = strategy.experimental_run_v2(fn=self.train_step, args=(inputs,))
            return per_replica_result

        if self.use_tf_function:
            dist_run_train_step = tf.function(dist_run_train_step)

        for im_q, im_k in dist_dataset:
            # shuffle data (maybe shuffle index is not needed)
            shuffled_im_k, shuffled_idx_im_k = self._batch_shuffle(im_k, strategy)

            out = dist_run_train_step((im_q, im_k, shuffled_im_k))

        return


def main():
    # prepare distribute training
    strategy = tf.distribute.MirroredStrategy()

    batch_size = 4
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    data_q = tf.range(100, 100 + global_batch_size * 5)
    data_k = tf.range(200, 200 + global_batch_size * 5)
    dataset = tf.data.Dataset.from_tensor_slices((data_q, data_k)).batch(global_batch_size)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    # create moco trainer
    with strategy.scope():
        moco_trainer = MoCoTrainer(batch_size, global_batch_size)

        print('Training...')
        moco_trainer.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
