import tensorflow as tf


class MoCoTrainer(object):
    def __init__(self, batch_size, global_batch_size):
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size
        return

    def _batch_shuffle(self, d, shuffle_idx):
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

        this_start = replica_id * self.batch_size
        this_end = this_start + self.batch_size
        this_idx = shuffle_idx[this_start:this_end]

        tf.print(f'Replica {replica_id}: {this_idx}')
        return

    def train(self, dist_dataset, strategy):

        for d in dist_dataset:

            all_idx = tf.range(self.global_batch_size)
            shuffle_idx = tf.random.shuffle(all_idx)

            strategy.experimental_run_v2(fn=self._batch_shuffle, args=(d, shuffle_idx,))
        return


def main():
    # prepare distribute training
    strategy = tf.distribute.MirroredStrategy()

    batch_size = 8
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    data = tf.range(100, 260)
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
