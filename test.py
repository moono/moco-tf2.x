import tensorflow as tf


class MoCoTrainer(object):
    def __init__(self, num_replicas_in_sync):
        self.batch_size = 8
        self.global_batch_size = self.batch_size * num_replicas_in_sync
        return

    def _batch_shuffle(self, shuffle_idx):
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

        this_start = replica_id * self.batch_size
        this_end = this_start + self.batch_size
        this_idx = shuffle_idx[this_start:this_end]

        tf.print(f'Replica {replica_id}: {this_idx}')
        return

    def train(self, strategy):
        all_idx = tf.range(self.global_batch_size)
        shuffle_idx = tf.random.shuffle(all_idx)

        strategy.experimental_run_v2(fn=self._batch_shuffle, args=(shuffle_idx,))
        return


def main():
    # prepare distribute training
    strategy = tf.distribute.MirroredStrategy()

    # create moco trainer
    with strategy.scope():
        moco_trainer = MoCoTrainer(strategy.num_replicas_in_sync)

        print('Training...')
        moco_trainer.train(strategy)
    return


if __name__ == '__main__':
    main()
