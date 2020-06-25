# ([100 101 102 ... 105 106 107], [108 109 110 ... 113 114 115])
# Replica 0: [ 7  4  3  8  0 13  1 15]
# Replica 0: [100 101 102 103 104 105 106 107]
# Replica 1: [10 12  6  9  5  2 14 11]
# Replica 1: [108 109 110 111 112 113 114 115]

# ([116 117 118 ... 121 122 123], [124 125 126 ... 129 130 131])
# Replica 0: [14  7 15  2 11  5  0 10]
# Replica 0: [116 117 118 119 120 121 122 123]
# Replica 1: [ 1 13  3  9  4  6 12  8]
# Replica 1: [124 125 126 127 128 129 130 131]

# ([132 133 134 ... 137 138 139], [140 141 142 ... 145 146 147])
# Replica 0: [14  7  6  9 15  2  1 11]
# Replica 0: [132 133 134 135 136 137 138 139]
# Replica 1: [10 13 12  8  0  4  3  5]
# Replica 1: [140 141 142 143 144 145 146 147]

# ([148 149 150 ... 153 154 155], [156 157 158 ... 161 162 163])
# Replica 0: [ 4  0  8  7 15  2 12 10]
# Replica 0: [148 149 150 151 152 153 154 155]
# Replica 1: [ 3 13  6  9 11 14  1  5]
# Replica 1: [156 157 158 159 160 161 162 163]

# ([164 165 166 ... 169 170 171], [172 173 174 ... 177 178 179])
# Replica 0: [10  5 14  6 13 12  9 11]
# Replica 0: [164 165 166 167 168 169 170 171]
# Replica 1: [ 3  7  1 15  4  2  8  0]
# Replica 1: [172 173 174 175 176 177 178 179]

# ([180 181 182 ... 185 186 187], [188 189 190 ... 193 194 195])
# Replica 0: [13  6  2 10  0  1 15  8]
# Replica 0: [180 181 182 183 184 185 186 187]
# Replica 1: [ 3 11  9 12  5  7 14  4]
# Replica 1: [188 189 190 191 192 193 194 195]

# ([196 197 198 ... 201 202 203], [204 205 206 ... 209 210 211])
# Replica 0: [ 9 14  6 13 12  1  2  4]
# Replica 0: [196 197 198 199 200 201 202 203]
# Replica 1: [ 0  7 11  3 10 15  8  5]
# Replica 1: [204 205 206 207 208 209 210 211]

# ([212 213 214 ... 217 218 219], [220 221 222 ... 225 226 227])
# Replica 0: [ 4  6  2  1 10  0  5 11]
# Replica 0: [212 213 214 215 216 217 218 219]
# Replica 1: [ 7 13  8  9 12 14  3 15]
# Replica 1: [220 221 222 223 224 225 226 227]

# ([228 229 230 ... 233 234 235], [236 237 238 ... 241 242 243])
# Replica 0: [13  3  5 12 10  9  0  1]
# Replica 0: [228 229 230 231 232 233 234 235]
# Replica 1: [ 7 14  6 11  8  2 15  4]
# Replica 1: [236 237 238 239 240 241 242 243]

# ([244 245 246 ... 249 250 251], [252 253 254 ... 257 258 259])
# Replica 0: [ 2  0  9  1 15 11  4 14]
# Replica 0: [244 245 246 247 248 249 250 251]
# Replica 1: [10  3 12  6 13  8  5  7]
# Replica 1: [252 253 254 255 256 257 258 259]

import tensorflow as tf


class MoCoTrainer(object):
    def __init__(self, batch_size, global_batch_size):
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size
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

    def per_replica_run_test(self, d, shuffle_idx):
        replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

        this_start = replica_id * self.batch_size
        this_end = this_start + self.batch_size
        this_idx = shuffle_idx[this_start:this_end]

        shuffled_data = tf.gather(d, indices=this_idx)

        tf.print(f'Replica {replica_id}: {this_idx}')
        tf.print(f'Run {replica_id}: {shuffled_data}')
        return

    def train(self, dist_dataset, strategy):
        def dist_run_shuffled_data(data, idx):
            per_replica_result = strategy.experimental_run_v2(fn=self.per_replica_run_test, args=(data, idx))
            return per_replica_result

        for d in dist_dataset:
            # tf.print(strategy.experimental_local_results(d))
            shuffled_data, shuffled_idx = self._batch_shuffle(d, strategy)

            tf.print(shuffled_data)
            tf.print(shuffled_idx)

            result = dist_run_shuffled_data(shuffled_data, shuffled_idx)
            tf.print(strategy.experimental_local_results(result))

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
