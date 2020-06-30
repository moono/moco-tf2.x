import tensorflow as tf


def some_random_augmentation(data):
    # just simulation
    return tf.identity(data)


def two_crops(data):
    data_q = tf.identity(data)
    data_k = tf.identity(data)

    data_q = some_random_augmentation(data_q)
    data_k = some_random_augmentation(data_k)
    return data_q, data_k


def get_dataset(global_batch_size):
    # Total data length: T
    # global batch size: N
    T = global_batch_size * 5
    data = tf.range(100, 100 + T)           # [T,  ]
    data = tf.expand_dims(data, axis=1)     # [T, 1]

    dataset = tf.data.Dataset.from_tensor_slices((data, ))
    dataset = dataset.map(map_func=lambda d: two_crops(d), num_parallel_calls=8)
    dataset = dataset.batch(global_batch_size)
    return dataset
