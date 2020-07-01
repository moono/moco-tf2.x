import tensorflow as tf

from pprint import pprint as pp
from misc.tf_utils import split_gpu_for_testing
from datasets.toy_dataset import get_dataset as get_toy_dataset

from moco import MoCo


def main():
    # GPU environment settings
    split_gpu_for_testing()

    # prepare distribute training
    epochs = 200
    dataset_n_images = 100000
    batch_size_per_replica = 32
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # training parameters
    training_parameters = {
        # global params
        'name': 'debugging',
        'use_tf_function': True,
        'model_base_dir': './models',

        # moco params
        'base_encoder': 'linear',
        'network_params': {'input_shape': [1], 'dim': 4, 'K': 16, 'm': 0.999, 'T': 0.07},

        # training params
        'n_images': dataset_n_images,
        'epochs': epochs,
        'weight_decay': 0.0001,
        'initial_lr': 0.001,
        'lr_decay': 0.1,
        'lr_decay_boundaries': None,
        'batch_size_per_replica': batch_size_per_replica,
        'global_batch_size': global_batch_size,
    }

    # print current details
    pp(training_parameters)

    # load dataset
    dataset = get_toy_dataset(global_batch_size, dataset_n_images, epochs)

    with strategy.scope():
        # create MoCo instance
        moco = MoCo(training_parameters)

        # distribute dataset
        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        # start training
        print('Training...')
        moco.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
