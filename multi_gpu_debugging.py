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
    dataset_n_images = 100000000
    batch_size_per_replica = 256
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # training parameters
    training_parameters = {
        # global params
        'name': 'debugging',
        'use_tf_function': False,
        'model_base_dir': './models',

        # moco params
        'moco_version': 1,
        'base_encoder': 'linear',
        'network_params': {'input_shape': [1], 'dim': 4, 'K': 1024, 'm': 0.999, 'T': 0.07, 'w_decay': 0.0},
        'learning_rate': {'schedule': 'step', 'initial_lr': 0.03, 'lr_decay': 0.1, 'lr_decay_boundaries': [120, 160]},

        # training params
        'n_images': dataset_n_images,
        'epochs': epochs,
        'batch_size_per_replica': batch_size_per_replica,
        'global_batch_size': global_batch_size,
    }

    # print current details
    pp(training_parameters)

    # load dataset
    dataset = get_toy_dataset(global_batch_size, dataset_n_images, epochs)

    # create MoCo instance
    # moco = MoCo(training_parameters, strategy)
    with strategy.scope():
        # distribute dataset
        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        # start training
        print('Training...')
        moco = MoCo(training_parameters)
        moco.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
