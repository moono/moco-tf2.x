import argparse
import tensorflow as tf

from pprint import pprint as pp
from misc.utils import str_to_bool
from misc.tf_utils import allow_memory_growth, split_gpu_for_testing
from datasets.imagenet import get_dataset as get_imagenet_dataset

from moco import MoCo


def moco_parameter_by_version(version):
    if version == 1:
        moco_params = {
            'base_encoder': 'resnet50',
            'network_params': {
                'input_shape': [224, 224, 3], 'dim': 128, 'K': 65536, 'm': 0.999, 'T': 0.07, 'mlp': False
            },
        }
    else:
        moco_params = {
            'base_encoder': 'resnet50',
            'network_params': {
                'input_shape': [224, 224, 3], 'dim': 128, 'K': 65536, 'm': 0.999, 'T': 0.2, 'mlp': True
            },
        }
    return moco_params


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--debug_split_gpu', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_tf_function', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--name', default='moco_v1_imagenet', type=str)
    parser.add_argument('--tfds_data_dir', default='/mnt/vision-nas/data-sets/tensorflow_datasets', type=str)
    parser.add_argument('--model_base_dir', default='./models', type=str)
    parser.add_argument('--moco_version', default=1, type=int)
    parser.add_argument('--batch_size_per_replica', default=2, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    args = vars(parser.parse_args())

    # check args
    assert args['moco_version'] in [1, 2]

    # GPU environment settings
    if args['allow_memory_growth']:
        allow_memory_growth()
    if args['debug_split_gpu']:
        split_gpu_for_testing()

    # default values
    dataset_n_images = {'train': 1281167, 'validation': 50000}
    initial_lr = 0.003
    lr_decay_factor = 0.1
    lr_decay_boundaries = [120, 160]
    weight_decay_factor = 0.0001

    # get MoCo parameters
    moco_params = moco_parameter_by_version(args['moco_version'])

    # prepare distribute training
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = args['batch_size_per_replica'] * strategy.num_replicas_in_sync

    # training parameters
    training_parameters = {
        # global params
        'name': args['name'],
        'use_tf_function': args['use_tf_function'],
        'model_base_dir': args['model_base_dir'],

        # moco params
        **moco_params,

        # training params
        'n_images': dataset_n_images['train'],
        'epochs': args['epochs'],
        'weight_decay': weight_decay_factor,
        'initial_lr': initial_lr,
        'lr_decay': lr_decay_factor,
        'lr_decay_boundaries': lr_decay_boundaries,
        'batch_size_per_replica': args['batch_size_per_replica'],
        'global_batch_size': global_batch_size,
    }

    # print current details
    pp(training_parameters)

    # load dataset
    dataset = get_imagenet_dataset(
        args['tfds_data_dir'], is_training=True, res=224, moco_ver=args['moco_version'],
        batch_size=global_batch_size, epochs=args['epochs'])

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
