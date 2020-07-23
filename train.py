import argparse
import tensorflow as tf

from pprint import pprint as pp
from misc.utils import str_to_bool
from misc.tf_utils import check_tf_version, allow_memory_growth, split_gpu_for_testing
from datasets.imagenet import get_dataset, augmentation_v1, augmentation_v2

from moco import MoCo


def main():
    # check tensorflow version
    cur_tf_ver = check_tf_version()

    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--debug_split_gpu', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_tf_function', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--name', default='test', type=str)
    parser.add_argument('--tfds_data_dir', default='/mnt/vision-nas/data-sets/tensorflow_datasets', type=str)
    parser.add_argument('--model_base_dir', default='./models', type=str)
    parser.add_argument('--moco_version', default=2, type=int)
    parser.add_argument('--aug_op', default='GPU', type=str)
    parser.add_argument('--batch_size_per_replica', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--initial_lr', default=0.03, type=float)
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
    w_decay = 0.0001
    if args['moco_version'] == 1:
        moco_params = {
            'base_encoder': 'resnet50',
            'network_params': {
                'input_shape': [224, 224, 3],
                'dim': 128,
                'K': 65536,
                'm': 0.999,
                'T': 0.07,
                'mlp': False,
                'w_decay': w_decay,
            },
            'learning_rate': {
                'schedule': 'step',
                'initial_lr': args['initial_lr'],
                'lr_decay': 0.1,
                'lr_decay_boundaries': [120, 160],
            }
        }
    else:
        moco_params = {
            'base_encoder': 'resnet50',
            'network_params': {
                'input_shape': [224, 224, 3],
                'dim': 128,
                'K': 65536,
                'm': 0.999,
                'T': 0.2,
                'mlp': True,
                'w_decay': w_decay,
            },
            'learning_rate': {
                'schedule': 'cos',
                'initial_lr': args['initial_lr'],
            }
        }
    res = moco_params['network_params']['input_shape'][0]

    # select proper image augmentation function
    if args['aug_op'] == 'GPU':
        aug_fn = augmentation_v1 if args['moco_version'] == 1 else augmentation_v2
    else:
        aug_fn = None

    # prepare distribute training
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = args['batch_size_per_replica'] * strategy.num_replicas_in_sync

    # training parameters
    training_parameters = {
        # global params
        'cur_tf_ver': cur_tf_ver,
        'name': f'{args["name"]}_moco_v{args["moco_version"]}',
        'use_tf_function': args['use_tf_function'],
        'model_base_dir': args['model_base_dir'],
        'aug_fn': aug_fn,
        'res': res,

        # moco params
        'moco_version': args['moco_version'],
        **moco_params,

        # training params
        'n_images': dataset_n_images['train'] * args['epochs'],
        'epochs': args['epochs'],
        'batch_size_per_replica': args['batch_size_per_replica'],
        'global_batch_size': global_batch_size,
    }

    # print current details
    pp(training_parameters)

    # load dataset
    dataset = get_dataset(args['tfds_data_dir'], is_training=True, res=res, moco_ver=args['moco_version'],
                          aug_op=args['aug_op'], batch_size=global_batch_size)

    # create MoCo instance
    moco = MoCo(training_parameters, strategy)

    with strategy.scope():
        # distribute dataset
        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        # start training
        print('Training...')
        moco.train(dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
