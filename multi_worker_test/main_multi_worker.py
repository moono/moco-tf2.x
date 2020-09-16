import os
import json
import argparse

from subprocess import call
from misc.utils import str_to_bool


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--port_start', default=30000, type=int)
    parser.add_argument('--index', default=0, type=int)

    parser.add_argument('--allow_memory_growth', type=str_to_bool, nargs='?', const=True, default=False)
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

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': [f'worker{ii}:{args["port_start"]+ii}' for ii in range(args['n_workers'])],
        },
        'task': {'type': 'worker', 'index': args['index']}
    })

    call(['python', 'train_multi_worker.py',
          '--allow_memory_growth', str(args['allow_memory_growth']),
          '--use_tf_function', str(args['use_tf_function']),
          '--name', str(args['name']),
          '--tfds_data_dir', str(args['tfds_data_dir']),
          '--model_base_dir', str(args['model_base_dir']),
          '--moco_version', str(args['moco_version']),
          '--aug_op', str(args['aug_op']),
          '--batch_size_per_replica', str(args['batch_size_per_replica']),
          '--epochs', str(args['epochs']),
          '--initial_lr', str(args['initial_lr'])
          ])
    return


if __name__ == '__main__':
    main()
