import os
import time
import argparse
import numpy as np
import tensorflow as tf

from pprint import pprint as pp
from misc.utils import str_to_bool
from misc.tf_utils import check_tf_version, allow_memory_growth, split_gpu_for_testing
from datasets.imagenet import get_dataset_lincls
from base_networks.load_model import load_model
from misc.learning_rate_schedule import StepDecay, CosineDecay


class LinearClassification(object):
    def __init__(self, t_params, strategy):
        # global parameters
        self.cur_tf_ver = t_params['cur_tf_ver']
        self.name = t_params['name']
        self.use_tf_function = t_params['use_tf_function']
        self.model_base_dir = t_params['model_base_dir']
        self.batch_size = t_params['batch_size_per_replica']
        self.global_batch_size = t_params['global_batch_size']
        self.epochs = t_params['epochs']
        self.n_images = t_params['n_images']
        self.max_steps = int(np.ceil(self.n_images / self.global_batch_size))
        self.epochs_per_step = self.global_batch_size / (self.n_images / self.epochs)
        self.reached_max_steps = False
        self.print_step = 100
        self.save_step = 500
        self.n_replica = self.global_batch_size // self.batch_size
        self.dist_loss_scaler = 1.0 / self.global_batch_size

        # network parameters
        self.base_encoder = t_params['base_encoder']

        # load pretrained encoder_q
        encoder_q = load_model('encoder_q', self.base_encoder, t_params['pretrained_network_params'], trainable=False)
        ckpt = tf.train.Checkpoint(encoder_q=encoder_q)
        ckpt.restore(t_params['pretrained_ckpt']).expect_partial()

        # model, optimizer, and checkpoint must be created under `strategy.scope`
        with strategy.scope():
            # create the encoder and setup
            self.model = load_model('lincls', self.base_encoder, t_params['network_params'], trainable=True)
            self.setup_model(encoder_q)

            # create optimizer
            assert t_params['learning_rate']['schedule'] in ['step', 'cos']
            if t_params['learning_rate']['schedule'] == 'step':
                self.lr_schedule_fn = StepDecay(t_params['learning_rate']['initial_lr'],
                                                t_params['learning_rate']['lr_decay'],
                                                t_params['n_images'],
                                                t_params['epochs'],
                                                t_params['learning_rate']['lr_decay_boundaries'],
                                                t_params['global_batch_size'])
            else:
                self.lr_schedule_fn = CosineDecay(t_params['learning_rate']['initial_lr'], self.max_steps)
            self.optimizer = tf.keras.optimizers.SGD(self.lr_schedule_fn, momentum=0.9, nesterov=False)

            # setup saving locations (object based savings)
            self.ckpt_dir = os.path.join(self.model_base_dir, self.name)
            self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=2)

            # try to restore
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print('Restored from {}'.format(self.manager.latest_checkpoint))

                # check if already trained in this resolution
                restored_step = self.optimizer.iterations.numpy()
                if restored_step >= self.max_steps:
                    print('Already reached max steps {}/{}'.format(restored_step, self.max_steps))
                    self.reached_max_steps = True
                    return
            else:
                print('Not restoring from saved checkpoint')

    def setup_model(self, encoder_q):
        # change weight initialization scheme for last layer
        for w in self.model.weights:
            if 'fc/2/kernel' in w.name:
                w.assign(tf.random.normal(w.shape, mean=0.0, stddev=0.01, dtype=tf.float32))
            if 'fc/2/bias' in w.name:
                w.assign(tf.zeros_like(w, dtype=tf.float32))

        # freeze all layers except last
        for layer in self.model.layers[:-1]:
            layer.trainable = False

        # copy weights from pretrained encoder_q
        for w in self.model.weights:
            if 'fc' in w.name:
                print(f'skipping {w.name}')
                continue

            # find corresponding weight
            qw = [weight for weight in encoder_q.weights if w.name == weight.name][0]

            assert qw.shape == w.shape
            assert qw.name == w.name
            w.assign(qw)
        return

    def train_step(self, inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            # forwards (even though batch norm layers are frozen, set training=True just for illustrated purpose)
            logits = self.model(images, training=True)  # [N, 1000]

            # [N, ]
            c_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            # weight regularization loss (should be 0)
            l2_w_reg = tf.reduce_sum(self.model.losses)

            # scale to global batch scale
            c_loss = tf.reduce_sum(c_loss) * self.dist_loss_scaler
            l2_w_reg = l2_w_reg * self.dist_loss_scaler
            loss = c_loss + l2_w_reg
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, c_loss, l2_w_reg

    def accuracy_step(self, inputs):
        images, labels = inputs
        global_batch_size = tf.shape(labels)[0] * self.n_replica
        global_batch_size = tf.cast(global_batch_size, dtype=tf.float32)

        # forward
        logits = self.model(images, training=False)     # [N, 1000]
        accuracy = tf.cast(tf.equal(tf.math.argmax(logits, axis=1), labels), tf.float32)  # [N, ]

        # scale to global batch scale
        accuracy = tf.reduce_sum(accuracy) * (1.0 / global_batch_size)
        return accuracy

    def train(self, dist_dataset_train, dist_dataset_val, strategy):
        def dist_train_step(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_out = strategy.experimental_run_v2(fn=self.train_step, args=(inputs,))
            else:
                per_replica_out = strategy.run(fn=self.train_step, args=(inputs,))
            mean_loss0 = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_out[0], axis=None)
            mean_loss1 = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_out[1], axis=None)
            mean_loss2 = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_out[2], axis=None)
            return mean_loss0, mean_loss1, mean_loss2

        def dist_accuracy_step(inputs):
            if self.cur_tf_ver == '2.0.0':
                per_replica_out = strategy.experimental_run_v2(fn=self.accuracy_step, args=(inputs,))
            else:
                per_replica_out = strategy.run(fn=self.accuracy_step, args=(inputs,))

            mean_acc = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_out, axis=None)
            return mean_acc

        if self.use_tf_function:
            dist_train_step = tf.function(dist_train_step)
            dist_accuracy_step = tf.function(dist_accuracy_step)

        if self.reached_max_steps:
            return

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

        # start actual training
        print(f'max_steps: {self.max_steps}')
        prev_epoch = 0
        t_start = time.time()
        for images, labels in dist_dataset_train:
            # train step
            mean_loss, mean_ce_loss, mean_l2_reg = dist_train_step((images, labels))

            # get current step
            step = self.optimizer.iterations.numpy()
            c_lr = self.optimizer.learning_rate(self.optimizer.iterations)
            c_ep = self.epochs_per_step * step

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('epochs', c_ep, step=step)
                tf.summary.scalar('learning_rate', c_lr, step=step)
                tf.summary.scalar('ce_loss', mean_ce_loss, step=step)
                tf.summary.scalar('w_l2_reg', mean_l2_reg, step=step)
                tf.summary.scalar('total_loss', mean_loss, step=step)

            # print every self.print_step
            if step % self.print_step == 0:
                elapsed = time.time() - t_start
                logs_h = f'[step/epoch/lr: {step}/{c_ep:.3f}/{c_lr.numpy():.3f} in {elapsed:.2f}s]: '
                logs_b = f'loss {mean_loss.numpy():.3f}, c_loss {mean_ce_loss.numpy():.3f}, l2_reg {mean_l2_reg.numpy():.3f}'
                print(f'{logs_h}{logs_b}')

                # reset timer
                t_start = time.time()

            # save every self.save_step
            if step % self.save_step == 0:
                self.manager.save(checkpoint_number=step)

            # compute accuracy every epoch
            floor_epoch = int(np.floor(c_ep))
            if prev_epoch != floor_epoch:
                v_acc = self.compute_validation_accuracy(dist_dataset_val, dist_accuracy_step)
                prev_epoch = floor_epoch

                with train_summary_writer.as_default():
                    tf.summary.scalar('validation_accuracy', v_acc, step=step)

            # check exit status
            if step >= self.max_steps:
                break

        # save last checkpoint
        step = self.optimizer.iterations.numpy()
        self.manager.save(checkpoint_number=step)
        return

    def compute_validation_accuracy(self, dist_dataset_val, dist_accuracy_step):
        accuracy_sum = tf.constant(0.0, dtype=tf.float32)
        n_items = tf.Variable(0, dtype=tf.int64)
        for images, labels in dist_dataset_val:
            # step
            accuracy = dist_accuracy_step((images, labels))
            accuracy_sum += accuracy
            n_items.assign_add(1)
        return accuracy_sum / tf.cast(n_items, dtype=tf.float32)


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
    parser.add_argument('--pretrained_ckpt',
                        default='/mnt/vision-nas/moono/trained_models/moco-tf-2.x/trial19_moco_v1/ckpt-1000500',
                        type=str)
    parser.add_argument('--pretrained_moco_version', default=1, type=int)
    parser.add_argument('--batch_size_per_replica', default=8, type=int)
    parser.add_argument('--initial_lr', default=30.0, type=float)
    args = vars(parser.parse_args())

    # check args
    assert args['pretrained_moco_version'] in [1, 2]

    # GPU environment settings
    if args['allow_memory_growth']:
        allow_memory_growth()
    if args['debug_split_gpu']:
        split_gpu_for_testing(mem_in_gb=4.7)

    # default values
    dataset_n_images = {'train': 1281167, 'validation': 50000}
    epochs = 100

    # prepare distribute training
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = args['batch_size_per_replica'] * strategy.num_replicas_in_sync

    # training parameters
    training_parameters = {
        # global params
        'cur_tf_ver': cur_tf_ver,
        'name': f'{args["name"]}_lincls',
        'use_tf_function': args['use_tf_function'],
        'model_base_dir': args['model_base_dir'],

        # pretrained model params
        'pretrained_ckpt': args['pretrained_ckpt'],
        'pretrained_network_params': {
            'input_shape': [224, 224, 3],
            'dim': 128,
            'mlp': False if args['pretrained_moco_version'] == 1 else True,
            'w_decay': 0.0001,
        },

        # network params
        'base_encoder': 'resnet50',
        'network_params': {
            'input_shape': [224, 224, 3],
            'dim': 1000,
            'mlp': False,
            'w_decay': 0.0,
        },
        'learning_rate': {
            'schedule': 'step',
            'initial_lr': args['initial_lr'],
            'lr_decay': 0.1,
            'lr_decay_boundaries': [60, 80],
        },

        # training params
        'n_images': dataset_n_images['train'] * epochs,
        'epochs': epochs,
        'batch_size_per_replica': args['batch_size_per_replica'],
        'global_batch_size': global_batch_size,
    }

    # print current details
    pp(training_parameters)

    # load dataset
    res = training_parameters['network_params']['input_shape'][0]
    t_dataset = get_dataset_lincls(args['tfds_data_dir'], is_training=True, res=res, batch_size=global_batch_size)
    v_dataset = get_dataset_lincls(args['tfds_data_dir'], is_training=False, res=res, batch_size=global_batch_size,
                                   epochs=1)

    # create MoCo instance
    lincls = LinearClassification(training_parameters, strategy)

    with strategy.scope():
        # distribute dataset
        t_dist_dataset = strategy.experimental_distribute_dataset(t_dataset)
        v_dist_dataset = strategy.experimental_distribute_dataset(v_dataset)

        # start training
        print('Training...')
        lincls.train(t_dist_dataset, v_dist_dataset, strategy)
    return


if __name__ == '__main__':
    main()
