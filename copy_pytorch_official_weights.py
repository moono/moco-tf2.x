import os
import argparse
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.models as models

from misc.utils import str_to_bool
from misc.tf_utils import allow_memory_growth
from base_networks.resnet50 import Resnet50


def get_weights_state_dict(pretrained):
    # create model
    arch = 'resnet50'
    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch](num_classes=128)

    if os.path.isfile(pretrained):
        print(f'Loading checkpoint {pretrained}')
        checkpoint = torch.load(pretrained, map_location='cpu')

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        state_dict_np = dict()
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict[k[len('module.encoder_q.'):]] = state_dict[k]
                state_dict_np[k[len('module.encoder_q.'):]] = state_dict[k].numpy()

            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        model = model.to('cuda')

        print("=> loaded pre-trained model '{}'".format(pretrained))
        return state_dict_np, model
    else:
        return None, None


def copy_weights(state_dict, tf_model):
    def convert_to_pytorch_name(w_name):
        w_name = w_name.replace(':0', '')
        w_name = w_name.replace('/', '.')
        w_name = w_name.replace('kernel', 'weight')
        w_name = w_name.replace('gamma', 'weight')
        w_name = w_name.replace('beta', 'bias')
        w_name = w_name.replace('moving_mean', 'running_mean')
        w_name = w_name.replace('moving_variance', 'running_var')
        return w_name

    # copy
    for tf_w in tf_model.weights:
        # find corresponding torch name and its weight
        tf_name = tf_w.name
        torch_name = convert_to_pytorch_name(tf_name)
        torch_w = state_dict[torch_name]

        # convert to tensorflow shape type
        if len(torch_w.shape) == 4:
            as_tf_format = tf.convert_to_tensor(np.transpose(torch_w, axes=[2, 3, 1, 0]))
        elif len(torch_w.shape) == 2:
            as_tf_format = tf.convert_to_tensor(np.transpose(torch_w, axes=[1, 0]))
        else:
            as_tf_format = tf.convert_to_tensor(torch_w)
        assert tuple(tf_w.shape) == as_tf_format.shape

        tf_w.assign(as_tf_format)
        tf.debugging.assert_equal(as_tf_format, tf_w)
    return


def copy_official_model(copy_mlp, pytorch_weight_fn, ckpt_dir, ret_model=False):
    # load pytorch model weights
    state_dict, pytorch_model = get_weights_state_dict(pytorch_weight_fn)

    # load tensorflow model
    resnet_params = {
        'input_shape': [224, 224, 3],
        'dim': 128,
        'mlp': copy_mlp,
        'w_decay': 0.0001,
    }
    tf_model = Resnet50(resnet_params, name='encoder_q')
    tf_model.build((None, *resnet_params['input_shape']))

    # copy weights: pytorch -> tensorflow
    copy_weights(state_dict, tf_model)

    # save in tensorflow format
    ckpt = tf.train.Checkpoint(encoder_q=tf_model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
    manager.save(0)

    if ret_model:
        return pytorch_model, tf_model
    else:
        return None, None


class SubModelTorch(nn.Module):
    def __init__(self, pretrained):
        super(SubModelTorch, self).__init__()
        # until avg pooling
        image_modules = list(pretrained.children())[:-1]
        self.modelA = nn.Sequential(*image_modules)

    def forward(self, inputs):
        x = self.modelA(inputs)
        x = torch.flatten(x, 1)
        return x


def run_tf_model(tf_model, test_inputs_tf, training=False):
    x = tf_model.preprocess(test_inputs_tf)
    x = tf_model.pad1(x)
    x = tf_model.conv1(x)
    x = tf_model.bn1(x, training=training)
    x = tf_model.relu1(x)
    x = tf_model.pool1_pad(x)
    x = tf_model.pool1_pool(x)
    x = tf_model.stack_1(x, training=training)
    x = tf_model.stack_2(x, training=training)
    x = tf_model.stack_3(x, training=training)
    x = tf_model.stack_4(x, training=training)
    x = tf_model.avg_pool(x)
    x = tf.reshape(x, shape=[1, -1])
    return x


def test(pytorch_model, tf_model):
    # can run until average pooling
    sub_pytorch_model = SubModelTorch(pytorch_model)

    # set etc
    training = False
    sub_pytorch_model.eval()
    image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]
    image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None]

    n_test_inputs = 1000
    test_inputs_list = np.random.uniform(0.0, 1.0, size=(n_test_inputs, 3, 224, 224))

    print(f'Running')
    outputs_torch, outputs_tf, distances = list(), list(), list()
    for ii in range(n_test_inputs):
        test_inputs_np = np.reshape(test_inputs_list[ii], newshape=(1, 3, 224, 224))
        test_inputs_np = test_inputs_np.astype(np.float32)

        test_inputs_torch = torch.from_numpy(test_inputs_np)
        test_inputs_torch = (test_inputs_torch - image_mean) / image_std
        test_inputs_torch = test_inputs_torch.to('cuda')
        pytorch_out = sub_pytorch_model(test_inputs_torch)
        pytorch_out = pytorch_out.cpu().detach().numpy()
        outputs_torch.append(pytorch_out)

        test_inputs_tf = np.transpose(test_inputs_np, axes=(0, 2, 3, 1))
        test_inputs_tf = tf.constant(test_inputs_tf, dtype=tf.float32)
        tf_out = run_tf_model(tf_model, test_inputs_tf, training=training)
        tf_out = tf_out.numpy()
        outputs_tf.append(tf_out)

        distances.append(np.linalg.norm(pytorch_out - tf_out))
    distances = np.array(distances, dtype=np.float32)
    print(f'distance mean: {distances.mean()}')
    print(f'distance std: {distances.std()}')
    return


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--pytorch_weight_fn', default='./official_pretrained/moco_v2_800ep_pretrain.pth.tar', type=str)
    parser.add_argument('--output_ckpt_dir', default='./official_pretrained/tensorflow_converted', type=str)
    parser.add_argument('--copy_mlp', type=str_to_bool, nargs='?', const=True, default=False)
    args = vars(parser.parse_args())

    if args['allow_memory_growth']:
        allow_memory_growth()

    # step 1: copy official pytorch weights to tensorflow model
    pytorch_model, tf_model = copy_official_model(args['copy_mlp'], args['pytorch_weight_fn'], args['output_ckpt_dir'],
                                                  ret_model=True)

    # step 2: check outputs
    test(pytorch_model, tf_model)
    return


if __name__ == '__main__':
    main()
