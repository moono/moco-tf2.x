"""
Just copy of keras_application/resnet_common.py

Model: Resnet50
channel_first,
preact = False,
use_bias = True
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


DATA_FORMAT = 'channels_first'
BN_AXIS = 1

# moco.tensorflow
BN_MOMENTUM = 0.9
BN_EPS = 1.001e-5
CONV_KERNEL_INIT = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')
FC_KERNEL_INIT = tf.keras.initializers.RandomNormal(stddev=0.01)

# # moco official
# BN_MOMENTUM = 0.9
# BN_EPS = 1e-5
# CONV_KERNEL_INIT = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')
# FC_KERNEL_INIT = tf.keras.initializers.VarianceScaling(scale=1/3, mode='fan_in', distribution='uniform')

# # test
# BN_MOMENTUM = 0.9
# BN_EPS = 1e-5
# CONV_KERNEL_INIT = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')
# FC_KERNEL_INIT = tf.keras.initializers.VarianceScaling(scale=1/3, mode='fan_in', distribution='uniform')


class Block1(models.Model):
    def __init__(self, filters, kernel_size, stride, conv_shortcut, w_decay, name, **kwargs):
        super(Block1, self).__init__(name=name, **kwargs)
        reg = l2(w_decay)
        self.conv_shortcut = conv_shortcut

        if self.conv_shortcut:
            self.conv_0 = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv', data_format=DATA_FORMAT,
                                        use_bias=False, kernel_initializer=CONV_KERNEL_INIT, kernel_regularizer=reg)
            self.bn_0 = layers.BatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name=name + '_0_bn')

        self.conv_1 = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv', data_format=DATA_FORMAT,
                                    use_bias=False, kernel_initializer=CONV_KERNEL_INIT, kernel_regularizer=reg)
        self.bn_1 = layers.BatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name=name + '_1_bn')
        self.relu_1 = layers.Activation('relu', name=name + '_1_relu')

        self.conv_2 = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv', data_format=DATA_FORMAT,
                                    use_bias=False, kernel_initializer=CONV_KERNEL_INIT, kernel_regularizer=reg)
        self.bn_2 = layers.BatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name=name + '_2_bn')
        self.relu_2 = layers.Activation('relu', name=name + '_2_relu')

        self.conv_3 = layers.Conv2D(4 * filters, 1, name=name + '_3_conv', data_format=DATA_FORMAT,
                                    use_bias=False, kernel_initializer=CONV_KERNEL_INIT, kernel_regularizer=reg)
        self.bn_3 = layers.BatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name=name + '_3_bn')

        self.relu_out = layers.Activation('relu', name=name + '_out')

    def call(self, inputs, training=None, mask=None):
        x = inputs

        if self.conv_shortcut:
            shortcut = self.conv_0(x)
            shortcut = self.bn_0(shortcut, training=training)
        else:
            shortcut = tf.identity(x)

        x = self.conv_1(x)
        x = self.bn_1(x, training=training)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = self.relu_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x, training=training)

        x = shortcut + x
        x = self.relu_out(x)

        return x


class Stack1(models.Model):
    def __init__(self, filters, n_blocks, stride, w_decay, name, **kwargs):
        super(Stack1, self).__init__(name=name, **kwargs)

        self.block_1 = Block1(filters, 3, stride, conv_shortcut=True, w_decay=w_decay, name=f'{name}_block1')
        self.blocks = list()
        for i in range(2, n_blocks + 1):
            self.blocks.append(Block1(filters, 3, 1, conv_shortcut=False,  w_decay=w_decay, name=f'{name}_block{i}'))

    def call(self, inputs, training=None, mask=None):
        x = inputs

        x = self.block_1(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return x


# wrap again for distribute training momentum update
class Resnet50(models.Model):
    def __init__(self, resnet_params, **kwargs):
        super(Resnet50, self).__init__(**kwargs)
        w_decay = resnet_params['w_decay']
        reg = l2(w_decay)

        self.res = resnet_params['input_shape'][0]
        self.classes = resnet_params['dim']
        self.with_projection_head = resnet_params['mlp']

        # layers
        self.preprocess = layers.Lambda(lambda x: self._preprocess_inputs(x))

        self.conv1_pad = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad', data_format=DATA_FORMAT)
        self.conv1_conv = layers.Conv2D(64, 7, strides=2, name='conv1_conv', data_format=DATA_FORMAT,
                                        use_bias=False, kernel_initializer=CONV_KERNEL_INIT, kernel_regularizer=reg)
        self.conv1_bn = layers.BatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='conv1_bn')
        self.conv1_relu = layers.Activation('relu', name='conv1_relu')
        self.pool1_pad = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad', data_format=DATA_FORMAT)
        self.pool1_pool = layers.MaxPooling2D(3, strides=2, name='pool1_pool', data_format=DATA_FORMAT)

        self.stack_1 = Stack1(64, 3, stride=1, w_decay=w_decay, name='conv2')
        self.stack_2 = Stack1(128, 4, stride=2, w_decay=w_decay, name='conv3')
        self.stack_3 = Stack1(256, 6, stride=2, w_decay=w_decay, name='conv4')
        self.stack_4 = Stack1(512, 3, stride=2, w_decay=w_decay, name='conv5')

        self.avg_pool = layers.GlobalAveragePooling2D(name='avg_pool', data_format=DATA_FORMAT)

        if self.with_projection_head:
            self.mlp = layers.Dense(2048, activation='relu', name='mlp',
                                    kernel_initializer=FC_KERNEL_INIT, kernel_regularizer=reg, bias_regularizer=reg)

        # [N, classes]
        self.last_dense = layers.Dense(self.classes, activation=None, name='probs',
                                       kernel_initializer=FC_KERNEL_INIT, kernel_regularizer=reg, bias_regularizer=reg)
        return

    @tf.function
    def momentum_update(self, src_net, m):
        for qw, kw in zip(src_net.weights, self.weights):
            # don't update ema variables
            if 'moving' in qw.name:
                continue

            updated_w = kw * m + qw * (1.0 - m)
            kw.assign(updated_w)
        return

    def _preprocess_inputs(self, image):
        # image: rgb, 0.0 ~ 1.0
        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)[None, None, None, :]
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)[None, None, None, :]

        # normalize
        image = (image - image_mean) / image_std

        # make channel first
        image = tf.transpose(image, perm=[0, 3, 1, 2])
        return image

    def call(self, inputs, training=None, mask=None):
        x = inputs  # [N, 224, 224, 3]

        x = self.preprocess(x)  # [N, 3, 224, 224]

        x = self.conv1_pad(x)   # [N, 3, 230, 230]
        x = self.conv1_conv(x)  # [N, 64, 112, 112]
        x = self.conv1_bn(x, training=training)     # [N, 64, 112, 112]
        x = self.conv1_relu(x)  # [N, 64, 112, 112]
        x = self.pool1_pad(x)   # [N, 64, 114, 114]
        x = self.pool1_pool(x)  # [N, 64, 56, 56]

        x = self.stack_1(x, training=training)  # [N, 256, 56, 56]
        x = self.stack_2(x, training=training)  # [N, 512, 28, 28]
        x = self.stack_3(x, training=training)  # [N, 1024, 14, 14]
        x = self.stack_4(x, training=training)  # [N, 2048, 7, 7]

        x = self.avg_pool(x)    # [N, 2048]
        if self.with_projection_head:
            x = self.mlp(x)     # [N, 2048]
        x = self.last_dense(x)  # [N, 128]
        return x


def test_raw():
    res = 224
    classes = 128
    resnet_params = {
        'input_shape': [res, res, 3],
        'dim': classes,
        'mlp': True,
        'w_decay': 0.0001,
        # 'w_decay': 0.0,
    }
    resnet50 = Resnet50(resnet_params, name='encoder')
    resnet50.build((None, *resnet_params['input_shape']))
    resnet50.summary()
    return


def main():
    from misc.tf_utils import allow_memory_growth

    allow_memory_growth()

    test_raw()
    return


if __name__ == '__main__':
    main()
