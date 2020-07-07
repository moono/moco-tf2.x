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


# def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
#     data_format = 'channels_first'
#     bn_axis = 1
#
#     if conv_shortcut is True:
#         shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv', data_format=data_format)(x)
#         shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
#     else:
#         shortcut = x
#
#     x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv', data_format=data_format)(x)
#     x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
#     x = layers.Activation('relu', name=name + '_1_relu')(x)
#
#     x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv', data_format=data_format)(x)
#     x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
#     x = layers.Activation('relu', name=name + '_2_relu')(x)
#
#     x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv', data_format=data_format)(x)
#     x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)
#
#     x = layers.Add(name=name + '_add')([shortcut, x])
#     x = layers.Activation('relu', name=name + '_out')(x)
#     return x
#
#
# def stack1(x, filters, blocks, stride1=2, name=None):
#     x = block1(x, filters, stride=stride1, name=name + '_block1')
#     for i in range(2, blocks + 1):
#         x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
#     return x
#
#
# def stack_fn(x):
#     x = stack1(x, 64, 3, stride1=1, name='conv2')
#     x = stack1(x, 128, 4, name='conv3')
#     x = stack1(x, 256, 6, name='conv4')
#     x = stack1(x, 512, 3, name='conv5')
#     return x
#
#
# def preprocess_inputs(image):
#     # image: rgb, 0.0 ~ 1.0
#     image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)[None, None, None, :]
#     image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)[None, None, None, :]
#
#     # normalize
#     image = (image - image_mean) / image_std
#
#     # make channel first
#     image = tf.transpose(image, perm=[0, 3, 1, 2])
#     return image
#
#
# def get_model(res, classes, with_projection_head=False):
#     model_name = 'resnet50'
#     data_format = 'channels_first'
#     bn_axis = 1
#     img_input = layers.Input(shape=(res, res, 3), dtype='float32', name='input_1')
#
#     preprocessed = layers.Lambda(lambda _x: preprocess_inputs(_x))(img_input)
#
#     x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad', data_format=data_format)(preprocessed)
#     x = layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv', data_format=data_format)(x)
#
#     x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
#     x = layers.Activation('relu', name='conv1_relu')(x)
#
#     x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad', data_format=data_format)(x)
#     x = layers.MaxPooling2D(3, strides=2, name='pool1_pool', data_format=data_format)(x)
#
#     x = stack_fn(x)
#
#     # [N, 2048]
#     x = layers.GlobalAveragePooling2D(name='avg_pool', data_format=data_format)(x)
#
#     if with_projection_head:
#         # [N, 2048]
#         x = layers.Dense(2048, activation='relu')(x)
#
#     # [N, classes]
#     x = layers.Dense(classes, activation='softmax', name='probs')(x)
#
#     inputs = img_input
#
#     # Create model.
#     model = models.Model(inputs, x, name=model_name)
#     return model


# # wrap again for distribute training momentum update
# class Resnet50(models.Model):
#     def __init__(self, resnet_params, **kwargs):
#         super(Resnet50, self).__init__(**kwargs)
#         self.resnet50 = get_model(
#             res=resnet_params['input_shape'][0],
#             classes=resnet_params['dim'],
#             with_projection_head=resnet_params['mlp'])
#         return
#
#     @tf.function
#     def momentum_update(self, src_net, m):
#         for qw, kw in zip(src_net.weights, self.weights):
#             assert qw.shape == kw.shape
#             updated_w = kw * m + qw * (1.0 - m)
#             kw.assign(updated_w)
#         return
#
#     def call(self, inputs, training=None, mask=None):
#         return self.resnet50(inputs, training=training, mask=mask)


class Block1(models.Model):
    def __init__(self, filters, kernel_size, stride, conv_shortcut, w_decay, name, **kwargs):
        super(Block1, self).__init__(name=name, **kwargs)
        data_format = 'channels_first'
        bn_axis = 1
        reg = l2(w_decay)
        self.conv_shortcut = conv_shortcut

        if self.conv_shortcut:
            self.conv_0 = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv',
                                        data_format=data_format, kernel_regularizer=reg, bias_regularizer=reg)
            self.bn_0 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')

        self.conv_1 = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv',
                                    data_format=data_format, kernel_regularizer=reg, bias_regularizer=reg)
        self.bn_1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')
        self.relu_1 = layers.Activation('relu', name=name + '_1_relu')

        self.conv_2 = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv',
                                    data_format=data_format, kernel_regularizer=reg, bias_regularizer=reg)
        self.bn_2 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')
        self.relu_2 = layers.Activation('relu', name=name + '_2_relu')

        self.conv_3 = layers.Conv2D(4 * filters, 1, name=name + '_3_conv',
                                    data_format=data_format, kernel_regularizer=reg, bias_regularizer=reg)
        self.bn_3 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')

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
        data_format = 'channels_first'
        bn_axis = 1
        reg = l2(w_decay)

        self.res = resnet_params['input_shape'][0]
        self.classes = resnet_params['dim']
        self.with_projection_head = resnet_params['mlp']

        # layers
        self.preprocess = layers.Lambda(lambda x: self._preprocess_inputs(x))

        self.conv1_pad = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad', data_format=data_format)
        self.conv1_conv = layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv',
                                        data_format=data_format, kernel_regularizer=reg, bias_regularizer=reg)
        self.conv1_bn = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')
        self.conv1_relu = layers.Activation('relu', name='conv1_relu')
        self.pool1_pad = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad', data_format=data_format)
        self.pool1_pool = layers.MaxPooling2D(3, strides=2, name='pool1_pool', data_format=data_format)

        self.stack_1 = Stack1(64, 3, stride=1, w_decay=w_decay, name='conv2')
        self.stack_2 = Stack1(128, 4, stride=2, w_decay=w_decay, name='conv3')
        self.stack_3 = Stack1(256, 6, stride=2, w_decay=w_decay, name='conv4')
        self.stack_4 = Stack1(512, 3, stride=2, w_decay=w_decay, name='conv5')

        self.avg_pool = layers.GlobalAveragePooling2D(name='avg_pool', data_format=data_format)

        if self.with_projection_head:
            self.mlp = layers.Dense(2048, activation='relu',
                                    kernel_regularizer=reg, bias_regularizer=reg, name='mlp')

        # [N, classes]
        self.last_dense = layers.Dense(self.classes, activation='softmax',
                                       kernel_regularizer=reg, bias_regularizer=reg, name='probs')
        return

    @tf.function
    def momentum_update(self, src_net, m):
        for qw, kw in zip(src_net.weights, self.weights):
            # print(f'{qw.name}: {kw.name}')
            # assert qw.shape == kw.shape
            # assert qw.name == kw.name
            updated_w = kw * m + qw * (1.0 - m)
            kw.assign(updated_w)

            # tf.debugging.assert_near(updated_w, kw)
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


def test_compare():
    from tensorflow.keras.applications.resnet50 import ResNet50 as OfficialResnet50

    res = 224
    classes = 1000
    input_shape = [res, res, 3]
    resnet_params = {
        'input_shape': input_shape,
        'dim': classes,
        'mlp': False,
        'w_decay': 0.0001,
    }

    # Total params: 25,636,712
    # Trainable params: 25,583,592
    # Non-trainable params: 53,120
    official = OfficialResnet50(include_top=True, weights=None, input_shape=input_shape, pooling=None, classes=classes)
    official.summary()

    # Total params: 25,636,712
    # Trainable params: 25,583,592
    # Non-trainable params: 53,120
    resnet50 = Resnet50(resnet_params)
    resnet50.build((None, *resnet_params['input_shape']))
    resnet50.summary()
    return


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

    test_compare()
    test_raw()
    return


if __name__ == '__main__':
    main()
