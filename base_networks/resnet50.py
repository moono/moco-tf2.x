"""
Just copy of keras_application/resnet_common.py

Model: Resnet50
channel_first,
preact = False,
use_bias = True
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    data_format = 'channels_first'
    bn_axis = 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv', data_format=data_format)(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv', data_format=data_format)(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv', data_format=data_format)(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv', data_format=data_format)(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 6, name='conv4')
    x = stack1(x, 512, 3, name='conv5')
    return x


def preprocess_inputs(image):
    # image: rgb, 0.0 ~ 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = mean[::-1]
    std = std[::-1]
    image_mean = tf.constant(mean, dtype=tf.float32) * 255.
    image_std = tf.constant(std, dtype=tf.float32) * 255.

    # convert to bgr
    image = tf.reverse(image, axis=[-1])

    # normalize
    image = (image - image_mean) / image_std

    # make channel first
    image = tf.transpose(image, perm=[0, 3, 1, 2])
    return image


def get_model(res, classes, with_projection_head=False):
    model_name = 'resnet50'
    data_format = 'channels_first'
    bn_axis = 1
    img_input = layers.Input(shape=(res, res, 3), dtype='float32', name='input_1')

    preprocessed = layers.Lambda(lambda _x: preprocess_inputs(_x))(img_input)

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad', data_format=data_format)(preprocessed)
    x = layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv', data_format=data_format)(x)

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad', data_format=data_format)(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool', data_format=data_format)(x)

    x = stack_fn(x)

    # [N, 2048]
    x = layers.GlobalAveragePooling2D(name='avg_pool', data_format=data_format)(x)

    if with_projection_head:
        # [N, 2048]
        x = layers.Dense(2048, activation='relu')(x)

    # [N, classes]
    x = layers.Dense(classes, activation='softmax', name='probs')(x)

    inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)
    return model


# wrap again for distribute training momentum update
class Resnet50(models.Model):
    def __init__(self, resnet_params, **kwargs):
        super(Resnet50, self).__init__(**kwargs)
        self.resnet50 = get_model(
            res=resnet_params['input_shape'][0],
            classes=resnet_params['dim'],
            with_projection_head=resnet_params['mlp'])
        return

    @tf.function
    def momentum_update(self, src_net, m):
        for qw, kw in zip(src_net.weights, self.weights):
            assert qw.shape == kw.shape
            kw.assign(kw * m + qw * (1.0 - m))
        return

    def call(self, inputs, training=None, mask=None):
        return self.resnet50(inputs)


def test_compare():
    from tensorflow.keras.applications.resnet50 import ResNet50 as OfficialResnet50

    res = 224
    classes = 1000
    input_shape = [res, res, 3]
    resnet_params = {
        'input_shape': input_shape,
        'dim': classes,
        'mlp': False,
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
    _ = resnet50(tf.random.normal(shape=[1] + resnet_params['input_shape']))
    resnet50.summary()
    print(_.shape)
    return


def test_raw():
    res = 224
    classes = 512
    resnet_params = {
        'input_shape': [res, res, 3],
        'dim': classes,
        'mlp': True,
    }
    resnet50 = Resnet50(resnet_params, name='encoder')
    _ = resnet50(tf.random.normal(shape=[1] + resnet_params['input_shape']))
    resnet50.summary()
    print(_.shape)
    return


def main():
    from misc.tf_utils import allow_memory_growth

    allow_memory_growth()

    test_compare()
    test_raw()
    return


if __name__ == '__main__':
    main()
