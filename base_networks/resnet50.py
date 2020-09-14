import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
# from tensorflow.python.distribute import distribution_strategy_context as ds
# from tensorflow.python.distribute import reduce_util
# from tensorflow.python.keras.layers import normalization


DATA_FORMAT = 'channels_first'
BN_AXIS = 1

# # moco.tensorflow
# BN_MOMENTUM = 0.9
# BN_EPS = 1.001e-5
# CONV_KERNEL_INIT = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')
# FC_KERNEL_INIT = tf.keras.initializers.RandomNormal(stddev=0.01)

# moco official
BN_MOMENTUM = 0.9
BN_EPS = 1e-5
CONV_KERNEL_INIT = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')
FC_KERNEL_INIT = tf.keras.initializers.VarianceScaling(scale=1/3, mode='fan_in', distribution='uniform')


# # code from: https://github.com/tensorflow/tensorflow/issues/41980
# # since tf.keras.layers.experimental.SyncBatchNormalization() causing Nan value error on l2 weight regularization?
# # it seems ok when using small batch size, but with larger batch size Nan value occurs
# class SyncBatchNormalization(normalization.BatchNormalizationBase):
#     """The SyncBatchNormalization in TF 2.2 seems causing NaN issue.
#     We implement this one to avoid the issue.
#     See https://github.com/google-research/simclr/blob/bfe07eed7f101ab51f3360100a28690e1bfbf6ec/resnet.py#L37-L85
#     """
#
#     def __init__(self,
#                  axis=-1,
#                  momentum=0.99,
#                  epsilon=1e-3,
#                  center=True,
#                  scale=True,
#                  beta_initializer='zeros',
#                  gamma_initializer='ones',
#                  moving_mean_initializer='zeros',
#                  moving_variance_initializer='ones',
#                  beta_regularizer=None,
#                  gamma_regularizer=None,
#                  beta_constraint=None,
#                  gamma_constraint=None,
#                  renorm=False,
#                  renorm_clipping=None,
#                  renorm_momentum=0.99,
#                  trainable=True,
#                  adjustment=None,
#                  name=None,
#                  **kwargs):
#         # Currently we only support aggregating over the global batch size.
#         super(SyncBatchNormalization, self).__init__(
#             axis=axis,
#             momentum=momentum,
#             epsilon=epsilon,
#             center=center,
#             scale=scale,
#             beta_initializer=beta_initializer,
#             gamma_initializer=gamma_initializer,
#             moving_mean_initializer=moving_mean_initializer,
#             moving_variance_initializer=moving_variance_initializer,
#             beta_regularizer=beta_regularizer,
#             gamma_regularizer=gamma_regularizer,
#             beta_constraint=beta_constraint,
#             gamma_constraint=gamma_constraint,
#             renorm=renorm,
#             renorm_clipping=renorm_clipping,
#             renorm_momentum=renorm_momentum,
#             fused=False,
#             trainable=trainable,
#             virtual_batch_size=None,
#             name=name,
#             **kwargs)
#
#     def _calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
#         shard_mean, shard_variance = super(SyncBatchNormalization, self)._calculate_mean_and_var(
#             inputs, reduction_axes, keep_dims=keep_dims)
#         replica_ctx = ds.get_replica_context()
#         if replica_ctx:
#             group_mean, group_variance = replica_ctx.all_reduce(reduce_util.ReduceOp.MEAN, [shard_mean, shard_variance])
#             mean_distance = tf.math.squared_difference(tf.stop_gradient(group_mean), shard_mean)
#             group_variance += replica_ctx.all_reduce(reduce_util.ReduceOp.MEAN, mean_distance)
#             return group_mean, group_variance
#         else:
#             return shard_mean, shard_variance


class Block1(models.Model):
    def __init__(self, filters, kernel_size, stride, downsample, w_decay, name, **kwargs):
        super(Block1, self).__init__(name=name, **kwargs)
        reg = l2(w_decay)
        self.downsample = downsample
        self.relu = layers.Activation('relu', name='relu')

        if self.downsample:
            self.conv = layers.Conv2D(4 * filters, 1, strides=stride, name='downsample/0', data_format=DATA_FORMAT,
                                      use_bias=False, kernel_initializer=CONV_KERNEL_INIT, kernel_regularizer=reg)
            # self.bn = layers.BatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='downsample/1')
            self.bn = layers.experimental.SyncBatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='downsample/1')
            # self.bn = SyncBatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='downsample/1')

        # 1x1
        self.conv_1 = layers.Conv2D(filters, 1, name='conv1', data_format=DATA_FORMAT,
                                    use_bias=False, kernel_initializer=CONV_KERNEL_INIT, kernel_regularizer=reg)
        # self.bn_1 = layers.BatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn1')
        self.bn_1 = layers.experimental.SyncBatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn1')
        # self.bn_1 = SyncBatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn1')

        # 3x3
        self.pad_2 = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pad2', data_format=DATA_FORMAT)
        self.conv_2 = layers.Conv2D(filters, kernel_size, strides=stride, name='conv2', data_format=DATA_FORMAT,
                                    use_bias=False, kernel_initializer=CONV_KERNEL_INIT, kernel_regularizer=reg)
        # self.bn_2 = layers.BatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn2')
        self.bn_2 = layers.experimental.SyncBatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn2')
        # self.bn_2 = SyncBatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn2')

        # 1x1
        self.conv_3 = layers.Conv2D(4 * filters, 1, name='conv3', data_format=DATA_FORMAT,
                                    use_bias=False, kernel_initializer=CONV_KERNEL_INIT, kernel_regularizer=reg)
        # self.bn_3 = layers.BatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn3')
        self.bn_3 = layers.experimental.SyncBatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn3')
        # self.bn_3 = SyncBatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn3')

    def call(self, inputs, training=None, mask=None):
        identity = tf.identity(inputs)

        out = self.conv_1(inputs)
        out = self.bn_1(out, training=training)
        out = self.relu(out)

        out = self.pad_2(out)
        out = self.conv_2(out)
        out = self.bn_2(out, training=training)
        out = self.relu(out)

        out = self.conv_3(out)
        out = self.bn_3(out, training=training)

        if self.downsample:
            identity = self.conv(inputs)
            identity = self.bn(identity, training=training)

        out += identity
        out = self.relu(out)

        return out


class Stack1(models.Model):
    def __init__(self, filters, n_blocks, stride, w_decay, name, **kwargs):
        super(Stack1, self).__init__(name=name, **kwargs)

        self.block_1 = Block1(filters, 3, stride, downsample=True, w_decay=w_decay, name=f'0')
        self.blocks = list()
        for i in range(1, n_blocks):
            self.blocks.append(Block1(filters, 3, 1, downsample=False, w_decay=w_decay, name=f'{i}'))

    def call(self, inputs, training=None, mask=None):
        x = inputs

        x = self.block_1(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return x


# wrap again for distribute training momentum update
class Resnet50(models.Model):
    def __init__(self, resnet_params, name, **kwargs):
        super(Resnet50, self).__init__(name=name, **kwargs)
        w_decay = resnet_params['w_decay']
        reg = l2(w_decay)

        self.res = resnet_params['input_shape'][0]
        self.classes = resnet_params['dim']
        self.with_projection_head = resnet_params['mlp']

        # layers
        self.preprocess = layers.Lambda(lambda x: self._preprocess_inputs(x))

        self.pad1 = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='pad1', data_format=DATA_FORMAT)
        self.conv1 = layers.Conv2D(64, 7, strides=2, name='conv1', data_format=DATA_FORMAT,
                                   use_bias=False, kernel_initializer=CONV_KERNEL_INIT, kernel_regularizer=reg)
        # self.bn1 = layers.BatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn1')
        self.bn1 = layers.experimental.SyncBatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn1')
        # self.bn1 = SyncBatchNormalization(axis=BN_AXIS, momentum=BN_MOMENTUM, epsilon=BN_EPS, name='bn1')
        self.relu1 = layers.Activation('relu', name='relu1')
        self.pool1_pad = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad', data_format=DATA_FORMAT)
        self.pool1_pool = layers.MaxPooling2D(3, strides=2, name='pool1_pool', data_format=DATA_FORMAT)

        self.stack_1 = Stack1(64, 3, stride=1, w_decay=w_decay, name='layer1')
        self.stack_2 = Stack1(128, 4, stride=2, w_decay=w_decay, name='layer2')
        self.stack_3 = Stack1(256, 6, stride=2, w_decay=w_decay, name='layer3')
        self.stack_4 = Stack1(512, 3, stride=2, w_decay=w_decay, name='layer4')

        self.avg_pool = layers.GlobalAveragePooling2D(name='avg_pool', data_format=DATA_FORMAT)

        if self.with_projection_head:
            self.mlp = layers.Dense(2048, activation='relu', name='fc/0',
                                    kernel_initializer=FC_KERNEL_INIT, kernel_regularizer=reg, bias_regularizer=reg)

        # [N, classes]
        self.last_dense = layers.Dense(self.classes, activation=None, name='fc/2',
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
        # image: rgb, (0.0 ~ 1.0)
        # make channel first
        image = tf.transpose(image, perm=[0, 3, 1, 2])

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        image_mean = tf.constant(image_mean, dtype=tf.float32)[None, :, None, None]
        image_std = tf.constant(image_std, dtype=tf.float32)[None, :, None, None]

        # normalize
        image = (image - image_mean) / image_std
        return image

    def call(self, inputs, training=None, mask=None):
        x = inputs  # [N, 224, 224, 3]

        x = self.preprocess(x)  # [N, 3, 224, 224]

        x = self.pad1(x)                        # [N, 3, 230, 230]
        x = self.conv1(x)                       # [N, 64, 112, 112]
        x = self.bn1(x, training=training)      # [N, 64, 112, 112]
        x = self.relu1(x)                       # [N, 64, 112, 112]
        x = self.pool1_pad(x)                   # [N, 64, 114, 114]
        x = self.pool1_pool(x)                  # [N, 64, 56, 56]

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

    for w in resnet50.weights:
        print(f'[{w.name}]: {w.shape}')
    return


def main():
    from misc.tf_utils import allow_memory_growth

    allow_memory_growth()

    test_raw()
    return


if __name__ == '__main__':
    main()
