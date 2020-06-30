import tensorflow as tf


class Linear(tf.keras.models.Model):
    def __init__(self, linear_params, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(linear_params['dim'], use_bias=True)
        return

    @tf.function
    def momentum_update(self, src_net, m):
        for qw, kw in zip(src_net.weights, self.weights):
            assert qw.shape == kw.shape
            kw.assign(kw * m + qw * (1.0 - m))
        return

    def call(self, inputs, training=None, mask=None):
        return self.dense(inputs)
