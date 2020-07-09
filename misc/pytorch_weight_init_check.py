# Try to mimic pytorch's weight initialization
import tensorflow as tf


def _calculate_fan_in_and_fan_out(w):
    dimensions = w.ndim
    assert dimensions in [2, 4]

    receptive_field_size = 1
    num_input_fmaps = tf.shape(w)[0]
    num_output_fmaps = tf.shape(w)[1]
    if dimensions == 4:
        receptive_field_size = tf.reduce_prod(tf.shape(w)[:1])
        num_input_fmaps = tf.shape(w)[2]
        num_output_fmaps = tf.shape(w)[3]

    fan_in = tf.cast(num_input_fmaps * receptive_field_size, dtype=tf.float32)
    fan_out = tf.cast(num_output_fmaps * receptive_field_size, dtype=tf.float32)
    return fan_in, fan_out


def _calculate_correct_fan(x, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f'Mode {mode} not supported, please use one of {valid_modes}')
    fan_in, fan_out = _calculate_fan_in_and_fan_out(x)
    return fan_in if mode == 'fan_in' else fan_out


def calculate_gain(nonlinearity, param=None):
    if nonlinearity == 'sigmoid':
        return 1.0
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return tf.math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f'negative_slope {param} not a valid number')
        return tf.math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError(f'Unsupported nonlinearity {nonlinearity}')


def kaiming_normal(w, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(w, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / tf.math.sqrt(fan)
    return tf.random.normal(shape=tf.shape(w), mean=0.0, stddev=std)


def kaiming_uniform(w, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(w, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / tf.math.sqrt(fan)
    bound = tf.math.sqrt(3.0) * std
    return tf.random.uniform(shape=tf.shape(w), minval=-bound, maxval=bound)


def test_conv():
    data_format = 'channels_first'
    batch_size = 1
    kernel = 3
    in_channel = 256
    out_channel = 512
    res = 4

    simulate_w = tf.random.normal(shape=[kernel, kernel, in_channel, out_channel])

    # pytorch resnet50's conv weight init
    # gain = sqrt(2.0)
    # std = sqrt(2.0) / sqrt(fan_out)
    conv_init = kaiming_normal(simulate_w, a=0, mode='fan_out', nonlinearity='relu')
    print(conv_init)
    return


def test_dense():
    in_channel = 256
    units = 512

    simulate_w = tf.random.normal(shape=[in_channel, units])
    simulate_b = tf.random.normal(shape=[units])

    # pytorch resnet50's dense weight init
    # bound == 1. / sqrt(fan_in) for both kernel & bias
    dense_w_init = kaiming_uniform(simulate_w, a=tf.math.sqrt(5.0))
    fan_in, _ = _calculate_fan_in_and_fan_out(simulate_w)
    bound = 1.0 / tf.math.sqrt(fan_in)
    dense_b_init = tf.random.uniform(shape=tf.shape(simulate_b), minval=-bound, maxval=bound)
    return


def main():
    test_conv()
    test_dense()
    return


if __name__ == '__main__':
    main()

