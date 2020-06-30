import importlib
import tensorflow as tf


def get_proper_module(module_name, object_name):
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    return obj


def load_model(name, network_name, network_params, trainable):
    class_name_table = {
        'resnet50': 'Resnet50',
        'linear': 'Linear',
    }
    class_name = class_name_table[network_name]
    model = get_proper_module(f'base_networks.{network_name}', class_name)
    m = model(network_params, name=name)

    test_input_images = tf.random.normal(shape=[1] + network_params['input_shape'])
    _ = m(test_input_images)

    if not trainable:
        for layer in m.layers:
            layer.trainable = False
    return m


def main():
    from misc.tf_utils import allow_memory_growth

    allow_memory_growth()

    linear_params = {
        'input_shape': [4],
        'dim': 4,
    }
    linear_q = load_model(name='linear_q', network_name='linear', network_params=linear_params, trainable=True)
    linear_k = load_model(name='linear_k', network_name='linear', network_params=linear_params, trainable=False)
    print(linear_q.summary())
    print(linear_k.summary())

    resnet_params = {
        'input_shape': [224, 224, 3],
        'last_dim': 512,
        'with_projection_head': True,
    }

    resnet50_q = load_model(name='encoder_q', network_name='resnet50', network_params=resnet_params, trainable=True)
    resnet50_k = load_model(name='encoder_k', network_name='resnet50', network_params=resnet_params, trainable=False)
    print(resnet50_q.summary())
    print(resnet50_k.summary())
    return


if __name__ == '__main__':
    main()
