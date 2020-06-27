import importlib
import tensorflow as tf


def get_proper_module(module_name, object_name):
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    return obj


def load_model(name, network_name, res, dim, mlp, trainable):
    class_name_table = {'resnet50': 'Resnet50'}
    class_name = class_name_table[network_name]
    model = get_proper_module(f'base_networks.{network_name}', class_name)
    m = model(res=res, classes=dim, with_projection_head=mlp, name=name)

    test_input_images = tf.random.normal(shape=[1, res, res, 3])
    _ = m(test_input_images)

    if not trainable:
        for layer in m.layers:
            layer.trainable = False
    return m


def main():
    resnet50_q = load_model(name='encoder_q', network_name='resnet50', res=224, dim=512, mlp=True, trainable=True)
    resnet50_k = load_model(name='encoder_k', network_name='resnet50', res=224, dim=512, mlp=True, trainable=False)
    print(resnet50_q.summary())
    print(resnet50_k.summary())
    return


if __name__ == '__main__':
    main()
