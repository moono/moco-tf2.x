import importlib
import tensorflow as tf


def get_proper_module(module_name, object_name):
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    return obj


# # not working??
# # https://stackoverflow.com/questions/41260042/global-weight-decay-in-keras
# def add_weight_decay(model, weight_decay):
#     if (weight_decay is None) or (weight_decay == 0.0):
#         return
#
#     # recursion inside the model
#     def add_decay_loss(m, factor):
#         if isinstance(m, tf.keras.Model):
#             for layer in m.layers:
#                 add_decay_loss(layer, factor)
#         else:
#             for param in m.trainable_weights:
#                 with tf.keras.backend.name_scope('weight_regularizer'):
#                     regularizer = lambda: tf.keras.regularizers.l2(factor)(param)
#                     m.add_loss(regularizer)
#
#     # weight decay and l2 regularization differs by a factor of 2
#     add_decay_loss(model, weight_decay/2.0)
#     return

def set_not_trainable(model):
    def set_model(m):
        if isinstance(m, tf.keras.Model):
            for layer in m.layers:
                set_model(layer)
        else:
            if not isinstance(m, tf.keras.layers.BatchNormalization):
                m.trainable = False
        return

    set_model(model)
    return


def load_model(name, network_name, network_params, trainable):
    class_name_table = {
        'resnet50': 'Resnet50',
    }
    class_name = class_name_table[network_name]
    model = get_proper_module(f'base_networks.{network_name}', class_name)
    m = model(network_params, name=name)
    m.build((None, *network_params['input_shape']))

    # set trainable or not
    if not trainable:
        # set_not_trainable(m)
        for layer in m.layers:
            layer.trainable = False
    return m


def main():
    from misc.tf_utils import allow_memory_growth

    allow_memory_growth()

    resnet_params = {
        'input_shape': [224, 224, 3],
        'dim': 512,
        'mlp': True,
        'w_decay': 0.0001,
    }

    resnet50_q = load_model(name='encoder_q', network_name='resnet50', network_params=resnet_params, trainable=True)
    resnet50_k = load_model(name='encoder_k', network_name='resnet50', network_params=resnet_params, trainable=False)
    print(resnet50_q.summary())
    print(resnet50_k.summary())
    return


if __name__ == '__main__':
    main()
