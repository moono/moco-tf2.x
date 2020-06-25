import importlib


def get_proper_module(module_name, object_name):
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    return obj


def load_model(model_name, res, dim, mlp, trainable):
    loader = get_proper_module(f'base_networks.{model_name}', 'get_model')
    model = loader(res=res, classes=dim, with_projection_head=mlp)

    if not trainable:
        for layer in model.layers:
            layer.trainable = False
    return model


def main():
    resnet50_q = load_model('resnet50', res=224, dim=512, mlp=True, trainable=True)
    resnet50_k = load_model('resnet50', res=224, dim=512, mlp=True, trainable=False)
    print(resnet50_q.summary())
    print(resnet50_k.summary())
    return


if __name__ == '__main__':
    main()
