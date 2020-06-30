import tensorflow as tf


def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return


def split_gpu_for_testing(mem_in_gb=4):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * mem_in_gb),
                 tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * mem_in_gb)]
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
        except RuntimeError as e:
            print(e)
    return


def main():
    # allow_memory_growth()
    split_gpu_for_testing()
    return


if __name__ == '__main__':
    main()
