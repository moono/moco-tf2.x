import os
import tensorflow as tf
import tensorflow_datasets as tfds

from datasets.simclr_data_augmentation import (
    CROP_PROPORTION, center_crop, to_grayscale, color_jitter_nonrand,
    random_apply, random_crop_with_resize, random_color_jitter, random_blur
)


"""
All augmentation's input & output spec
input
    image: [None, None, 3] (0 ~ 255) uint8
    res: output resolution
output
    image: [res, res, 3] (0.0 ~ 1.0) float32
"""


def fit_batch(image, res, is_training):
    # convert to (0.0 ~ 1.0) float32 first
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if is_training:
        image = random_crop_with_resize(image, res, res)
    else:
        image = center_crop(image, res, res, crop_proportion=CROP_PROPORTION)
        image = tf.reshape(image, [res, res, 3])
        image = tf.clip_by_value(image, 0., 1.)
    return image


def augmentation_v1(image, res):
    image = random_apply(to_grayscale, p=0.2, x=image)
    image = color_jitter_nonrand(image, 0.4, 0.4, 0.4, 0.4)
    image = tf.image.random_flip_left_right(image)
    if image.shape.ndims == 3:
        image = tf.reshape(image, [res, res, 3])
    else:
        batch_size = tf.shape(image)[0]
        image = tf.reshape(image, [batch_size, res, res, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image


def augmentation_v2(image, res):
    image = random_color_jitter(image, color_jitter_strength=0.5)
    image = random_apply(to_grayscale, p=0.2, x=image)
    image = random_blur(image, res, res, p=0.5)
    image = tf.image.random_flip_left_right(image)
    if image.shape.ndims == 3:
        image = tf.reshape(image, [res, res, 3])
    else:
        batch_size = tf.shape(image)[0]
        image = tf.reshape(image, [batch_size, res, res, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image


def two_crops(image, res, aug_fn, is_training):
    image_q = tf.identity(image)
    image_k = tf.identity(image)

    image_q = fit_batch(image_q, res, is_training)
    image_k = fit_batch(image_k, res, is_training)

    if aug_fn is not None:
        image_q = aug_fn(image_q, res)
        image_k = aug_fn(image_k, res)
    return image_q, image_k


def get_dataset(tfds_data_dir, is_training, res, moco_ver, aug_op, batch_size, epochs=None):
    assert aug_op in ['CPU', 'GPU']

    dataset_name = 'imagenet2012'
    manual_dir = os.path.join(tfds_data_dir, 'manual')
    extract_dir = os.path.join(tfds_data_dir, dataset_name)

    # prepare
    dl_config = tfds.download.DownloadConfig(extract_dir=extract_dir, manual_dir=manual_dir)
    builder = tfds.builder(dataset_name, data_dir=tfds_data_dir)
    builder.download_and_prepare(download_config=dl_config)
    info = builder.info
    # print(info.features)

    # select augmentations
    if aug_op == 'GPU' or not is_training:
        augmentation_fn = None
    else:
        augmentation_fn = augmentation_v1 if moco_ver == 1 else augmentation_v2

    # instantiate tf.data.Dataset
    dataset_options = {
        'split': tfds.Split.TRAIN if is_training else tfds.Split.VALIDATION,
        'decoders': {'image': tfds.decode.SkipDecoding()},
        'shuffle_files': True,
        'as_supervised': True
    }
    dataset = builder.as_dataset(**dataset_options)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(map_func=lambda i, l: info.features['image'].decode_example(i),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(map_func=lambda i: two_crops(i, res, augmentation_fn, is_training),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def parse_fn(image, label, res, is_training):
    image = fit_batch(image, res, is_training)
    if is_training:
        image = tf.image.random_flip_left_right(image)
    image = tf.reshape(image, [res, res, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image, label


def get_dataset_lincls(tfds_data_dir, is_training, res, batch_size, epochs=None):
    dataset_name = 'imagenet2012'
    manual_dir = os.path.join(tfds_data_dir, 'manual')
    extract_dir = os.path.join(tfds_data_dir, dataset_name)

    # prepare
    dl_config = tfds.download.DownloadConfig(extract_dir=extract_dir, manual_dir=manual_dir)
    builder = tfds.builder(dataset_name, data_dir=tfds_data_dir)
    builder.download_and_prepare(download_config=dl_config)
    info = builder.info
    # print(info.features)

    # instantiate tf.data.Dataset
    dataset_options = {
        'split': tfds.Split.TRAIN if is_training else tfds.Split.VALIDATION,
        'decoders': {'image': tfds.decode.SkipDecoding()},
        'shuffle_files': True,
        'as_supervised': True
    }
    dataset = builder.as_dataset(**dataset_options)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(map_func=lambda i, l: (info.features['image'].decode_example(i), l),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(map_func=lambda i, l: parse_fn(i, l, res, is_training),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def test_unsupervised_training_dataset():
    import time
    from PIL import Image

    def postprocess_image(imgs):
        img = imgs * 255.0
        img = tf.cast(img, dtype=tf.uint8)
        img = img[0].numpy()
        img = Image.fromarray(img)
        img = img.convert('RGB')
        return img

    res = 224
    batch_size = 256
    is_training = True
    moco_ver = 2
    epochs = 1
    aug_op = 'GPU'
    tfds_data_dir = '/mnt/vision-nas/data-sets/tensorflow_datasets'
    dataset = get_dataset(tfds_data_dir, is_training, res, moco_ver, aug_op, batch_size, epochs)

    aug_fn = None
    if aug_op == 'GPU':
        aug_fn = augmentation_v1 if moco_ver == 1 else augmentation_v2

    t_start = time.time()
    for ii, (images_q, images_k) in enumerate(dataset.take(100)):
        # images_*: [batch_size, res, res, 3] (0.0 ~ 1.0) float32

        # im_q = postprocess_image(images_q)
        # im_k = postprocess_image(images_k)
        # im_q.show()
        # im_k.show()

        if aug_fn is not None:
            images_q_aug = aug_fn(images_q, res)
            images_k_aug = aug_fn(images_k, res)
            # im_q_aug = postprocess_image(images_q_aug)
            # im_k_aug = postprocess_image(images_k_aug)
            # im_q_aug.show()
            # im_k_aug.show()

        if ii % 10 == 0:
            elapsed = time.time() - t_start

            print(f'[{elapsed:.3f}]: {ii * batch_size}')
            t_start = time.time()
    return


def test_lincls_dataset():
    from PIL import Image

    def postprocess_image(imgs):
        img = imgs * 255.0
        img = tf.cast(img, dtype=tf.uint8)
        img = img[0].numpy()
        img = Image.fromarray(img)
        img = img.convert('RGB')
        return img

    res = 224
    batch_size = 256
    tfds_data_dir = '/mnt/vision-nas/data-sets/tensorflow_datasets'
    t_dataset = get_dataset_lincls(tfds_data_dir, True, res, batch_size, epochs=1)
    v_dataset = get_dataset_lincls(tfds_data_dir, False, res, batch_size, epochs=1)

    for images, labels in t_dataset.take(4):
        # images: [batch_size, res, res, 3] (0.0 ~ 1.0) float32
        print(labels[0])
        img = postprocess_image(images)
        img.show()

    for images, labels in v_dataset.take(4):
        # images: [batch_size, res, res, 3] (0.0 ~ 1.0) float32
        print(labels[0])
        img = postprocess_image(images)
        img.show()
    return


def main():
    # test_unsupervised_training_dataset()
    test_lincls_dataset()
    return


if __name__ == '__main__':
    main()

# CPU MoCo ver 1
# [12.188]: 0
# [4.440]: 2560
# [4.219]: 5120
# [4.286]: 7680
# [4.308]: 10240
# [4.217]: 12800
# [4.284]: 15360
# [4.242]: 17920
# [4.378]: 20480
# [4.331]: 23040

# GPU MoCo ver 1
# [13.545]: 0
# [5.640]: 2560
# [5.722]: 5120
# [5.841]: 7680
# [5.944]: 10240
# [5.862]: 12800
# [5.931]: 15360
# [5.856]: 17920
# [5.845]: 20480
# [5.993]: 23040
# [5.878]: 25600

# CPU MoCo ver 2
# [14.570]: 0
# [17.250]: 2560
# [14.522]: 5120
# [14.805]: 7680
# [14.459]: 10240
# [14.644]: 12800
# [14.393]: 15360
# [14.791]: 17920

# GPU MoCo ver 2
# [13.584]: 0
# [5.478]: 2560
# [5.225]: 5120
# [5.835]: 7680
# [5.622]: 10240
# [6.214]: 12800
# [5.870]: 15360
# [5.915]: 17920
# [5.878]: 20480
# [5.216]: 23040
# [5.093]: 25600
# [5.237]: 28160
# [5.554]: 30720
