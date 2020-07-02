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


def augmentation_eval(image, res):
    # convert to (0.0 ~ 1.0) float32 first
    # to fit SimCLR's image augmentation code
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = center_crop(image, res, res, crop_proportion=CROP_PROPORTION)
    image = tf.reshape(image, [res, res, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image


def augmentation_v1(image, res):
    # convert to (0.0 ~ 1.0) float32 first
    # to fit SimCLR's image augmentation code
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = random_crop_with_resize(image, res, res)
    image = random_apply(to_grayscale, p=0.2, x=image)
    image = color_jitter_nonrand(image, 0.4, 0.4, 0.4, 0.4)
    image = tf.image.random_flip_left_right(image)
    image = tf.reshape(image, [res, res, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image


def augmentation_v2(image, res):
    # convert to (0.0 ~ 1.0) float32 first
    # to fit SimCLR's image augmentation code
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = random_crop_with_resize(image, res, res)
    image = random_color_jitter(image, color_jitter_strength=0.5)
    image = random_blur(image, res, res, p=0.5)
    image = tf.image.random_flip_left_right(image)
    image = tf.reshape(image, [res, res, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image


def two_crops(image, res, aug_fn):
    image_q = tf.identity(image)
    image_k = tf.identity(image)

    image_q = aug_fn(image_q, res)
    image_k = aug_fn(image_k, res)
    return image_q, image_k


def get_dataset(tfds_data_dir, is_training, res, moco_ver, batch_size, epochs=None):
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
    if is_training and moco_ver == 1:
        augmentation_fn = augmentation_v1
    elif is_training and moco_ver == 2:
        augmentation_fn = augmentation_v2
    else:   # eval
        augmentation_fn = augmentation_eval

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
    dataset = dataset.map(map_func=lambda i: two_crops(i, res, augmentation_fn),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def main():
    from PIL import Image

    res = 224
    batch_size = 32
    is_training = True
    moco_ver = 1
    epochs = 1
    tfds_data_dir = '/mnt/vision-nas/data-sets/tensorflow_datasets'
    dataset = get_dataset(tfds_data_dir, is_training, res, moco_ver, batch_size, epochs)

    def postprocess_image(imgs):
        img = imgs * 255.0
        img = tf.cast(img, dtype=tf.uint8)
        img = img[0].numpy()
        img = Image.fromarray(img)
        img = img.convert('RGB')
        return img

    for images_q, images_k in dataset.take(1):
        # images_*: [batch_size, res, res, 3] (0.0 ~ 1.0) float32

        im_q = postprocess_image(images_q)
        im_q.show()

        im_k = postprocess_image(images_k)
        im_k.show()
    return


if __name__ == '__main__':
    main()
