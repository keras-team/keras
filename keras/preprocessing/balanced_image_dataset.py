# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras image dataset loading utilities."""

import tensorflow.compat.v2 as tf
# pylint: disable=g-classes-have-attributes

import numpy as np
from keras.layers.preprocessing import image_preprocessing
from keras.preprocessing import dataset_utils
from keras.preprocessing import image as keras_image_ops
from keras.preprocessing.image_dataset import load_image
from numpy.random import default_rng
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

ALLOWLIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')


@keras_export('keras.utils.balanced_image_dataset_from_directory',
              'keras.preprocessing.balanced_image_dataset_from_directory',
              v1=[])

def balanced_image_dataset_from_directory(directory,
                                 num_classes_per_batch=2, 
                                 num_images_per_class=16,
                                 labels='inferred',
                                 label_mode='int',
                                 class_names=None,
                                 color_mode='rgb',
                                 image_size=(256, 256),
                                 shuffle=True,
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 safe_triplet=False,
                                 samples_per_epoch=None,
                                 interpolation='bilinear',
                                 follow_links=False,
                                 crop_to_aspect_ratio=False,
                                 **kwargs):
  """Generates a balanced per batch `tf.data.Dataset` from image files in a directory.

  If your directory structure is:

  ```
  main_directory/
  ...class_a/
  ......a_image_1.jpg
  ......a_image_2.jpg
  ...class_b/
  ......b_image_1.jpg
  ......b_image_2.jpg
  ```

  Then calling `balanced_image_dataset_from_directory(main_directory, labels='inferred')`
  will return a `tf.data.Dataset` that yields batches of images from
  the subdirectories `class_a` and `class_b`, together with labels
  0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

  Supported image formats: jpeg, png, bmp, gif.
  Animated gifs are truncated to the first frame.

  Setting repeat to True does not guarantee that every epoch will include all
  different samples from the dataset. But as sampling is weighted per class, 
  every epoch will include a very high percentage of the dataset and should
  approach 100% as dataset size increases. This however guarantee that both
  num_classes_per_batch and num_images_per_class are fixed for all batches 
  including later ones.

  Batch size is calculated by multiplying num_classes_per_batch and num_images_per_class.

  Args:
    directory: Directory where the data is located.
        If `labels` is "inferred", it should contain
        subdirectories, each containing images for a class.
        Otherwise, the directory structure is ignored.
    num_classes_per_batch: Number of different classes to include per batch
        This can only be guaranteed in later batches if safe_triplet is set to True.
    num_images_per_class: Number of samples per class per batch.
        This can only be guaranteed in later batches if safe_triplet is set to True.
    labels:
        - 'inferred': labels are generated from the directory structure,
        - None (no labels).
            Images with no associated labels are ignored 
            as labels are needed to generated balanced batches,
        - a list/tuple of integer labels of the same size as the number of
            image files found in the directory. Labels should be sorted according
            to the alphanumeric order of the image file paths
            (obtained via `os.walk(directory)` in Python).
    label_mode:
        - 'int': means that the labels are encoded as integers
            (e.g. for `sparse_categorical_crossentropy` loss).
        - 'categorical' means that the labels are
            encoded as a categorical vector
            (e.g. for `categorical_crossentropy` loss).
        - 'binary' means that the labels (there can be only 2)
            are encoded as `float32` scalars with values 0 or 1
            (e.g. for `binary_crossentropy`).
        - None (no labels). Images with no associated labels are ignored
            as labels are needed to generated balanced batches.
    class_names: Only valid if "labels" is "inferred". This is the explicit
        list of class names (must match names of subdirectories). Used
        to control the order of the classes
        (otherwise alphanumerical order is used).
    color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
        Whether the images will be converted to
        have 1, 3, or 4 channels.
    image_size: Size to resize images to after they are read from disk.
        Defaults to `(256, 256)`.
        Since the pipeline processes batches of images that must all have
        the same size, this must be provided.
    shuffle: Whether to shuffle the data before separating classes datasets 
        Default: True. If set to False, samples are still randomly drawn, 
        but not shuffled before sampling.
    seed: Optional random seed for shuffling and transformations.
    validation_split: Optional float between 0 and 1,
        fraction of data to reserve for validation.
    subset: One of "training" or "validation".
        Only used if `validation_split` is set.
    safe_triplet: If True, datasets are repeated in order to include triplets
        in every batch by enforcing num_classes_per_batch and 
        num_images_per_class even in later batches. This however does not 
        guarantee that the whole dataset is used in every epoch. However, as
        the sampling is weighted, the majority of the data is utilized. This 
        can approach 100% if the dataset size is big enough. Default: False
    samples_per_epoch: Only valid safe_triplet is True. This is the number of images
    to use per epoch as dataset is infinite. If left to default 'None', it is
    calculated to be the number of images available.
    interpolation: String, the interpolation method used when resizing images.
      Defaults to `bilinear`. Supports `bilinear`, `nearest`, `bicubic`,
      `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
    follow_links: Whether to visit subdirectories pointed to by symlinks.
        Defaults to False.
    crop_to_aspect_ratio: If True, resize the images without aspect
      ratio distortion. When the original aspect ratio differs from the target
      aspect ratio, the output image will be cropped so as to return the largest
      possible window in the image (of size `image_size`) that matches
      the target aspect ratio. By default (`crop_to_aspect_ratio=False`),
      aspect ratio may not be preserved.
    **kwargs: Legacy keyword arguments.

  Returns:
    A `tf.data.Dataset` object.
      - If `label_mode` is None, it yields `float32` tensors of shape
        `(num_classes_per_batch*num_images_per_class, image_size[0], image_size[1], num_channels)`,
        encoding images (see below for rules regarding `num_channels`).
      - Otherwise, it yields a tuple `(images, labels)`, where `images`
        has shape `(num_classes_per_batch*num_images_per_class, image_size[0], image_size[1], num_channels)`,
        and `labels` follows the format described below.

  Rules regarding labels format:
    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
      `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
      1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorial`, the labels are a `float32` tensor
      of shape `(batch_size, num_classes)`, representing a one-hot
      encoding of the class index.

  Rules regarding number of channels in the yielded images:
    - if `color_mode` is `grayscale`,
      there's 1 channel in the image tensors.
    - if `color_mode` is `rgb`,
      there are 3 channel in the image tensors.
    - if `color_mode` is `rgba`,
      there are 4 channel in the image tensors.

  """
  if 'smart_resize' in kwargs:
    crop_to_aspect_ratio = kwargs.pop('smart_resize')
  if kwargs:
    raise TypeError(f'Unknown keywords argument(s): {tuple(kwargs.keys())}')
  if labels not in ('inferred', None):
    if not isinstance(labels, (list, tuple)):
      raise ValueError(
          '`labels` argument should be a list/tuple of integer labels, of '
          'the same size as the number of image files in the target '
          'directory. If you wish to infer the labels from the subdirectory '
          'names in the target directory, pass `labels="inferred"`. '
          'If you wish to get a dataset that only contains images '
          f'(no labels), pass `labels=None`. Received: labels={labels}')
    if class_names:
      raise ValueError('You can only pass `class_names` if '
                       f'`labels="inferred"`. Received: labels={labels}, and '
                       f'class_names={class_names}')
  if label_mode not in {'int', 'categorical', 'binary', None}:
    raise ValueError(
        '`label_mode` argument must be one of "int", "categorical", "binary", '
        f'or None. Received: label_mode={label_mode}')
  if labels is None or label_mode is None:
    labels = 'inferred'
    label_mode = None
    logging.warning(
      'Passing `labels=None` or `label_mode=None` will ignore all images '
      'not associated with a label. If you want to generate all images '
      'regardless of their labels please use '
      'keras.preprocessing.image_dataset_from_directory instead.'
    )
  if color_mode == 'rgb':
    num_channels = 3
  elif color_mode == 'rgba':
    num_channels = 4
  elif color_mode == 'grayscale':
    num_channels = 1
  else:
    raise ValueError(
        '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
        f'Received: color_mode={color_mode}')
  interpolation = image_preprocessing.get_interpolation(interpolation)
  dataset_utils.check_validation_split_arg(
      validation_split, subset, shuffle, seed)

  if seed is None:
    seed = np.random.randint(1e6)
  image_paths, labels, class_names = dataset_utils.index_directory(
      directory,
      labels,
      formats=ALLOWLIST_FORMATS,
      class_names=class_names,
      shuffle=shuffle,
      seed=seed,
      follow_links=follow_links)

  if label_mode == 'binary' and len(class_names) != 2:
    raise ValueError(
        f'When passing `label_mode="binary"`, there must be exactly 2 '
        f'class_names. Received: class_names={class_names}')

  if not safe_triplet:
    if samples_per_epoch is not None:
      raise ValueError(
        f'You can only pass `samples_per_epoch` if '
        f'safe_triplet is set to False '
        f'Received: safe_triplet={safe_triplet}, and '
        f'samples_per_epoch={samples_per_epoch}')
  else:
    if not isinstance(samples_per_epoch, int) and samples_per_epoch is not None:
      raise ValueError(
        f'`samples_per_epoch` should only be of '
        f'type integer. Received type={type(samples_per_epoch)}')

  image_paths, labels = dataset_utils.get_training_or_validation_split(
      image_paths, labels, validation_split, subset)

  if not image_paths:
    raise ValueError(f'No images found in directory {directory}. '
                     f'Allowed formats: {ALLOWLIST_FORMATS}')

  dataset = paths_and_labels_to_dataset(
      image_paths=image_paths,
      image_size=image_size,
      num_channels=num_channels,
      labels=labels,
      label_mode=label_mode,
      num_classes=len(class_names),
      interpolation=interpolation,
      num_classes_per_batch=num_classes_per_batch,
      num_images_per_class=num_images_per_class,
      safe_triplet=safe_triplet,
      seed=seed,
      samples_per_epoch=samples_per_epoch,
      crop_to_aspect_ratio=crop_to_aspect_ratio)
  
  batch_size = int(num_classes_per_batch * num_images_per_class)
  dataset = dataset.prefetch(tf.data.AUTOTUNE).batch(batch_size)
  # Users may need to reference `class_names`.
  dataset.class_names = class_names
  # Include file paths for images as attribute.
  dataset.file_paths = image_paths
  return dataset


def paths_and_labels_to_dataset(image_paths,
                                image_size,
                                num_channels,
                                labels,
                                label_mode,
                                num_classes,
                                interpolation,
                                num_classes_per_batch,
                                num_images_per_class,
                                safe_triplet,
                                seed,
                                samples_per_epoch,
                                crop_to_aspect_ratio=False):
  """Constructs a dataset of images and labels."""
  # TODO(fchollet): consider making num_parallel_calls settable
  
  image_paths = np.array(image_paths)
  labels = np.array(labels)
  unique_labels, counts = np.unique(labels, return_counts=True)
  num_samples = counts.sum()
  labels_probability = counts / num_samples
  label_indexes = [np.where(labels == label)[0] for label in unique_labels]
  image_paths_per_label = [image_paths[idx].tolist() for idx in label_indexes]
  labels_per_label = [labels[idx].tolist() for idx in label_indexes]
  paths_datasets = [tf.data.Dataset.from_tensor_slices(x) 
                    for x in image_paths_per_label]

  args = (image_size, num_channels, interpolation, crop_to_aspect_ratio)

  label_datasets = [dataset_utils.labels_to_dataset(
      labels, label_mode, num_classes) for labels in labels_per_label]

  if safe_triplet:
    paths_datasets = [tf.data.Dataset.zip((path_ds, label_ds)).repeat() 
    for (path_ds, label_ds) in zip(paths_datasets, label_datasets)]
  else:
    paths_datasets = [tf.data.Dataset.zip((path_ds, label_ds)) 
    for (path_ds, label_ds) in zip(paths_datasets, label_datasets)]

  if num_classes_per_batch > len(unique_labels):
      raise ValueError(
        f'num_classes_per_batch must be less than number of available '
        f'classes in the dataset (or dataset split). '
        f'Received: num_classes_per_batch={num_classes_per_batch} '
        f'but available classes={len(unique_labels)}. '
        f'Try reducing `num_classes_per_batch` or increasing dataset samples.'
        )
  choice_dataset = tf.data.Dataset.from_generator(
      generator,
      output_types=tf.int64,
      args=(range(len(unique_labels)), num_classes_per_batch, 
            num_images_per_class, labels_probability, seed))
  balanced_path_dataset = tf.data.Dataset.choose_from_datasets(paths_datasets, 
                                                                choice_dataset, 
                                                                stop_on_empty_dataset=safe_triplet)
  if safe_triplet:
    if samples_per_epoch is None:
      multiple = int(num_classes_per_batch * num_images_per_class)
      x = num_samples + (multiple - 1)
      samples_per_epoch =  x - (x % multiple)
    balanced_path_dataset = balanced_path_dataset.take(samples_per_epoch)
  balanced_img_dataset = balanced_path_dataset.map(
      lambda x, y: load_image(x, y, *args))
  if label_mode is None:
    balanced_img_dataset = balanced_img_dataset.map(lambda x, y: x)
  return balanced_img_dataset


def generator(choice_indexes, 
              num_classes_per_batch, 
              num_images_per_class, 
              labels_probability, 
              seed):
  rng = default_rng(seed=seed)
  while True:
    labels = rng.choice(choice_indexes,
                            num_classes_per_batch,
                            replace=False,
                            p=labels_probability)
    labels = labels.repeat(num_images_per_class)
    for label in labels:
        yield label


def load_image(path, label, image_size, num_channels, interpolation,
               crop_to_aspect_ratio=False):
  """Load an image from a path and resize it."""
  img = tf.io.read_file(path)
  img = tf.image.decode_image(
      img, channels=num_channels, expand_animations=False)
  if crop_to_aspect_ratio:
    img = keras_image_ops.smart_resize(img, image_size,
                                       interpolation=interpolation)
  else:
    img = tf.image.resize(img, image_size, method=interpolation)
  img.set_shape((image_size[0], image_size[1], num_channels))
  return img, label
