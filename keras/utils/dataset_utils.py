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

import multiprocessing
import os

import numpy as np


def index_directory(directory,
                    labels,
                    formats,
                    class_names=None,
                    shuffle=True,
                    seed=None,
                    follow_links=False):
  """Make list of all files in the subdirs of `directory`, with their labels.

  Args:
    directory: The target directory (string).
    labels: Either "inferred"
        (labels are generated from the directory structure),
        None (no labels),
        or a list/tuple of integer labels of the same size as the number of
        valid files found in the directory. Labels should be sorted according
        to the alphanumeric order of the image file paths
        (obtained via `os.walk(directory)` in Python).
    formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").
    class_names: Only valid if "labels" is "inferred". This is the explicit
        list of class names (must match names of subdirectories). Used
        to control the order of the classes
        (otherwise alphanumerical order is used).
    shuffle: Whether to shuffle the data. Default: True.
        If set to False, sorts the data in alphanumeric order.
    seed: Optional random seed for shuffling.
    follow_links: Whether to visits subdirectories pointed to by symlinks.

  Returns:
    tuple (file_paths, labels, class_names).
      file_paths: list of file paths (strings).
      labels: list of matching integer labels (same length as file_paths)
      class_names: names of the classes corresponding to these labels, in order.
  """
  if labels is None:
    # in the no-label case, index from the parent directory down.
    subdirs = ['']
    class_names = subdirs
  else:
    subdirs = []
    for subdir in sorted(tf.io.gfile.listdir(directory)):
      if tf.io.gfile.isdir(tf.io.gfile.join(directory, subdir)):
        if subdir.endswith('/'):
          subdir = subdir[:-1]
        subdirs.append(subdir)
    if not class_names:
      class_names = subdirs
    else:
      if set(class_names) != set(subdirs):
        raise ValueError(
            'The `class_names` passed did not match the '
            'names of the subdirectories of the target directory. '
            'Expected: %s, but received: %s' %
            (subdirs, class_names))
  class_indices = dict(zip(class_names, range(len(class_names))))

  # Build an index of the files
  # in the different class subfolders.
  pool = multiprocessing.pool.ThreadPool()
  results = []
  filenames = []

  for dirpath in (tf.io.gfile.join(directory, subdir) for subdir in subdirs):
    results.append(
        pool.apply_async(index_subdirectory,
                         (dirpath, class_indices, follow_links, formats)))
  labels_list = []
  for res in results:
    partial_filenames, partial_labels = res.get()
    labels_list.append(partial_labels)
    filenames += partial_filenames
  if labels not in ('inferred', None):
    if len(labels) != len(filenames):
      raise ValueError('Expected the lengths of `labels` to match the number '
                       'of files in the target directory. len(labels) is %s '
                       'while we found %s files in %s.' % (
                           len(labels), len(filenames), directory))
  else:
    i = 0
    labels = np.zeros((len(filenames),), dtype='int32')
    for partial_labels in labels_list:
      labels[i:i + len(partial_labels)] = partial_labels
      i += len(partial_labels)

  if labels is None:
    print('Found %d files.' % (len(filenames),))
  else:
    print('Found %d files belonging to %d classes.' %
          (len(filenames), len(class_names)))
  pool.close()
  pool.join()
  file_paths = [tf.io.gfile.join(directory, fname) for fname in filenames]

  if shuffle:
    # Shuffle globally to erase macro-structure
    if seed is None:
      seed = np.random.randint(1e6)
    rng = np.random.RandomState(seed)
    rng.shuffle(file_paths)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)
  return file_paths, labels, class_names


def iter_valid_files(directory, follow_links, formats):
  if not follow_links:
    walk = tf.io.gfile.walk(directory)
  else:
    walk = os.walk(directory, followlinks=follow_links)
  for root, _, files in sorted(walk, key=lambda x: x[0]):
    for fname in sorted(files):
      if fname.lower().endswith(formats):
        yield root, fname


def index_subdirectory(directory, class_indices, follow_links, formats):
  """Recursively walks directory and list image paths and their class index.

  Args:
    directory: string, target directory.
    class_indices: dict mapping class names to their index.
    follow_links: boolean, whether to recursively follow subdirectories
      (if False, we only list top-level images in `directory`).
    formats: Allowlist of file extensions to index (e.g. ".jpg", ".txt").

  Returns:
    tuple `(filenames, labels)`. `filenames` is a list of relative file
      paths, and `labels` is a list of integer labels corresponding to these
      files.
  """
  dirname = os.path.basename(directory)
  valid_files = iter_valid_files(directory, follow_links, formats)
  labels = []
  filenames = []
  for root, fname in valid_files:
    labels.append(class_indices[dirname])
    absolute_path = tf.io.gfile.join(root, fname)
    relative_path = tf.io.gfile.join(
        dirname, os.path.relpath(absolute_path, directory))
    filenames.append(relative_path)
  return filenames, labels


def get_training_or_validation_split(samples, labels, validation_split, subset):
  """Potentially restict samples & labels to a training or validation split.

  Args:
    samples: List of elements.
    labels: List of corresponding labels.
    validation_split: Float, fraction of data to reserve for validation.
    subset: Subset of the data to return.
      Either "training", "validation", or None. If None, we return all of the
      data.

  Returns:
    tuple (samples, labels), potentially restricted to the specified subset.
  """
  if not validation_split:
    return samples, labels

  num_val_samples = int(validation_split * len(samples))
  if subset == 'training':
    print('Using %d files for training.' % (len(samples) - num_val_samples,))
    samples = samples[:-num_val_samples]
    labels = labels[:-num_val_samples]
  elif subset == 'validation':
    print('Using %d files for validation.' % (num_val_samples,))
    samples = samples[-num_val_samples:]
    labels = labels[-num_val_samples:]
  else:
    raise ValueError('`subset` must be either "training" '
                     'or "validation", received: %s' % (subset,))
  return samples, labels


def labels_to_dataset(labels, label_mode, num_classes):
  """Create a tf.data.Dataset from the list/tuple of labels.

  Args:
    labels: list/tuple of labels to be converted into a tf.data.Dataset.
    label_mode: String describing the encoding of `labels`. Options are:
    - 'binary' indicates that the labels (there can be only 2) are encoded as
      `float32` scalars with values 0 or 1 (e.g. for `binary_crossentropy`).
    - 'categorical' means that the labels are mapped into a categorical vector.
      (e.g. for `categorical_crossentropy` loss).
    num_classes: number of classes of labels.

  Returns:
    A `Dataset` instance.
  """
  label_ds = tf.data.Dataset.from_tensor_slices(labels)
  if label_mode == 'binary':
    label_ds = label_ds.map(
        lambda x: tf.expand_dims(tf.cast(x, 'float32'), axis=-1),
        num_parallel_calls=tf.data.AUTOTUNE)
  elif label_mode == 'categorical':
    label_ds = label_ds.map(lambda x: tf.one_hot(x, num_classes),
                            num_parallel_calls=tf.data.AUTOTUNE)
  return label_ds


def check_validation_split_arg(validation_split, subset, shuffle, seed):
  """Raise errors in case of invalid argument values.

  Args:
    validation_split: float between 0 and 1, fraction of data to reserve for
      validation.
    subset: One of "training" or "validation". Only used if `validation_split`
      is set.
    shuffle: Whether to shuffle the data. Either True or False.
    seed: random seed for shuffling and transformations.
  """
  if validation_split and not 0 < validation_split < 1:
    raise ValueError(
        '`validation_split` must be between 0 and 1, received: %s' %
        (validation_split,))
  if (validation_split or subset) and not (validation_split and subset):
    raise ValueError(
        'If `subset` is set, `validation_split` must be set, and inversely.')
  if subset not in ('training', 'validation', None):
    raise ValueError('`subset` must be either "training" '
                     'or "validation", received: %s' % (subset,))
  if validation_split and shuffle and seed is None:
    raise ValueError(
        'If using `validation_split` and shuffling the data, you must provide '
        'a `seed` argument, to make sure that there is no overlap between the '
        'training and validation subset.')
