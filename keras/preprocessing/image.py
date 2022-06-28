# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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


"""Utilies for image preprocessing and augmentation.

Deprecated: `tf.keras.preprocessing.image` APIs do not operate on tensors and
are not recommended for new code. Prefer loading data with
`tf.keras.utils.image_dataset_from_directory`, and then transforming the output
`tf.data.Dataset` with preprocessing layers. For more information, see the
tutorials for [loading images](
https://www.tensorflow.org/tutorials/load_data/images) and [augmenting images](
https://www.tensorflow.org/tutorials/images/data_augmentation), as well as the
[preprocessing layer guide](
https://www.tensorflow.org/guide/keras/preprocessing_layers).
"""

import collections
import multiprocessing
import os
import threading
import warnings

import numpy as np

from keras import backend
from keras.utils import data_utils
from keras.utils import image_utils
from keras.utils import io_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export

try:
    import scipy
    from scipy import linalg  # noqa: F401
    from scipy import ndimage  # noqa: F401
except ImportError:
    pass
try:
    from PIL import ImageEnhance
except ImportError:
    ImageEnhance = None


@keras_export("keras.preprocessing.image.Iterator")
class Iterator(data_utils.Sequence):
    """Base class for image data iterators.

    Deprecated: `tf.keras.preprocessing.image.Iterator` is not recommended for
    new code. Prefer loading images with
    `tf.keras.utils.image_dataset_from_directory` and transforming the output
    `tf.data.Dataset` with preprocessing layers. For more information, see the
    tutorials for [loading images](
    https://www.tensorflow.org/tutorials/load_data/images) and
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    Args:
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    white_list_formats = ("png", "jpg", "jpeg", "bmp", "ppm", "tif", "tiff")

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError(
                "Asked to retrieve element {idx}, "
                "but the Sequence "
                "has length {length}".format(idx=idx, length=len(self))
            )
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[
            self.batch_size * idx : self.batch_size * (idx + 1)
        ]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                # Avoiding modulo by zero error
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[
                current_index : current_index + self.batch_size
            ]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 2.x.

        Returns:
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        Args:
            index_array: Array of sample indices to include in batch.
        Returns:
            A batch of transformed samples.
        """
        raise NotImplementedError


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension.

    Args:
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean, follow symbolic links to subdirectories.
    Yields:
        Tuple of (root, filename) with extension in `white_list_formats`.
    """

    def _recursive_list(subpath):
        return sorted(
            os.walk(subpath, followlinks=follow_links), key=lambda x: x[0]
        )

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            if fname.lower().endswith(".tiff"):
                warnings.warn(
                    'Using ".tiff" files with multiple bands '
                    "will cause distortion. Please verify your output."
                )
            if fname.lower().endswith(white_list_formats):
                yield root, fname


def _list_valid_filenames_in_directory(
    directory, white_list_formats, split, class_indices, follow_links
):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.

    Args:
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean, follow symbolic links to subdirectories.

    Returns:
         classes: a list of class indices
         filenames: the path of valid files in `directory`, relative from
             `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be
            `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        all_files = list(
            _iter_valid_files(directory, white_list_formats, follow_links)
        )
        num_files = len(all_files)
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = all_files[start:stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links
        )
    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory)
        )
        filenames.append(relative_path)

    return classes, filenames


class BatchFromFilesMixin:
    """Adds methods related to getting batches from filenames.

    It includes the logic to transform image files to batches.
    """

    def set_processing_attrs(
        self,
        image_data_generator,
        target_size,
        color_mode,
        data_format,
        save_to_dir,
        save_prefix,
        save_format,
        subset,
        interpolation,
        keep_aspect_ratio,
    ):
        """Sets attributes to use later for processing files into a batch.

        Args:
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            target_size: tuple of integers, dimensions to resize input images
            to.
            color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
                Color mode to read images.
            data_format: String, one of `channels_first`, `channels_last`.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images
                (if `save_to_dir` is set).
            subset: Subset of data (`"training"` or `"validation"`) if
                validation_split is set in ImageDataGenerator.
            interpolation: Interpolation method used to resample the image if
                the target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic". If
                PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
            keep_aspect_ratio: Boolean, whether to resize images to a target
                size without aspect ratio distortion. The image is cropped in
                the center with target aspect ratio before resizing.
        """
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.keep_aspect_ratio = keep_aspect_ratio
        if color_mode not in {"rgb", "rgba", "grayscale"}:
            raise ValueError(
                "Invalid color mode:",
                color_mode,
                '; expected "rgb", "rgba", or "grayscale".',
            )
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == "rgba":
            if self.data_format == "channels_last":
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == "rgb":
            if self.data_format == "channels_last":
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == "channels_last":
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == "validation":
                split = (0, validation_split)
            elif subset == "training":
                split = (validation_split, 1)
            else:
                raise ValueError(
                    "Invalid subset name: %s;"
                    'expected "training" or "validation"' % (subset,)
                )
        else:
            split = None
        self.split = split
        self.subset = subset

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        Args:
            index_array: Array of sample indices to include in batch.
        Returns:
            A batch of transformed samples.
        """
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape, dtype=self.dtype
        )
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = image_utils.load_img(
                filepaths[j],
                color_mode=self.color_mode,
                target_size=self.target_size,
                interpolation=self.interpolation,
                keep_aspect_ratio=self.keep_aspect_ratio,
            )
            x = image_utils.img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, "close"):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = image_utils.array_to_img(
                    batch_x[i], self.data_format, scale=True
                )
                fname = "{prefix}_{index}_{hash}.{format}".format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format,
                )
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == "input":
            batch_y = batch_x.copy()
        elif self.class_mode in {"binary", "sparse"}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == "categorical":
            batch_y = np.zeros(
                (len(batch_x), len(self.class_indices)), dtype=self.dtype
            )
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.0
        elif self.class_mode == "multi_output":
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == "raw":
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]

    @property
    def filepaths(self):
        """List of absolute paths to image files."""
        raise NotImplementedError(
            "`filepaths` property method has not "
            "been implemented in {}.".format(type(self).__name__)
        )

    @property
    def labels(self):
        """Class labels of every observation."""
        raise NotImplementedError(
            "`labels` property method has not been implemented in {}.".format(
                type(self).__name__
            )
        )

    @property
    def sample_weight(self):
        raise NotImplementedError(
            "`sample_weight` property method has not "
            "been implemented in {}.".format(type(self).__name__)
        )


@keras_export("keras.preprocessing.image.DirectoryIterator")
class DirectoryIterator(BatchFromFilesMixin, Iterator):
    """Iterator capable of reading images from a directory on disk.

    Deprecated: `tf.keras.preprocessing.image.DirectoryIterator` is not
    recommended for new code. Prefer loading images with
    `tf.keras.utils.image_dataset_from_directory` and transforming the output
    `tf.data.Dataset` with preprocessing layers. For more information, see the
    tutorials for [loading images](
    https://www.tensorflow.org/tutorials/load_data/images) and
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        directory: Path to the directory to read images from. Each subdirectory
          in this directory will be considered to contain images from one class,
          or alternatively you could specify class subdirectories via the
          `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator` to use for random
          transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`. Color mode to read
          images.
        classes: Optional list of strings, names of subdirectories containing
          images from each class (e.g. `["dogs", "cats"]`). It will be computed
          automatically if not set.
        class_mode: Mode for yielding the targets:
            - `"binary"`: binary targets (if there are only two classes),
            - `"categorical"`: categorical targets,
            - `"sparse"`: integer targets,
            - `"input"`: targets are images identical to input images (mainly
              used to work with autoencoders),
            - `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures being
          yielded, in a viewable format. This is useful for visualizing the
          random transformations being applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample images (if
          `save_to_dir` is set).
        save_format: Format to use for saving sample images (if `save_to_dir` is
          set).
        subset: Subset of data (`"training"` or `"validation"`) if
          validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image. Supported
          methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3
          or newer is installed, "lanczos" is also supported. If PIL version
          3.4.0 or newer is installed, "box" and "hamming" are also supported.
          By default, "nearest" is used.
        keep_aspect_ratio: Boolean, whether to resize images to a target size
            without aspect ratio distortion. The image is cropped in the center
            with target aspect ratio before resizing.
        dtype: Dtype to use for generated arrays.
    """

    allowed_class_modes = {"categorical", "binary", "sparse", "input", None}

    def __init__(
        self,
        directory,
        image_data_generator,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        data_format=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
        keep_aspect_ratio=False,
        dtype=None,
    ):
        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()
        super().set_processing_attrs(
            image_data_generator,
            target_size,
            color_mode,
            data_format,
            save_to_dir,
            save_prefix,
            save_format,
            subset,
            interpolation,
            keep_aspect_ratio,
        )
        self.directory = directory
        self.classes = classes
        if class_mode not in self.allowed_class_modes:
            raise ValueError(
                "Invalid class_mode: {}; expected one of: {}".format(
                    class_mode, self.allowed_class_modes
                )
            )
        self.class_mode = class_mode
        self.dtype = dtype
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(
                    _list_valid_filenames_in_directory,
                    (
                        dirpath,
                        self.white_list_formats,
                        self.split,
                        self.class_indices,
                        follow_links,
                    ),
                )
            )
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype="int32")
        for classes in classes_list:
            self.classes[i : i + len(classes)] = classes
            i += len(classes)

        io_utils.print_msg(
            f"Found {self.samples} images belonging to "
            f"{self.num_classes} classes."
        )
        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        super().__init__(self.samples, batch_size, shuffle, seed)

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None


@keras_export("keras.preprocessing.image.NumpyArrayIterator")
class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    Deprecated: `tf.keras.preprocessing.image.NumpyArrayIterator` is not
    recommended for new code. Prefer loading images with
    `tf.keras.utils.image_dataset_from_directory` and transforming the output
    `tf.data.Dataset` with preprocessing layers. For more information, see the
    tutorials for [loading images](
    https://www.tensorflow.org/tutorials/load_data/images) and
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        x: Numpy array of input data or tuple. If tuple, the second elements is
          either another numpy array or a list of numpy arrays, each of which
          gets passed through as an output without any modifications.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator` to use for random
          transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        sample_weight: Numpy array of sample weights.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures being
          yielded, in a viewable format. This is useful for visualizing the
          random transformations being applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample images (if
          `save_to_dir` is set).
        save_format: Format to use for saving sample images (if `save_to_dir` is
          set).
        subset: Subset of data (`"training"` or `"validation"`) if
          validation_split is set in ImageDataGenerator.
        ignore_class_split: Boolean (default: False), ignore difference
          in number of classes in labels across train and validation
          split (useful for non-classification tasks)
        dtype: Dtype to use for the generated arrays.
    """

    def __init__(
        self,
        x,
        y,
        image_data_generator,
        batch_size=32,
        shuffle=False,
        sample_weight=None,
        seed=None,
        data_format=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
        ignore_class_split=False,
        dtype=None,
    ):
        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()
        self.dtype = dtype
        if isinstance(x, tuple) or isinstance(x, list):
            if not isinstance(x[1], list):
                x_misc = [np.asarray(x[1])]
            else:
                x_misc = [np.asarray(xx) for xx in x[1]]
            x = x[0]
            for xx in x_misc:
                if len(x) != len(xx):
                    raise ValueError(
                        "All of the arrays in `x` "
                        "should have the same length. "
                        "Found a pair with: len(x[0]) = %s, len(x[?]) = %s"
                        % (len(x), len(xx))
                    )
        else:
            x_misc = []

        if y is not None and len(x) != len(y):
            raise ValueError(
                "`x` (images tensor) and `y` (labels) "
                "should have the same length. "
                "Found: x.shape = %s, y.shape = %s"
                % (np.asarray(x).shape, np.asarray(y).shape)
            )
        if sample_weight is not None and len(x) != len(sample_weight):
            raise ValueError(
                "`x` (images tensor) and `sample_weight` "
                "should have the same length. "
                "Found: x.shape = %s, sample_weight.shape = %s"
                % (np.asarray(x).shape, np.asarray(sample_weight).shape)
            )
        if subset is not None:
            if subset not in {"training", "validation"}:
                raise ValueError(
                    "Invalid subset name:",
                    subset,
                    '; expected "training" or "validation".',
                )
            split_idx = int(len(x) * image_data_generator._validation_split)

            if (
                y is not None
                and not ignore_class_split
                and not np.array_equal(
                    np.unique(y[:split_idx]), np.unique(y[split_idx:])
                )
            ):
                raise ValueError(
                    "Training and validation subsets "
                    "have different number of classes after "
                    "the split. If your numpy arrays are "
                    "sorted by the label, you might want "
                    "to shuffle them."
                )

            if subset == "validation":
                x = x[:split_idx]
                x_misc = [np.asarray(xx[:split_idx]) for xx in x_misc]
                if y is not None:
                    y = y[:split_idx]
            else:
                x = x[split_idx:]
                x_misc = [np.asarray(xx[split_idx:]) for xx in x_misc]
                if y is not None:
                    y = y[split_idx:]

        self.x = np.asarray(x, dtype=self.dtype)
        self.x_misc = x_misc
        if self.x.ndim != 4:
            raise ValueError(
                "Input data in `NumpyArrayIterator` "
                "should have rank 4. You passed an array "
                "with shape",
                self.x.shape,
            )
        channels_axis = 3 if data_format == "channels_last" else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            warnings.warn(
                "NumpyArrayIterator is set to use the "
                'data format convention "' + data_format + '" '
                "(channels on axis "
                + str(channels_axis)
                + "), i.e. expected either 1, 3, or 4 "
                "channels on axis " + str(channels_axis) + ". "
                "However, it was passed an array with shape "
                + str(self.x.shape)
                + " ("
                + str(self.x.shape[channels_axis])
                + " channels)."
            )
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        if sample_weight is not None:
            self.sample_weight = np.asarray(sample_weight)
        else:
            self.sample_weight = None
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super().__init__(x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            tuple([len(index_array)] + list(self.x.shape)[1:]), dtype=self.dtype
        )
        for i, j in enumerate(index_array):
            x = self.x[j]
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(
                x.astype(self.dtype), params
            )
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = image_utils.array_to_img(
                    batch_x[i], self.data_format, scale=True
                )
                fname = "{prefix}_{index}_{hash}.{format}".format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format,
                )
                img.save(os.path.join(self.save_to_dir, fname))
        batch_x_miscs = [xx[index_array] for xx in self.x_misc]
        output = (batch_x if not batch_x_miscs else [batch_x] + batch_x_miscs,)
        if self.y is None:
            return output[0]
        output += (self.y[index_array],)
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)
        return output


def validate_filename(filename, white_list_formats):
    """Check if a filename refers to a valid file.

    Args:
        filename: String, absolute path to a file
        white_list_formats: Set, allowed file extensions
    Returns:
        A boolean value indicating if the filename is valid or not
    """
    return filename.lower().endswith(white_list_formats) and os.path.isfile(
        filename
    )


class DataFrameIterator(BatchFromFilesMixin, Iterator):
    """Iterator capable of reading images from a directory on disk as a dataframe.

    Args:
        dataframe: Pandas dataframe containing the filepaths relative to
          `directory` (or absolute paths if `directory` is None) of the images
          in a string column. It should include other column/s depending on the
          `class_mode`: - if `class_mode` is `"categorical"` (default value) it
          must include the `y_col` column with the class/es of each image.
          Values in column can be string/list/tuple if a single class or
          list/tuple if multiple classes.
            - if `class_mode` is `"binary"` or `"sparse"` it must include the
              given `y_col` column with class values as strings.
            - if `class_mode` is `"raw"` or `"multi_output"` it should contain
              the columns specified in `y_col`.
            - if `class_mode` is `"input"` or `None` no extra column is needed.
        directory: string, path to the directory to read images from. If `None`,
          data in `x_col` column should be absolute paths.
        image_data_generator: Instance of `ImageDataGenerator` to use for random
          transformations and normalization. If None, no transformations and
          normalizations are made.
        x_col: string, column in `dataframe` that contains the filenames (or
          absolute paths if `directory` is `None`).
        y_col: string or list, column/s in `dataframe` that has the target data.
        weight_col: string, column in `dataframe` that contains the sample
            weights. Default: `None`.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`. Color mode to read
          images.
        classes: Optional list of strings, classes to use (e.g. `["dogs",
          "cats"]`). If None, all classes in `y_col` will be used.
        class_mode: one of "binary", "categorical", "input", "multi_output",
          "raw", "sparse" or None. Default: "categorical".
          Mode for yielding the targets:
            - `"binary"`: 1D numpy array of binary labels,
            - `"categorical"`: 2D numpy array of one-hot encoded labels.
              Supports multi-label output.
            - `"input"`: images identical to input images (mainly used to work
              with autoencoders),
            - `"multi_output"`: list with the values of the different columns,
            - `"raw"`: numpy array of values in `y_col` column(s),
            - `"sparse"`: 1D numpy array of integer labels, - `None`, no targets
              are returned (the generator will only yield batches of image data,
              which is useful to use in `model.predict()`).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures being
          yielded, in a viewable format. This is useful for visualizing the
          random transformations being applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample images (if
          `save_to_dir` is set).
        save_format: Format to use for saving sample images (if `save_to_dir` is
          set).
        subset: Subset of data (`"training"` or `"validation"`) if
          validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
          target size is different from that of the loaded image. Supported
          methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3
          or newer is installed, "lanczos" is also supported. If PIL version
          3.4.0 or newer is installed, "box" and "hamming" are also supported.
          By default, "nearest" is used.
        keep_aspect_ratio: Boolean, whether to resize images to a target size
          without aspect ratio distortion. The image is cropped in the center
          with target aspect ratio before resizing.
        dtype: Dtype to use for the generated arrays.
        validate_filenames: Boolean, whether to validate image filenames in
          `x_col`. If `True`, invalid images will be ignored. Disabling this
          option can lead to speed-up in the instantiation of this class.
          Default: `True`.
    """

    allowed_class_modes = {
        "binary",
        "categorical",
        "input",
        "multi_output",
        "raw",
        "sparse",
        None,
    }

    def __init__(
        self,
        dataframe,
        directory=None,
        image_data_generator=None,
        x_col="filename",
        y_col="class",
        weight_col=None,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        data_format="channels_last",
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
        interpolation="nearest",
        keep_aspect_ratio=False,
        dtype="float32",
        validate_filenames=True,
    ):
        super().set_processing_attrs(
            image_data_generator,
            target_size,
            color_mode,
            data_format,
            save_to_dir,
            save_prefix,
            save_format,
            subset,
            interpolation,
            keep_aspect_ratio,
        )
        df = dataframe.copy()
        self.directory = directory or ""
        self.class_mode = class_mode
        self.dtype = dtype
        # check that inputs match the required class_mode
        self._check_params(df, x_col, y_col, weight_col, classes)
        if (
            validate_filenames
        ):  # check which image files are valid and keep them
            df = self._filter_valid_filepaths(df, x_col)
        if class_mode not in ["input", "multi_output", "raw", None]:
            df, classes = self._filter_classes(df, y_col, classes)
            num_classes = len(classes)
            # build an index of all the unique classes
            self.class_indices = dict(zip(classes, range(len(classes))))
        # retrieve only training or validation set
        if self.split:
            num_files = len(df)
            start = int(self.split[0] * num_files)
            stop = int(self.split[1] * num_files)
            df = df.iloc[start:stop, :]
        # get labels for each observation
        if class_mode not in ["input", "multi_output", "raw", None]:
            self.classes = self.get_classes(df, y_col)
        self.filenames = df[x_col].tolist()
        self._sample_weight = df[weight_col].values if weight_col else None

        if class_mode == "multi_output":
            self._targets = [np.array(df[col].tolist()) for col in y_col]
        if class_mode == "raw":
            self._targets = df[y_col].values
        self.samples = len(self.filenames)
        validated_string = (
            "validated" if validate_filenames else "non-validated"
        )
        if class_mode in ["input", "multi_output", "raw", None]:
            io_utils.print_msg(
                f"Found {self.samples} {validated_string} image filenames."
            )
        else:
            io_utils.print_msg(
                f"Found {self.samples} {validated_string} image filenames "
                f"belonging to {num_classes} classes."
            )
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        super().__init__(self.samples, batch_size, shuffle, seed)

    def _check_params(self, df, x_col, y_col, weight_col, classes):
        # check class mode is one of the currently supported
        if self.class_mode not in self.allowed_class_modes:
            raise ValueError(
                "Invalid class_mode: {}; expected one of: {}".format(
                    self.class_mode, self.allowed_class_modes
                )
            )
        # check that y_col has several column names if class_mode is
        # multi_output
        if (self.class_mode == "multi_output") and not isinstance(y_col, list):
            raise TypeError(
                'If class_mode="{}", y_col must be a list. Received {}.'.format(
                    self.class_mode, type(y_col).__name__
                )
            )
        # check that filenames/filepaths column values are all strings
        if not all(df[x_col].apply(lambda x: isinstance(x, str))):
            raise TypeError(
                "All values in column x_col={} must be strings.".format(x_col)
            )
        # check labels are string if class_mode is binary or sparse
        if self.class_mode in {"binary", "sparse"}:
            if not all(df[y_col].apply(lambda x: isinstance(x, str))):
                raise TypeError(
                    'If class_mode="{}", y_col="{}" column '
                    "values must be strings.".format(self.class_mode, y_col)
                )
        # check that if binary there are only 2 different classes
        if self.class_mode == "binary":
            if classes:
                classes = set(classes)
                if len(classes) != 2:
                    raise ValueError(
                        'If class_mode="binary" there must be 2 '
                        "classes. {} class/es were given.".format(len(classes))
                    )
            elif df[y_col].nunique() != 2:
                raise ValueError(
                    'If class_mode="binary" there must be 2 classes. '
                    "Found {} classes.".format(df[y_col].nunique())
                )
        # check values are string, list or tuple if class_mode is categorical
        if self.class_mode == "categorical":
            types = (str, list, tuple)
            if not all(df[y_col].apply(lambda x: isinstance(x, types))):
                raise TypeError(
                    'If class_mode="{}", y_col="{}" column '
                    "values must be type string, list or tuple.".format(
                        self.class_mode, y_col
                    )
                )
        # raise warning if classes are given but will be unused
        if classes and self.class_mode in {
            "input",
            "multi_output",
            "raw",
            None,
        }:
            warnings.warn(
                '`classes` will be ignored given the class_mode="{}"'.format(
                    self.class_mode
                )
            )
        # check that if weight column that the values are numerical
        if weight_col and not issubclass(df[weight_col].dtype.type, np.number):
            raise TypeError(
                "Column weight_col={} must be numeric.".format(weight_col)
            )

    def get_classes(self, df, y_col):
        labels = []
        for label in df[y_col]:
            if isinstance(label, (list, tuple)):
                labels.append([self.class_indices[lbl] for lbl in label])
            else:
                labels.append(self.class_indices[label])
        return labels

    @staticmethod
    def _filter_classes(df, y_col, classes):
        df = df.copy()

        def remove_classes(labels, classes):
            if isinstance(labels, (list, tuple)):
                labels = [cls for cls in labels if cls in classes]
                return labels or None
            elif isinstance(labels, str):
                return labels if labels in classes else None
            else:
                raise TypeError(
                    "Expect string, list or tuple "
                    "but found {} in {} column ".format(type(labels), y_col)
                )

        if classes:
            # prepare for membership lookup
            classes = list(collections.OrderedDict.fromkeys(classes).keys())
            df[y_col] = df[y_col].apply(lambda x: remove_classes(x, classes))
        else:
            classes = set()
            for v in df[y_col]:
                if isinstance(v, (list, tuple)):
                    classes.update(v)
                else:
                    classes.add(v)
            classes = sorted(classes)
        return df.dropna(subset=[y_col]), classes

    def _filter_valid_filepaths(self, df, x_col):
        """Keep only dataframe rows with valid filenames.

        Args:
            df: Pandas dataframe containing filenames in a column
            x_col: string, column in `df` that contains the filenames or
                filepaths
        Returns:
            absolute paths to image files
        """
        filepaths = df[x_col].map(
            lambda fname: os.path.join(self.directory, fname)
        )
        mask = filepaths.apply(
            validate_filename, args=(self.white_list_formats,)
        )
        n_invalid = (~mask).sum()
        if n_invalid:
            warnings.warn(
                'Found {} invalid image filename(s) in x_col="{}". '
                "These filename(s) will be ignored.".format(n_invalid, x_col)
            )
        return df[mask]

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        if self.class_mode in {"multi_output", "raw"}:
            return self._targets
        else:
            return self.classes

    @property
    def sample_weight(self):
        return self._sample_weight


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


@keras_export("keras.preprocessing.image.ImageDataGenerator")
class ImageDataGenerator:
    """Generate batches of tensor image data with real-time data augmentation.

    Deprecated: `tf.keras.preprocessing.image.ImageDataGenerator` is not
    recommended for new code. Prefer loading images with
    `tf.keras.utils.image_dataset_from_directory` and transforming the output
    `tf.data.Dataset` with preprocessing layers. For more information, see the
    tutorials for [loading images](
    https://www.tensorflow.org/tutorials/load_data/images) and
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).

     The data will be looped over (in batches).

    Args:
        featurewise_center: Boolean. Set input mean to 0 over the dataset,
          feature-wise.
        samplewise_center: Boolean. Set each sample mean to 0.
        featurewise_std_normalization: Boolean. Divide inputs by std of the
          dataset, feature-wise.
        samplewise_std_normalization: Boolean. Divide each input by its std.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening: Boolean. Apply ZCA whitening.
        rotation_range: Int. Degree range for random rotations.
        width_shift_range: Float, 1-D array-like or int
            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval `(-width_shift_range,
              +width_shift_range)` - With `width_shift_range=2` possible values
              are integers `[-1, 0, +1]`, same as with `width_shift_range=[-1,
              0, +1]`, while with `width_shift_range=1.0` possible values are
              floats in the interval [-1.0, +1.0).
        height_shift_range: Float, 1-D array-like or int
            - float: fraction of total height, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval `(-height_shift_range,
              +height_shift_range)` - With `height_shift_range=2` possible
              values are integers `[-1, 0, +1]`, same as with
              `height_shift_range=[-1, 0, +1]`, while with
              `height_shift_range=1.0` possible values are floats in the
              interval [-1.0, +1.0).
        brightness_range: Tuple or list of two floats. Range for picking a
          brightness shift value from.
        shear_range: Float. Shear Intensity (Shear angle in counter-clockwise
          direction in degrees)
        zoom_range: Float or [lower, upper]. Range for random zoom. If a float,
          `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        channel_shift_range: Float. Range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}. Default
          is 'nearest'. Points outside the boundaries of the input are filled
          according to the given mode:
            - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
            - 'nearest':  aaaaaaaa|abcd|dddddddd
            - 'reflect':  abcddcba|abcd|dcbaabcd
            - 'wrap':  abcdabcd|abcd|abcdabcd
        cval: Float or Int. Value used for points outside the boundaries when
          `fill_mode = "constant"`.
        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        vertical_flip: Boolean. Randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling
          is applied, otherwise we multiply the data by the value provided
          (after applying all other transformations).
        preprocessing_function: function that will be applied on each input. The
          function will run after the image is resized and augmented.
            The function should take one argument: one image (Numpy tensor with
              rank 3), and should output a Numpy tensor with the same shape.
        data_format: Image data format, either "channels_first" or
          "channels_last". "channels_last" mode means that the images should
          have shape `(samples, height, width, channels)`, "channels_first" mode
          means that the images should have shape `(samples, channels, height,
          width)`.  It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`. If you never set it, then
          it will be "channels_last".
        validation_split: Float. Fraction of images reserved for validation
          (strictly between 0 and 1).
        dtype: Dtype to use for the generated arrays.

    Raises:
      ValueError: If the value of the argument, `data_format` is other than
            `"channels_last"` or `"channels_first"`.
      ValueError: If the value of the argument, `validation_split` > 1
            or `validation_split` < 0.

    Examples:

    Example of using `.flow(x, y)`:

    ```python
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)
    # fits the model on batches with real-time data augmentation:
    model.fit(datagen.flow(x_train, y_train, batch_size=32,
             subset='training'),
             validation_data=datagen.flow(x_train, y_train,
             batch_size=8, subset='validation'),
             steps_per_epoch=len(x_train) / 32, epochs=epochs)
    # here's a more "manual" example
    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
            model.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
    ```

    Example of using `.flow_from_directory(directory)`:

    ```python
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')
    model.fit(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)
    ```

    Example of transforming images and masks together.

    ```python
    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)
    image_generator = image_datagen.flow_from_directory(
        'data/images',
        class_mode=None,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        'data/masks',
        class_mode=None,
        seed=seed)
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    model.fit(
        train_generator,
        steps_per_epoch=2000,
        epochs=50)
    ```
    """

    def __init__(
        self,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        interpolation_order=1,
        dtype=None,
    ):
        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()

        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype
        self.interpolation_order = interpolation_order

        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(
                '`data_format` should be `"channels_last"` '
                "(channel after row and column) or "
                '`"channels_first"` (channel before row and column). '
                "Received: %s" % data_format
            )
        self.data_format = data_format
        if data_format == "channels_first":
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == "channels_last":
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2
        if validation_split and not 0 < validation_split < 1:
            raise ValueError(
                "`validation_split` must be strictly between 0 and 1. "
                " Received: %s" % validation_split
            )
        self._validation_split = validation_split

        self.mean = None
        self.std = None
        self.zca_whitening_matrix = None

        if isinstance(zoom_range, (float, int)):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2 and all(
            isinstance(val, (float, int)) for val in zoom_range
        ):
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError(
                "`zoom_range` should be a float or "
                "a tuple or list of two floats. "
                "Received: %s" % (zoom_range,)
            )
        if zca_whitening:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`zca_whitening`, which overrides "
                    "setting of `featurewise_center`."
                )
            if featurewise_std_normalization:
                self.featurewise_std_normalization = False
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`zca_whitening` "
                    "which overrides setting of"
                    "`featurewise_std_normalization`."
                )
        if featurewise_std_normalization:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`featurewise_std_normalization`, "
                    "which overrides setting of "
                    "`featurewise_center`."
                )
        if samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`samplewise_std_normalization`, "
                    "which overrides setting of "
                    "`samplewise_center`."
                )
        if brightness_range is not None:
            if (
                not isinstance(brightness_range, (tuple, list))
                or len(brightness_range) != 2
            ):
                raise ValueError(
                    "`brightness_range should be tuple or list of two floats. "
                    "Received: %s" % (brightness_range,)
                )
        self.brightness_range = brightness_range

    def flow(
        self,
        x,
        y=None,
        batch_size=32,
        shuffle=True,
        sample_weight=None,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        ignore_class_split=False,
        subset=None,
    ):
        """Takes data & label arrays, generates batches of augmented data.

        Args:
            x: Input data. Numpy array of rank 4 or a tuple. If tuple, the first
              element should contain the images and the second element another
              numpy array or a list of numpy arrays that gets passed to the
              output without any modifications. Can be used to feed the model
              miscellaneous data along with the images. In case of grayscale
              data, the channels axis of the image array should have value 1, in
              case of RGB data, it should have value 3, and in case of RGBA
              data, it should have value 4.
            y: Labels.
            batch_size: Int (default: 32).
            shuffle: Boolean (default: True).
            sample_weight: Sample weights.
            seed: Int (default: None).
            save_to_dir: None or str (default: None). This allows you to
              optionally specify a directory to which to save the augmented
              pictures being generated (useful for visualizing what you are
              doing).
            save_prefix: Str (default: `''`). Prefix to use for filenames of
              saved pictures (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg", "bmp", "pdf", "ppm", "gif",
              "tif", "jpg" (only relevant if `save_to_dir` is set). Default:
              "png".
            ignore_class_split: Boolean (default: False), ignore difference
              in number of classes in labels across train and validation
              split (useful for non-classification tasks)
            subset: Subset of data (`"training"` or `"validation"`) if
              `validation_split` is set in `ImageDataGenerator`.

        Returns:
            An `Iterator` yielding tuples of `(x, y)`
                where `x` is a numpy array of image data
                (in the case of a single image input) or a list
                of numpy arrays (in the case with
                additional inputs) and `y` is a numpy array
                of corresponding labels. If 'sample_weight' is not None,
                the yielded tuples are of the form `(x, y, sample_weight)`.
                If `y` is None, only the numpy array `x` is returned.
        Raises:
          ValueError: If the Value of the argument, `subset` is other than
                "training" or "validation".

        """
        return NumpyArrayIterator(
            x,
            y,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            ignore_class_split=ignore_class_split,
            subset=subset,
            dtype=self.dtype,
        )

    def flow_from_directory(
        self,
        directory,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset=None,
        interpolation="nearest",
        keep_aspect_ratio=False,
    ):
        """Takes the path to a directory & generates batches of augmented data.

        Args:
            directory: string, path to the target directory. It should contain
              one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images
              inside each of the subdirectories directory tree will be included
              in the generator. See [this script](
              https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
              for more details.
            target_size: Tuple of integers `(height, width)`, defaults to `(256,
              256)`. The dimensions to which all images found will be resized.
            color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
              Whether the images will be converted to have 1, 3, or 4 channels.
            classes: Optional list of class subdirectories (e.g. `['dogs',
              'cats']`). Default: None. If not provided, the list of classes
              will be automatically inferred from the subdirectory
              names/structure under `directory`, where each subdirectory will be
              treated as a different class (and the order of the classes, which
              will map to the label indices, will be alphanumeric). The
              dictionary containing the mapping from class names to class
              indices can be obtained via the attribute `class_indices`.
            class_mode: One of "categorical", "binary", "sparse",
                "input", or None. Default: "categorical".
                Determines the type of label arrays that are returned:
                - "categorical" will be 2D one-hot encoded labels,
                - "binary" will be 1D binary labels,
                    "sparse" will be 1D integer labels,
                - "input" will be images identical
                    to input images (mainly used to work with autoencoders).
                - If None, no labels are returned
                  (the generator will only yield batches of image data,
                  which is useful to use with `model.predict_generator()`).
                  Please note that in case of class_mode None,
                  the data still needs to reside in a subdirectory
                  of `directory` for it to work correctly.
            batch_size: Size of the batches of data (default: 32).
            shuffle: Whether to shuffle the data (default: True) If set to
              False, sorts the data in alphanumeric order.
            seed: Optional random seed for shuffling and transformations.
            save_to_dir: None or str (default: None). This allows you to
              optionally specify a directory to which to save the augmented
              pictures being generated (useful for visualizing what you are
              doing).
            save_prefix: Str. Prefix to use for filenames of saved pictures
              (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg", "bmp", "pdf", "ppm", "gif",
              "tif", "jpg" (only relevant if `save_to_dir` is set). Default:
              "png".
            follow_links: Whether to follow symlinks inside
                class subdirectories (default: False).
            subset: Subset of data (`"training"` or `"validation"`) if
              `validation_split` is set in `ImageDataGenerator`.
            interpolation: Interpolation method used to resample the image if
              the target size is different from that of the loaded image.
              Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
              If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
              supported. If PIL version 3.4.0 or newer is installed, `"box"` and
              `"hamming"` are also supported. By default, `"nearest"` is used.
            keep_aspect_ratio: Boolean, whether to resize images to a target
              size without aspect ratio distortion. The image is cropped in
              the center with target aspect ratio before resizing.

        Returns:
            A `DirectoryIterator` yielding tuples of `(x, y)`
                where `x` is a numpy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a numpy array of corresponding labels.
        """
        return DirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            keep_aspect_ratio=keep_aspect_ratio,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            dtype=self.dtype,
        )

    def flow_from_dataframe(
        self,
        dataframe,
        directory=None,
        x_col="filename",
        y_col="class",
        weight_col=None,
        target_size=(256, 256),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir=None,
        save_prefix="",
        save_format="png",
        subset=None,
        interpolation="nearest",
        validate_filenames=True,
        **kwargs,
    ):
        """Takes the dataframe and the path to a directory + generates batches.

         The generated batches contain augmented/normalized data.

        **A simple tutorial can be found **[here](
                                    http://bit.ly/keras_flow_from_dataframe).

        Args:
            dataframe: Pandas dataframe containing the filepaths relative to
                `directory` (or absolute paths if `directory` is None) of the
                images in a string column. It should include other column/s
                depending on the `class_mode`:
                - if `class_mode` is `"categorical"` (default value) it must
                    include the `y_col` column with the class/es of each image.
                    Values in column can be string/list/tuple if a single class
                    or list/tuple if multiple classes.
                - if `class_mode` is `"binary"` or `"sparse"` it must include
                    the given `y_col` column with class values as strings.
                - if `class_mode` is `"raw"` or `"multi_output"` it should
                    contain the columns specified in `y_col`.
                - if `class_mode` is `"input"` or `None` no extra column is
                    needed.
            directory: string, path to the directory to read images from. If
              `None`, data in `x_col` column should be absolute paths.
            x_col: string, column in `dataframe` that contains the filenames (or
              absolute paths if `directory` is `None`).
            y_col: string or list, column/s in `dataframe` that has the target
              data.
            weight_col: string, column in `dataframe` that contains the sample
                weights. Default: `None`.
            target_size: tuple of integers `(height, width)`, default: `(256,
              256)`. The dimensions to which all images found will be resized.
            color_mode: one of "grayscale", "rgb", "rgba". Default: "rgb".
              Whether the images will be converted to have 1 or 3 color
              channels.
            classes: optional list of classes (e.g. `['dogs', 'cats']`). Default
              is None. If not provided, the list of classes will be
              automatically inferred from the `y_col`, which will map to the
              label indices, will be alphanumeric). The dictionary containing
              the mapping from class names to class indices can be obtained via
              the attribute `class_indices`.
            class_mode: one of "binary", "categorical", "input", "multi_output",
                "raw", sparse" or None. Default: "categorical".
                Mode for yielding the targets:
                - `"binary"`: 1D numpy array of binary labels,
                - `"categorical"`: 2D numpy array of one-hot encoded labels.
                  Supports multi-label output.
                - `"input"`: images identical to input images (mainly used to
                  work with autoencoders),
                - `"multi_output"`: list with the values of the different
                  columns,
                - `"raw"`: numpy array of values in `y_col` column(s),
                - `"sparse"`: 1D numpy array of integer labels,
                - `None`, no targets are returned (the generator will only yield
                  batches of image data, which is useful to use in
                  `model.predict()`).
            batch_size: size of the batches of data (default: 32).
            shuffle: whether to shuffle the data (default: True)
            seed: optional random seed for shuffling and transformations.
            save_to_dir: None or str (default: None). This allows you to
              optionally specify a directory to which to save the augmented
              pictures being generated (useful for visualizing what you are
              doing).
            save_prefix: str. Prefix to use for filenames of saved pictures
              (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg", "bmp", "pdf", "ppm", "gif",
              "tif", "jpg" (only relevant if `save_to_dir` is set). Default:
              "png".
            subset: Subset of data (`"training"` or `"validation"`) if
              `validation_split` is set in `ImageDataGenerator`.
            interpolation: Interpolation method used to resample the image if
              the target size is different from that of the loaded image.
              Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
              If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
              supported. If PIL version 3.4.0 or newer is installed, `"box"` and
              `"hamming"` are also supported. By default, `"nearest"` is used.
            validate_filenames: Boolean, whether to validate image filenames in
              `x_col`. If `True`, invalid images will be ignored. Disabling this
              option can lead to speed-up in the execution of this function.
              Defaults to `True`.
            **kwargs: legacy arguments for raising deprecation warnings.

        Returns:
            A `DataFrameIterator` yielding tuples of `(x, y)`
            where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.
        """
        if "has_ext" in kwargs:
            warnings.warn(
                "has_ext is deprecated, filenames in the dataframe have "
                "to match the exact filenames in disk.",
                DeprecationWarning,
            )
        if "sort" in kwargs:
            warnings.warn(
                "sort is deprecated, batches will be created in the"
                "same order than the filenames provided if shuffle"
                "is set to False.",
                DeprecationWarning,
            )
        if class_mode == "other":
            warnings.warn(
                '`class_mode` "other" is deprecated, please use '
                '`class_mode` "raw".',
                DeprecationWarning,
            )
            class_mode = "raw"
        if "drop_duplicates" in kwargs:
            warnings.warn(
                "drop_duplicates is deprecated, you can drop duplicates "
                "by using the pandas.DataFrame.drop_duplicates method.",
                DeprecationWarning,
            )

        return DataFrameIterator(
            dataframe,
            directory,
            self,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            validate_filenames=validate_filenames,
            dtype=self.dtype,
        )

    def standardize(self, x):
        """Applies the normalization configuration in-place to a batch of
        inputs.

        `x` is changed in-place since the function is mainly used internally
        to standardize images and feed them to your network. If a copy of `x`
        would be created instead it would have a significant performance cost.
        If you want to apply this method without changing the input in-place
        you can call the method creating a copy before:

        standardize(np.copy(x))

        Args:
            x: Batch of inputs to be normalized.

        Returns:
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= np.std(x, keepdims=True) + 1e-6

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`featurewise_center`, but it hasn't "
                    "been fit on any training data. Fit it "
                    "first by calling `.fit(numpy_data)`."
                )
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= self.std + 1e-6
            else:
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`featurewise_std_normalization`, "
                    "but it hasn't "
                    "been fit on any training data. Fit it "
                    "first by calling `.fit(numpy_data)`."
                )
        if self.zca_whitening:
            if self.zca_whitening_matrix is not None:
                flat_x = x.reshape(-1, np.prod(x.shape[-3:]))
                white_x = flat_x @ self.zca_whitening_matrix
                x = np.reshape(white_x, x.shape)
            else:
                warnings.warn(
                    "This ImageDataGenerator specifies "
                    "`zca_whitening`, but it hasn't "
                    "been fit on any training data. Fit it "
                    "first by calling `.fit(numpy_data)`."
                )
        return x

    def get_random_transform(self, img_shape, seed=None):
        """Generates random parameters for a transformation.

        Args:
            img_shape: Tuple of integers.
                Shape of the image that is transformed.
            seed: Random seed.

        Returns:
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range:
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            try:  # 1-D array-like or int
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(
                    -self.height_shift_range, self.height_shift_range
                )
            if np.max(self.height_shift_range) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(
                    -self.width_shift_range, self.width_shift_range
                )
            if np.max(self.width_shift_range) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 2
            )

        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        channel_shift_intensity = None
        if self.channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(
                -self.channel_shift_range, self.channel_shift_range
            )

        brightness = None
        if self.brightness_range is not None:
            brightness = np.random.uniform(
                self.brightness_range[0], self.brightness_range[1]
            )

        transform_parameters = {
            "theta": theta,
            "tx": tx,
            "ty": ty,
            "shear": shear,
            "zx": zx,
            "zy": zy,
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
            "channel_shift_intensity": channel_shift_intensity,
            "brightness": brightness,
        }

        return transform_parameters

    def apply_transform(self, x, transform_parameters):
        """Applies a transformation to an image according to given parameters.

        Args:
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intensity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.

        Returns:
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        x = apply_affine_transform(
            x,
            transform_parameters.get("theta", 0),
            transform_parameters.get("tx", 0),
            transform_parameters.get("ty", 0),
            transform_parameters.get("shear", 0),
            transform_parameters.get("zx", 1),
            transform_parameters.get("zy", 1),
            row_axis=img_row_axis,
            col_axis=img_col_axis,
            channel_axis=img_channel_axis,
            fill_mode=self.fill_mode,
            cval=self.cval,
            order=self.interpolation_order,
        )

        if transform_parameters.get("channel_shift_intensity") is not None:
            x = apply_channel_shift(
                x,
                transform_parameters["channel_shift_intensity"],
                img_channel_axis,
            )

        if transform_parameters.get("flip_horizontal", False):
            x = flip_axis(x, img_col_axis)

        if transform_parameters.get("flip_vertical", False):
            x = flip_axis(x, img_row_axis)

        if transform_parameters.get("brightness") is not None:
            x = apply_brightness_shift(
                x, transform_parameters["brightness"], False
            )

        return x

    def random_transform(self, x, seed=None):
        """Applies a random transformation to an image.

        Args:
            x: 3D tensor, single image.
            seed: Random seed.

        Returns:
            A randomly transformed version of the input (same shape).
        """
        params = self.get_random_transform(x.shape, seed)
        return self.apply_transform(x, params)

    def fit(self, x, augment=False, rounds=1, seed=None):
        """Fits the data generator to some sample data.

        This computes the internal data stats related to the
        data-dependent transformations, based on an array of sample data.

        Only required if `featurewise_center` or
        `featurewise_std_normalization` or `zca_whitening` are set to True.

        When `rescale` is set to a value, rescaling is applied to
        sample data before computing the internal data stats.

        Args:
            x: Sample data. Should have rank 4.
             In case of grayscale data,
             the channels axis should have value 1, in case
             of RGB data, it should have value 3, and in case
             of RGBA data, it should have value 4.
            augment: Boolean (default: False).
                Whether to fit on randomly augmented samples.
            rounds: Int (default: 1).
                If using data augmentation (`augment=True`),
                this is how many augmentation passes over the data to use.
            seed: Int (default: None). Random seed.
        """
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 4:
            raise ValueError(
                "Input to `.fit()` should have rank 4. "
                "Got array with shape: " + str(x.shape)
            )
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                "Expected input to be images (as Numpy array) "
                'following the data format convention "'
                + self.data_format
                + '" (channels on axis '
                + str(self.channel_axis)
                + "), i.e. expected "
                "either 1, 3 or 4 channels on axis "
                + str(self.channel_axis)
                + ". "
                "However, it was passed an array with shape "
                + str(x.shape)
                + " ("
                + str(x.shape[self.channel_axis])
                + " channels)."
            )

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if self.rescale:
            x *= self.rescale

        if augment:
            ax = np.zeros(
                tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                dtype=self.dtype,
            )
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= self.std + 1e-6

        if self.zca_whitening:
            n = len(x)
            flat_x = np.reshape(x, (n, -1))

            u, s, _ = np.linalg.svd(flat_x.T, full_matrices=False)
            s_inv = np.sqrt(n) / (s + self.zca_epsilon)
            self.zca_whitening_matrix = (u * s_inv).dot(u.T)


@keras_export("keras.preprocessing.image.random_rotation")
def random_rotation(
    x,
    rg,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode="nearest",
    cval=0.0,
    interpolation_order=1,
):
    """Performs a random rotation of a Numpy image tensor.

    Deprecated: `tf.keras.preprocessing.image.random_rotation` does not operate
    on tensors and is not recommended for new code. Prefer
    `tf.keras.layers.RandomRotation` which provides equivalent functionality as
    a preprocessing layer. For more information, see the tutorial for
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    Returns:
        Rotated Numpy image tensor.
    """
    theta = np.random.uniform(-rg, rg)
    x = apply_affine_transform(
        x,
        theta=theta,
        row_axis=row_axis,
        col_axis=col_axis,
        channel_axis=channel_axis,
        fill_mode=fill_mode,
        cval=cval,
        order=interpolation_order,
    )
    return x


@keras_export("keras.preprocessing.image.random_shift")
def random_shift(
    x,
    wrg,
    hrg,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode="nearest",
    cval=0.0,
    interpolation_order=1,
):
    """Performs a random spatial shift of a Numpy image tensor.

    Deprecated: `tf.keras.preprocessing.image.random_shift` does not operate on
    tensors and is not recommended for new code. Prefer
    `tf.keras.layers.RandomTranslation` which provides equivalent functionality
    as a preprocessing layer. For more information, see the tutorial for
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    Returns:
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    x = apply_affine_transform(
        x,
        tx=tx,
        ty=ty,
        row_axis=row_axis,
        col_axis=col_axis,
        channel_axis=channel_axis,
        fill_mode=fill_mode,
        cval=cval,
        order=interpolation_order,
    )
    return x


@keras_export("keras.preprocessing.image.random_shear")
def random_shear(
    x,
    intensity,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode="nearest",
    cval=0.0,
    interpolation_order=1,
):
    """Performs a random spatial shear of a Numpy image tensor.

    Args:
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    Returns:
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    x = apply_affine_transform(
        x,
        shear=shear,
        row_axis=row_axis,
        col_axis=col_axis,
        channel_axis=channel_axis,
        fill_mode=fill_mode,
        cval=cval,
        order=interpolation_order,
    )
    return x


@keras_export("keras.preprocessing.image.random_zoom")
def random_zoom(
    x,
    zoom_range,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode="nearest",
    cval=0.0,
    interpolation_order=1,
):
    """Performs a random spatial zoom of a Numpy image tensor.

    Deprecated: `tf.keras.preprocessing.image.random_zoom` does not operate on
    tensors and is not recommended for new code. Prefer
    `tf.keras.layers.RandomZoom` which provides equivalent functionality as
    a preprocessing layer. For more information, see the tutorial for
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`

    Returns:
        Zoomed Numpy image tensor.

    Raises:
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError(
            "`zoom_range` should be a tuple or list of two"
            " floats. Received: %s" % (zoom_range,)
        )

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    x = apply_affine_transform(
        x,
        zx=zx,
        zy=zy,
        row_axis=row_axis,
        col_axis=col_axis,
        channel_axis=channel_axis,
        fill_mode=fill_mode,
        cval=cval,
        order=interpolation_order,
    )
    return x


@keras_export("keras.preprocessing.image.apply_channel_shift")
def apply_channel_shift(x, intensity, channel_axis=0):
    """Performs a channel shift.

    Args:
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    Returns:
        Numpy image tensor.
    """
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [
        np.clip(x_channel + intensity, min_x, max_x) for x_channel in x
    ]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


@keras_export("keras.preprocessing.image.random_channel_shift")
def random_channel_shift(x, intensity_range, channel_axis=0):
    """Performs a random channel shift.

    Args:
        x: Input tensor. Must be 3D.
        intensity_range: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    Returns:
        Numpy image tensor.
    """
    intensity = np.random.uniform(-intensity_range, intensity_range)
    return apply_channel_shift(x, intensity, channel_axis=channel_axis)


@keras_export("keras.preprocessing.image.apply_brightness_shift")
def apply_brightness_shift(x, brightness, scale=True):
    """Performs a brightness shift.

    Args:
        x: Input tensor. Must be 3D.
        brightness: Float. The new brightness value.
        scale: Whether to rescale the image such that minimum and maximum values
            are 0 and 255 respectively. Default: True.

    Returns:
        Numpy image tensor.

    Raises:
        ImportError: if PIL is not available.
    """
    if ImageEnhance is None:
        raise ImportError(
            "Using brightness shifts requires PIL. " "Install PIL or Pillow."
        )
    x_min, x_max = np.min(x), np.max(x)
    local_scale = (x_min < 0) or (x_max > 255)
    x = image_utils.array_to_img(x, scale=local_scale or scale)
    x = imgenhancer_Brightness = ImageEnhance.Brightness(x)
    x = imgenhancer_Brightness.enhance(brightness)
    x = image_utils.img_to_array(x)
    if not scale and local_scale:
        x = x / 255 * (x_max - x_min) + x_min
    return x


@keras_export("keras.preprocessing.image.random_brightness")
def random_brightness(x, brightness_range, scale=True):
    """Performs a random brightness shift.

    Deprecated: `tf.keras.preprocessing.image.random_brightness` does not
    operate on tensors and is not recommended for new code. Prefer
    `tf.keras.layers.RandomBrightness` which provides equivalent functionality
    as a preprocessing layer. For more information, see the tutorial for
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        x: Input tensor. Must be 3D.
        brightness_range: Tuple of floats; brightness range.
        scale: Whether to rescale the image such that minimum and maximum values
            are 0 and 255 respectively. Default: True.

    Returns:
        Numpy image tensor.

    Raises:
        ValueError if `brightness_range` isn't a tuple.
    """
    if len(brightness_range) != 2:
        raise ValueError(
            "`brightness_range should be tuple or list of two floats. "
            "Received: %s" % (brightness_range,)
        )

    u = np.random.uniform(brightness_range[0], brightness_range[1])
    return apply_brightness_shift(x, u, scale)


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 - 0.5
    o_y = float(y) / 2 - 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


@keras_export("keras.preprocessing.image.apply_affine_transform")
def apply_affine_transform(
    x,
    theta=0,
    tx=0,
    ty=0,
    shear=0,
    zx=1,
    zy=1,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode="nearest",
    cval=0.0,
    order=1,
):
    """Applies an affine transformation specified by the parameters given.

    Args:
        x: 3D numpy array - a 2D image with one or more channels.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows (aka Y axis) in the input
            image. Direction: left to right.
        col_axis: Index of axis for columns (aka X axis) in the input
            image. Direction: top to bottom.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order: int, order of interpolation

    Returns:
        The transformed version of the input.

    Raises:
        ImportError: if SciPy is not available.
    """
    if scipy is None:
        raise ImportError(
            "Image transformations require SciPy. " "Install SciPy."
        )

    # Input sanity checks:
    # 1. x must 2D image with one or more channels (i.e., a 3D tensor)
    # 2. channels must be either first or last dimension
    if np.unique([row_axis, col_axis, channel_axis]).size != 3:
        raise ValueError(
            "'row_axis', 'col_axis', and 'channel_axis'" " must be distinct"
        )

    # shall we support negative indices?
    valid_indices = set([0, 1, 2])
    actual_indices = set([row_axis, col_axis, channel_axis])
    if actual_indices != valid_indices:
        raise ValueError(
            f"Invalid axis' indices: {actual_indices - valid_indices}"
        )

    if x.ndim != 3:
        raise ValueError("Input arrays must be multi-channel 2D images.")
    if channel_axis not in [0, 2]:
        raise ValueError(
            "Channels are allowed and the first and last dimensions."
        )

    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array(
            [[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]]
        )
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w
        )
        x = np.rollaxis(x, channel_axis, 0)

        # Matrix construction assumes that coordinates are x, y (in that order).
        # However, regular numpy arrays use y,x (aka i,j) indexing.
        # Possible solution is:
        #   1. Swap the x and y axes.
        #   2. Apply transform.
        #   3. Swap the x and y axes again to restore image-like data ordering.
        # Mathematically, it is equivalent to the following transformation:
        # M' = PMP, where P is the permutation matrix, M is the original
        # transformation matrix.
        if col_axis > row_axis:
            transform_matrix[:, [0, 1]] = transform_matrix[:, [1, 0]]
            transform_matrix[[0, 1]] = transform_matrix[[1, 0]]
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [
            ndimage.interpolation.affine_transform(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=order,
                mode=fill_mode,
                cval=cval,
            )
            for x_channel in x
        ]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x
