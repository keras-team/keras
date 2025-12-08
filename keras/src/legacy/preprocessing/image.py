"""Deprecated image preprocessing APIs from Keras 1."""

import collections
import multiprocessing
import os
import threading
import warnings

import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from keras.src.utils import image_utils
from keras.src.utils import io_utils
from keras.src.utils.module_utils import scipy


@keras_export("keras._legacy.preprocessing.image.Iterator")
class Iterator(PyDataset):
    """Base class for image data iterators.

    DEPRECATED.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    Args:
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
        **kwargs: Additional keyword arguments for the `PyDataset` base class,
            such as `workers`, `use_multiprocessing`, and `max_queue_size`.
    """

    white_list_formats = ("png", "jpg", "jpeg", "bmp", "ppm", "tif", "tiff")

    def __init__(self, n, batch_size, shuffle, seed, **kwargs):
        super().__init__(**kwargs)
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

    def __next__(self):
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
                f"Invalid color mode: {color_mode}"
                '; expected "rgb", "rgba", or "grayscale".'
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
                    f"Invalid subset name: {subset};"
                    'expected "training" or "validation"'
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


@keras_export("keras._legacy.preprocessing.image.DirectoryIterator")
class DirectoryIterator(BatchFromFilesMixin, Iterator):
    """Iterator capable of reading images from a directory on disk.

    DEPRECATED.
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


@keras_export("keras._legacy.preprocessing.image.NumpyArrayIterator")
class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    DEPRECATED.
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
                        "Found a pair with: "
                        f"len(x[0]) = {len(x)}, len(x[?]) = {len(xx)}"
                    )
        else:
            x_misc = []

        if y is not None and len(x) != len(y):
            raise ValueError(
                "`x` (images tensor) and `y` (labels) "
                "should have the same length. "
                f"Found: x.shape = {np.asarray(x).shape}, "
                f"y.shape = {np.asarray(y).shape}"
            )
        if sample_weight is not None and len(x) != len(sample_weight):
            raise ValueError(
                "`x` (images tensor) and `sample_weight` "
                "should have the same length. "
                f"Found: x.shape = {np.asarray(x).shape}, "
                f"sample_weight.shape = {np.asarray(sample_weight).shape}"
            )
        if subset is not None:
            if subset not in {"training", "validation"}:
                raise ValueError(
                    f"Invalid subset name: {subset}"
                    '; expected "training" or "validation".'
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
                f"with shape {self.x.shape}"
            )
        channels_axis = 3 if data_format == "channels_last" else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            warnings.warn(
                f"NumpyArrayIterator is set to use the data format convention"
                f' "{data_format}" (channels on axis {channels_axis})'
                ", i.e. expected either 1, 3, or 4 channels "
                f"on axis {channels_axis}. "
                f"However, it was passed an array with shape {self.x.shape}"
                f" ({self.x.shape[channels_axis]} channels)."
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
    """Iterator capable of reading images from a directory as a dataframe."""

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
                f"All values in column x_col={x_col} must be strings."
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
            raise TypeError(f"Column weight_col={weight_col} must be numeric.")

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


@keras_export("keras._legacy.preprocessing.image.ImageDataGenerator")
class ImageDataGenerator:
    """DEPRECATED."""

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
                f"Received: {data_format}"
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
                f" Received: {validation_split}"
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
                f"Received: {zoom_range}"
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
                    f"Received: {brightness_range}"
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
        if "has_ext" in kwargs:
            warnings.warn(
                "has_ext is deprecated, filenames in the dataframe have "
                "to match the exact filenames in disk.",
                DeprecationWarning,
            )
        if "sort" in kwargs:
            warnings.warn(
                "sort is deprecated, batches will be created in the"
                "same order than the filenames provided if `shuffle`"
                "is set to `False`.",
                DeprecationWarning,
            )
        if class_mode == "other":
            warnings.warn(
                '`class_mode="other"` is deprecated, please use '
                '`class_mode="raw"`.',
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
        `featurewise_std_normalization` or `zca_whitening`
        are set to `True`.

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
                "Input to `.fit()` should have rank 4. Got array with shape: "
                + str(x.shape)
            )
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                "Expected input to be images (as Numpy array) "
                f'following the data format convention "{self.data_format}'
                f'" (channels on axis {self.channel_axis})'
                ", i.e. expected either 1, 3 or 4 channels on axis "
                f"{self.channel_axis}. However, it was passed an array with"
                f" shape {x.shape} ({x.shape[self.channel_axis]} channels)."
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


@keras_export("keras._legacy.preprocessing.image.random_rotation")
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
    """DEPRECATED."""
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


@keras_export("keras._legacy.preprocessing.image.random_shift")
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
    """DEPRECATED."""
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


@keras_export("keras._legacy.preprocessing.image.random_shear")
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
    """DEPRECATED."""
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


@keras_export("keras._legacy.preprocessing.image.random_zoom")
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
    """DEPRECATED."""
    if len(zoom_range) != 2:
        raise ValueError(
            "`zoom_range` should be a tuple or list of two floats. "
            f"Received: {zoom_range}"
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


@keras_export("keras._legacy.preprocessing.image.apply_channel_shift")
def apply_channel_shift(x, intensity, channel_axis=0):
    """Performs a channel shift.

    DEPRECATED.

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


@keras_export("keras._legacy.preprocessing.image.random_channel_shift")
def random_channel_shift(x, intensity_range, channel_axis=0):
    """Performs a random channel shift.

    DEPRECATED.

    Args:
        x: Input tensor. Must be 3D.
        intensity_range: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    Returns:
        Numpy image tensor.
    """
    intensity = np.random.uniform(-intensity_range, intensity_range)
    return apply_channel_shift(x, intensity, channel_axis=channel_axis)


@keras_export("keras._legacy.preprocessing.image.apply_brightness_shift")
def apply_brightness_shift(x, brightness, scale=True):
    """Performs a brightness shift.

    DEPRECATED.

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
    from PIL import ImageEnhance

    x_min, x_max = np.min(x), np.max(x)
    local_scale = (x_min < 0) or (x_max > 255)
    x = image_utils.array_to_img(x, scale=local_scale or scale)
    x = imgenhancer_Brightness = ImageEnhance.Brightness(x)
    x = imgenhancer_Brightness.enhance(brightness)
    x = image_utils.img_to_array(x)
    if not scale and local_scale:
        x = x / 255 * (x_max - x_min) + x_min
    return x


@keras_export("keras._legacy.preprocessing.image.random_brightness")
def random_brightness(x, brightness_range, scale=True):
    """Performs a random brightness shift.

    DEPRECATED.

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
            f"Received: {brightness_range}"
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


@keras_export("keras._legacy.preprocessing.image.apply_affine_transform")
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

    DEPRECATED.
    """
    # Input sanity checks:
    # 1. x must 2D image with one or more channels (i.e., a 3D tensor)
    # 2. channels must be either first or last dimension
    if np.unique([row_axis, col_axis, channel_axis]).size != 3:
        raise ValueError(
            "'row_axis', 'col_axis', and 'channel_axis' must be distinct"
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
            scipy.ndimage.interpolation.affine_transform(
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
