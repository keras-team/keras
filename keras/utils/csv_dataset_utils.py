import numpy as np

from keras.api_export import keras_export
from keras.utils import dataset_utils
from keras.utils.module_utils import tensorflow as tf
from keras.utils.module_utils import tensorflow_io as tfio

ALLOWED_FORMATS = (".csv",)


@keras_export("keras.utils.csv_dataset_from_directory")
def csv_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    ragged=False,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
    stride=1,
    head=True,
):
    """Generates a `tf.data.Dataset` from csv files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_data_1.csv
    ......a_data_2.csv
    ...class_b/
    ......b_data_1.csv
    ......b_data_2.csv
    ```

    Then calling `csv_dataset_from_directory(main_directory,
    labels='inferred')`
    will return a `tf.data.Dataset` that yields batches of csv files from
    the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Only `.csv` files are supported at this time.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain subdirectories,
            each containing csv files for a class. Otherwise, the directory
            structure is ignored.
        labels: Either "inferred" (labels are generated from the directory
            structure), `None` (no labels), or a list/tuple of integer labels
            of the same size as the number of csv files found in
            the directory. Labels should be sorted according to the
            alphanumeric order of the csv file paths
            (obtained via `os.walk(directory)` in Python).
        label_mode: String describing the encoding of `labels`. Options are:
            - `"int"`: means that the labels are encoded as integers (e.g. for
              `sparse_categorical_crossentropy` loss).
            - `"categorical"` means that the labels are encoded as a categorical
              vector (e.g. for `categorical_crossentropy` loss)
            - `"binary"` means that the labels (there can be only 2)
              are encoded as `float32` scalars with values 0
              or 1 (e.g. for `binary_crossentropy`).
            - `None` (no labels).
        class_names: Only valid if "labels" is `"inferred"`.
            This is the explicit list of class names
            (must match names of subdirectories). Used to control the order
            of the classes (otherwise alphanumerical order is used).
        batch_size: Size of the batches of data. Default: 32. If `None`,
            the data will not be batched
            (the dataset will yield individual samples).
        ragged: Whether to return a Ragged dataset (where each sequence has its
            own length). Defaults to `False`.
        shuffle: Whether to shuffle the data. Defaults to `True`.
            If set to `False`, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1, fraction of data to
            reserve for validation.
        subset: Subset of the data to return. One of `"training"`,
            `"validation"` or `"both"`. Only used if `validation_split` is set.
        follow_links: Whether to visits subdirectories pointed to by symlinks.
            Defaults to `False`.
        head: If the csv files contains head or not.
        stride: If you desire any stride upon you data.

    Returns:

    A `tf.data.Dataset` object.

    - If `label_mode` is `None`, it yields `string` tensors of shape
      `(batch_size,)`, containing the contents of a batch of csv files.
    - Otherwise, it yields a tuple `(csv, labels)`, where `csv`
      has shape `(batch_size, sequence_length, num_channels)` and `labels`
      follows the format described
      below.

    Rules regarding labels format:

    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
      `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
      1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorical`, the labels are a `float32` tensor
      of shape `(batch_size, num_classes)`, representing a one-hot
      encoding of the class index.
    """
    if labels not in ("inferred", None):
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                "The `labels` argument should be a list/tuple of integer "
                "labels, of the same size as the number of csv files in "
                "the target directory. If you wish to infer the labels from "
                "the subdirectory names in the target directory,"
                ' pass `labels="inferred"`. '
                "If you wish to get a dataset that only contains csvs"
                f"(no labels), pass `labels=None`. Received: labels={labels}"
            )
        if class_names:
            raise ValueError(
                "You can only pass `class_names` if "
                f'`labels="inferred"`. Received: labels={labels}, and '
                f"class_names={class_names}"
            )
    if label_mode not in {"int", "categorical", "binary", None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", "categorical", '
            '"binary", '
            f"or None. Received: label_mode={label_mode}"
        )

    if labels is None or label_mode is None:
        labels = None
        label_mode = None

    dataset_utils.check_validation_split_arg(
        validation_split, subset, shuffle, seed
    )

    if seed is None:
        seed = np.random.randint(1e6)

    file_paths, labels, class_names = dataset_utils.index_directory(
        directory,
        labels,
        formats=ALLOWED_FORMATS,
        class_names=class_names,
        shuffle=shuffle,
        seed=seed,
        follow_links=follow_links,
    )

    if label_mode == "binary" and len(class_names) != 2:
        raise ValueError(
            'When passing `label_mode="binary"`, there must be exactly 2 '
            f"class_names. Received: class_names={class_names}"
        )

    if subset == "both":
        train_dataset, val_dataset = get_training_and_validation_dataset(
            file_paths=file_paths,
            labels=labels,
            validation_split=validation_split,
            directory=directory,
            label_mode=label_mode,
            class_names=class_names,
            ragged=ragged,
            stride=stride,
            head=head
        )
        train_dataset = prepare_dataset(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            class_names=class_names,
            ragged=ragged,
        )
        val_dataset = prepare_dataset(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            class_names=class_names,
            ragged=ragged,
        )
        return train_dataset, val_dataset

    else:
        dataset = get_dataset(
            file_paths=file_paths,
            labels=labels,
            directory=directory,
            validation_split=validation_split,
            subset=subset,
            label_mode=label_mode,
            class_names=class_names,
            ragged=ragged,
            stride=stride,
            head=head
        )
        dataset = prepare_dataset(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            class_names=class_names,
            ragged=ragged,
        )
        return dataset


def prepare_dataset(
    dataset,
    batch_size,
    shuffle,
    seed,
    class_names,
    ragged,
):
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)

        if not ragged:
            dataset = dataset.padded_batch(
                batch_size, padded_shapes=([None, None], [])
            )
        else:
            dataset = dataset.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)

    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    return dataset


def get_training_and_validation_dataset(
    file_paths,
    labels,
    validation_split,
    directory,
    label_mode,
    class_names,
    ragged,
    stride,
        head
):
    (
        file_paths_train,
        labels_train,
    ) = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, "training"
    )
    if not file_paths_train:
        raise ValueError(
            f"No training csv files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    file_paths_val, labels_val = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, "validation"
    )
    if not file_paths_val:
        raise ValueError(
            f"No validation csv files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    train_dataset = paths_and_labels_to_dataset(
        file_paths=file_paths_train,
        labels=labels_train,
        label_mode=label_mode,
        num_classes=len(class_names) if class_names else 0,
        ragged=ragged,
        stride=stride,
        head=head
    )

    val_dataset = paths_and_labels_to_dataset(
        file_paths=file_paths_val,
        labels=labels_val,
        label_mode=label_mode,
        num_classes=len(class_names) if class_names else 0,
        ragged=ragged,
        stride=stride,
        head=head
    )

    return train_dataset, val_dataset


def get_dataset(
    file_paths,
    labels,
    directory,
    validation_split,
    subset,
    label_mode,
    class_names,
    ragged,
    stride,
    head
):
    file_paths, labels = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, subset
    )
    if not file_paths:
        raise ValueError(
            f"No csv files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    dataset = paths_and_labels_to_dataset(
        file_paths=file_paths,
        labels=labels,
        label_mode=label_mode,
        num_classes=len(class_names) if class_names else 0,
        ragged=ragged,
        stride=stride,
        head=head
    )

    return dataset


def getReadings(path, stride: int = 0, head: bool = True):
    return tf.strings.to_number(tf.strings.split(tf.strings.split(tf.io.read_file(path)), sep=","), out_type=tf.float32)[1::stride]


def paths_and_labels_to_dataset(
    file_paths,
    labels,
    label_mode,
    num_classes,
    ragged,
    stride,
    head
):
    """Constructs a fixed-size dataset of csvs and labels."""
    args = {"stride": stride, "head": head}
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    readings_ds = path_ds.map(
        lambda x: getReadings(
            x, **args
        ).to_tensor(),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if ragged:
        readings_ds = readings_ds.map(
            lambda x: tf.RaggedTensor.from_tensor(x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if label_mode:
        label_ds = dataset_utils.labels_to_dataset(
            labels, label_mode, num_classes
        )
        readings_ds = tf.data.Dataset.zip((readings_ds, label_ds))
    return readings_ds
