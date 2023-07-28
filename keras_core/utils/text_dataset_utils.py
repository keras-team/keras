from keras_core.api_export import keras_core_export
from keras_core.utils.module_utils import tensorflow as tf


@keras_core_export(
    [
        "keras_core.utils.text_dataset_from_directory",
        "keras_core.preprocessing.text_dataset_from_directory",
    ]
)
def text_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    max_length=None,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
):
    """Generates a `tf.data.Dataset` from text files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_text_1.txt
    ......a_text_2.txt
    ...class_b/
    ......b_text_1.txt
    ......b_text_2.txt
    ```

    Then calling `text_dataset_from_directory(main_directory,
    labels='inferred')` will return a `tf.data.Dataset` that yields batches of
    texts from the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Only `.txt` files are supported at this time.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain
            subdirectories, each containing text files for a class.
            Otherwise, the directory structure is ignored.
        labels: Either `"inferred"`
            (labels are generated from the directory structure),
            `None` (no labels),
            or a list/tuple of integer labels of the same size as the number of
            text files found in the directory. Labels should be sorted according
            to the alphanumeric order of the text file paths
            (obtained via `os.walk(directory)` in Python).
        label_mode: String describing the encoding of `labels`. Options are:
            - `"int"`: means that the labels are encoded as integers
                (e.g. for `sparse_categorical_crossentropy` loss).
            - `"categorical"` means that the labels are
                encoded as a categorical vector
                (e.g. for `categorical_crossentropy` loss).
            - `"binary"` means that the labels (there can be only 2)
                are encoded as `float32` scalars with values 0 or 1
                (e.g. for `binary_crossentropy`).
            - `None` (no labels).
        class_names: Only valid if `"labels"` is `"inferred"`.
            This is the explicit list of class names
            (must match names of subdirectories). Used to control the order
            of the classes (otherwise alphanumerical order is used).
        batch_size: Size of the batches of data. Defaults to 32.
            If `None`, the data will not be batched
            (the dataset will yield individual samples).
        max_length: Maximum size of a text string. Texts longer than this will
            be truncated to `max_length`.
        shuffle: Whether to shuffle the data. Defaults to `True`.
            If set to `False`, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: Subset of the data to return.
            One of `"training"`, `"validation"` or `"both"`.
            Only used if `validation_split` is set.
            When `subset="both"`, the utility returns a tuple of two datasets
            (the training and validation datasets respectively).
        follow_links: Whether to visits subdirectories pointed to by symlinks.
            Defaults to `False`.

    Returns:

    A `tf.data.Dataset` object.

    - If `label_mode` is `None`, it yields `string` tensors of shape
        `(batch_size,)`, containing the contents of a batch of text files.
    - Otherwise, it yields a tuple `(texts, labels)`, where `texts`
        has shape `(batch_size,)` and `labels` follows the format described
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
    # TODO: long-term, port implementation.
    return tf.keras.utils.text_dataset_from_directory(
        directory,
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        follow_links=follow_links,
    )
