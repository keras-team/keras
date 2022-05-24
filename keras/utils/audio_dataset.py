# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Keras audio dataset loading utilities."""

import tensorflow.compat.v2 as tf

# pylint: disable=g-classes-have-attributes

import numpy as np

from keras.utils import dataset_utils
from tensorflow.python.util.tf_export import keras_export

try:
    import tensorflow_io as tfio
except ImportError:
    tfio = None

ALLOWED_FORMATS = (".wav",)


@keras_export("keras.utils.audio_dataset_from_directory", v1=[])
def audio_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    sampling_rate=None,
    output_sequence_length=None,
    ragged=False,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
):
    """Generates a `tf.data.Dataset` from audio files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_audio_1.wav
    ......a_audio_2.wav
    ...class_b/
    ......b_audio_1.wav
    ......b_audio_2.wav
    ```

    Then calling `audio_dataset_from_directory(main_directory,
    labels='inferred')`
    will return a `tf.data.Dataset` that yields batches of audio files from
    the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Only `.wav` files are supported at this time.

    Args:
      directory: Directory where the data is located. If `labels` is "inferred",
        it should contain subdirectories, each containing audio files for a
        class. Otherwise, the directory structure is ignored.
      labels: Either "inferred" (labels are generated from the directory
        structure), None (no labels), or a list/tuple of integer labels of the
        same size as the number of audio files found in the directory. Labels
        should be sorted according to the alphanumeric order of the audio file
        paths (obtained via `os.walk(directory)` in Python).
      label_mode: String describing the encoding of `labels`. Options are:
          - 'int': means that the labels are encoded as integers (e.g. for
            `sparse_categorical_crossentropy` loss). - 'categorical' means that
            the labels are encoded as a categorical vector (e.g. for
            `categorical_crossentropy` loss). - 'binary' means that the labels
            (there can be only 2) are encoded as `float32` scalars with values 0
            or 1 (e.g. for `binary_crossentropy`). - None (no labels).
      class_names: Only valid if "labels" is "inferred". This is the explicit
        list of class names (must match names of subdirectories). Used to
        control the order of the classes (otherwise alphanumerical order is
        used).
      batch_size: Size of the batches of data. Default: 32. If `None`, the data
        will not be batched (the dataset will yield individual samples).
      sampling_rate: Audio sampling rate (in samples per second).
      output_sequence_length: Maximum length of an audio sequence. Audio files
        longer than this will be truncated to `output_sequence_length`. If set
        to `None`, then all sequences in the same batch will be padded to the
        length of the longest sequence in the batch.
      ragged: Whether to return a Ragged dataset (where each sequence has its
        own length). Default: False.
      shuffle: Whether to shuffle the data. Default: True. If set to False,
        sorts the data in alphanumeric order.
      seed: Optional random seed for shuffling and transformations.
      validation_split: Optional float between 0 and 1, fraction of data to
        reserve for validation.
      subset: Subset of the data to return. One of "training", "validation" or
        "both". Only used if `validation_split` is set.
      follow_links: Whether to visits subdirectories pointed to by symlinks.
        Defaults to False.

    Returns:
      A `tf.data.Dataset` object.
        - If `label_mode` is None, it yields `string` tensors of shape
          `(batch_size,)`, containing the contents of a batch of audio files.
        - Otherwise, it yields a tuple `(audio, labels)`, where `audio`
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
                "The `labels` argument should be a list/tuple of integer labels, of "
                "the same size as the number of audio files in the target "
                "directory. If you wish to infer the labels from the subdirectory "
                'names in the target directory, pass `labels="inferred"`. '
                "If you wish to get a dataset that only contains audio samples "
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
            '`label_mode` argument must be one of "int", "categorical", "binary", '
            f"or None. Received: label_mode={label_mode}"
        )

    if ragged and output_sequence_length is not None:
        raise ValueError(
            "Cannot set both `ragged` and `output_sequence_length`"
        )

    if sampling_rate is not None:
        if not isinstance(sampling_rate, int):
            raise ValueError(
                "`sampling_rate` should have an integer value. "
                f"Received: sampling_rate={sampling_rate}"
            )

        if sampling_rate <= 0:
            raise ValueError(
                f"`sampling_rate` should be higher than 0. "
                f"Received: sampling_rate={sampling_rate}"
            )

        if tfio is None:
            raise ImportError(
                "To use the argument `sampling_rate`, you should install "
                "tensorflow_io. You can install it via `pip install tensorflow-io`."
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
            f'When passing `label_mode="binary"`, there must be exactly 2 '
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
            sampling_rate=sampling_rate,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
        )

        train_dataset = prepare_dataset(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            class_names=class_names,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
        )
        val_dataset = prepare_dataset(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            class_names=class_names,
            output_sequence_length=output_sequence_length,
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
            sampling_rate=sampling_rate,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
        )

        dataset = prepare_dataset(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            class_names=class_names,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
        )
        return dataset


def prepare_dataset(
    dataset,
    batch_size,
    shuffle,
    seed,
    class_names,
    output_sequence_length,
    ragged,
):
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)

        if output_sequence_length is None and not ragged:
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
    sampling_rate,
    output_sequence_length,
    ragged,
):
    (
        file_paths_train,
        labels_train,
    ) = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, "training"
    )
    if not file_paths_train:
        raise ValueError(
            f"No training audio files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    file_paths_val, labels_val = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, "validation"
    )
    if not file_paths_val:
        raise ValueError(
            f"No validation audio files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    train_dataset = paths_and_labels_to_dataset(
        file_paths=file_paths_train,
        labels=labels_train,
        label_mode=label_mode,
        num_classes=len(class_names),
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        ragged=ragged,
    )

    val_dataset = paths_and_labels_to_dataset(
        file_paths=file_paths_val,
        labels=labels_val,
        label_mode=label_mode,
        num_classes=len(class_names),
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        ragged=ragged,
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
    sampling_rate,
    output_sequence_length,
    ragged,
):
    file_paths, labels = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, subset
    )
    if not file_paths:
        raise ValueError(
            f"No audio files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    dataset = paths_and_labels_to_dataset(
        file_paths=file_paths,
        labels=labels,
        label_mode=label_mode,
        num_classes=len(class_names),
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        ragged=ragged,
    )

    return dataset


def read_and_decode_audio(
    path, sampling_rate=None, output_sequence_length=None
):
    """Reads and decodes audio file."""
    audio = tf.io.read_file(path)

    if output_sequence_length is None:
        output_sequence_length = -1

    audio, default_audio_rate = tf.audio.decode_wav(
        contents=audio, desired_samples=output_sequence_length
    )
    if sampling_rate is not None:
        # default_audio_rate should have dtype=int64
        default_audio_rate = tf.cast(default_audio_rate, tf.int64)
        audio = tfio.audio.resample(
            input=audio, rate_in=default_audio_rate, rate_out=sampling_rate
        )
    return audio


def paths_and_labels_to_dataset(
    file_paths,
    labels,
    label_mode,
    num_classes,
    sampling_rate,
    output_sequence_length,
    ragged,
):
    """Constructs a fixed-size dataset of audio and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    audio_ds = path_ds.map(
        lambda x: read_and_decode_audio(
            x, sampling_rate, output_sequence_length
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if ragged:
        audio_ds = audio_ds.map(
            lambda x: tf.RaggedTensor.from_tensor(x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if label_mode:
        label_ds = dataset_utils.labels_to_dataset(
            labels, label_mode, num_classes
        )
        audio_ds = tf.data.Dataset.zip((audio_ds, label_ds))
    return audio_ds
