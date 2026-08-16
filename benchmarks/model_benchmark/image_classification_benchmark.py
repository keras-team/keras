"""Image classification benchmark.

This script runs image classification benchmark with "dogs vs cats" datasets.
It supports the following 3 models:

- EfficientNetV2B0
- Xception
- ResNet50V2

To run the benchmark, make sure you are in model_benchmark/ directory, and run
the command below:

python3 -m model_benchmark.image_classification_benchmark \
    --model="EfficientNetV2B0" \
    --epochs=2 \
    --batch_size=32 \
    --mixed_precision_policy="mixed_float16"
"""

import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from absl import logging
from model_benchmark.benchmark_utils import BenchmarkMetricsCallback

import keras

flags.DEFINE_string("model", "EfficientNetV2B0", "The model to benchmark.")
flags.DEFINE_integer("epochs", 1, "The number of epochs.")
flags.DEFINE_integer("batch_size", 4, "Batch Size.")
flags.DEFINE_string(
    "mixed_precision_policy",
    "mixed_float16",
    "The global precision policy to use, e.g., 'mixed_float16' or 'float32'.",
)

FLAGS = flags.FLAGS

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
CHANNELS = 3

MODEL_MAP = {
    "EfficientNetV2B0": keras.applications.EfficientNetV2B0,
    "Xception": keras.applications.Xception,
    "ResNet50V2": keras.applications.ResNet50V2,
}


def load_data():
    # Load cats vs dogs dataset, and split into train and validation sets.
    train_dataset, val_dataset = tfds.load(
        "cats_vs_dogs", split=["train[:90%]", "train[90%:]"], as_supervised=True
    )

    resizing = keras.layers.Resizing(
        IMAGE_SIZE[0], IMAGE_SIZE[1], crop_to_aspect_ratio=True
    )

    def preprocess_inputs(image, label):
        image = tf.cast(image, "float32")
        return resizing(image), label

    train_dataset = (
        train_dataset.map(
            preprocess_inputs, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(FLAGS.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        val_dataset.map(preprocess_inputs, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(FLAGS.batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_dataset, val_dataset


def load_model():
    model_class = MODEL_MAP[FLAGS.model]
    # Load the EfficientNetV2B0 model and add a classification head.
    model = model_class(include_top=False, weights="imagenet")
    classifier = keras.models.Sequential(
        [
            keras.Input([IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS]),
            model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(2),
        ]
    )
    return classifier


def main(_):
    keras.mixed_precision.set_dtype_policy(FLAGS.mixed_precision_policy)

    logging.info(
        "Benchmarking configs...\n"
        "=========================\n"
        f"MODEL: {FLAGS.model}\n"
        f"TASK: image classification/dogs-vs-cats \n"
        f"BATCH_SIZE: {FLAGS.batch_size}\n"
        f"EPOCHS: {FLAGS.epochs}\n"
        "=========================\n"
    )

    # Load datasets.
    train_ds, validation_ds = load_data()

    # Load the model.
    classifier = load_model()

    lr = keras.optimizers.schedules.PolynomialDecay(
        5e-4,
        decay_steps=train_ds.cardinality() * FLAGS.epochs,
        end_learning_rate=0.0,
    )
    optimizer = keras.optimizers.Adam(lr)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    benchmark_metrics_callback = BenchmarkMetricsCallback(
        start_batch=1,
        stop_batch=train_ds.cardinality().numpy() - 1,
    )

    classifier.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["sparse_categorical_accuracy"],
    )
    # Start training.
    logging.info("Starting Training...")

    st = time.time()

    history = classifier.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=FLAGS.epochs,
        callbacks=[benchmark_metrics_callback],
    )

    wall_time = time.time() - st
    validation_accuracy = history.history["val_sparse_categorical_accuracy"][-1]

    examples_per_second = (
        np.mean(np.array(benchmark_metrics_callback.state["throughput"]))
        * FLAGS.batch_size
    )

    logging.info("Training Finished!")
    logging.info(f"Wall Time: {wall_time:.4f} seconds.")
    logging.info(f"Validation Accuracy: {validation_accuracy:.4f}")
    logging.info(f"examples_per_second: {examples_per_second:.4f}")


if __name__ == "__main__":
    app.run(main)
