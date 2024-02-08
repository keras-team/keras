"""Benchmark BERT model on GLUE/MRPC task.

To run the script, make sure you are in benchmarks/ directory, abd run the
command below:
```
python3 -m model_benchmark.bert_benchmark \
    --epochs 2 \
    --batch_size 32
```

"""

import time

import keras_nlp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from absl import logging
from model_benchmark.benchmark_utils import BenchmarkMetricsCallback

import keras

flags.DEFINE_string("model_size", "small", "The size of model to benchmark.")
flags.DEFINE_string(
    "mixed_precision_policy",
    "mixed_float16",
    "The global precision policy to use, e.g., 'mixed_float16' or 'float32'.",
)
flags.DEFINE_integer("epochs", 2, "The number of epochs.")
flags.DEFINE_integer("batch_size", 8, "Batch Size.")


FLAGS = flags.FLAGS


MODEL_SIZE_MAP = {
    "tiny": "bert_tiny_en_uncased",
    "small": "bert_small_en_uncased",
    "base": "bert_base_en_uncased",
    "large": "bert_large_en_uncased",
}


def load_data():
    """Load data.

    Load GLUE/MRPC dataset, and convert the dictionary format to
    (features, label), where `features` is a tuple of all input sentences.
    """
    feature_names = ("sentence1", "sentence2")

    def split_features(x):
        # GLUE comes with dictionary data, we convert it to a uniform format
        # (features, label), where features is a tuple consisting of all
        # features. This format is necessary for using KerasNLP preprocessors.
        features = tuple([x[name] for name in feature_names])
        label = x["label"]
        return (features, label)

    train_ds, test_ds, validation_ds = tfds.load(
        "glue/mrpc",
        split=["train", "test", "validation"],
    )

    train_ds = (
        train_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(FLAGS.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(FLAGS.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_ds = (
        validation_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(FLAGS.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds, validation_ds


def load_model():
    if FLAGS.model_size not in MODEL_SIZE_MAP.keys():
        raise KeyError(
            f"`model_size` must be one of {MODEL_SIZE_MAP.keys()}, but "
            f"received {FLAGS.model_size}."
        )
    return keras_nlp.models.BertClassifier.from_preset(
        MODEL_SIZE_MAP[FLAGS.model_size], num_classes=2
    )


def main(_):
    keras.mixed_precision.set_dtype_policy(FLAGS.mixed_precision_policy)

    logging.info(
        "Benchmarking configs...\n"
        "=========================\n"
        f"MODEL: BERT {FLAGS.model_size}\n"
        f"TASK: glue/mrpc \n"
        f"BATCH_SIZE: {FLAGS.batch_size}\n"
        f"EPOCHS: {FLAGS.epochs}\n"
        "=========================\n"
    )

    # Load datasets.
    train_ds, test_ds, validation_ds = load_data()

    # Load the model.
    model = load_model()
    # Set loss and metrics.
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.SparseCategoricalAccuracy()]
    # Configure optimizer.
    lr = keras.optimizers.schedules.PolynomialDecay(
        5e-4,
        decay_steps=train_ds.cardinality() * FLAGS.epochs,
        end_learning_rate=0.0,
    )
    optimizer = keras.optimizers.AdamW(lr, weight_decay=0.01)
    optimizer.exclude_from_weight_decay(
        var_names=["LayerNorm", "layer_norm", "bias"]
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    benchmark_metrics_callback = BenchmarkMetricsCallback(
        start_batch=1,
        stop_batch=train_ds.cardinality().numpy() - 1,
    )

    # Start training.
    logging.info("Starting Training...")

    st = time.time()
    history = model.fit(
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
