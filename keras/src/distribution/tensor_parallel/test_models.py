import logging
import os
import sys
import time

import numpy as np

try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
except Exception:
    print(
        "Could not add project root to sys.path. "
        "Please run from the 'keras' directory or install as a package."
    )

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import keras_nlp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

import keras

# --- Configuration and Initialization ---
# This ensures we don't accidentally use GPUs if they are available,
# forcing the test to run on the simulated CPU/TPU devices from XLA_FLAGS.
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- JAX Device Detection ---
# Verifies that JAX can see the simulated devices from XLA_FLAGS.
try:
    devices = jax.devices()
    logger.info(f"JAX devices found: {[str(d) for d in devices]}")
    # Prefer CPUs for this host-simulated test
    host_devices = [d for d in devices if d.platform == "cpu"]
    if not host_devices:
        host_devices = devices  # Fallback to any available device

    DEVICES_AVAILABLE = len(host_devices)
    WORLD_SIZE = 2  # Hardcode target world size for a 2-way sharded test

    if DEVICES_AVAILABLE < WORLD_SIZE:
        logger.warning(
            f"Requested {WORLD_SIZE} devices, but only {DEVICES_AVAILABLE} available."
        )
        TARGET_DEVICES = host_devices
        TARGET_WORLD_SIZE = DEVICES_AVAILABLE
    else:
        TARGET_DEVICES = host_devices[:WORLD_SIZE]
        TARGET_WORLD_SIZE = WORLD_SIZE
        logger.info(
            f"Targeting the first {TARGET_WORLD_SIZE} devices for parallelism: "
            f"{[str(d) for d in TARGET_DEVICES]}"
        )

except Exception as e:
    logger.error(f"Could not initialize JAX or find devices. Error: {e}")
    TARGET_WORLD_SIZE = 0


# --- STEP 3: Import the custom TensorParallelKeras class ---
# This import now directly attempts to load the real implementation.
# If it fails, the script will raise an ImportError and stop, which is
# the desired behavior for a non-mock test.
from keras.src.distribution.tensor_parallel.tensorparallelkeras import (
    TensorParallelKeras,
)

# --- Constants ---
BATCH_SIZE = 16
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4
EPOCHS = 2
STEPS_PER_EPOCH = 10  # Keep low for a quick test
VALIDATION_STEPS = 5

MODEL_MAPPING = {
    "opt_125m_en": keras_nlp.models.OPTCausalLM,
}

# ----------------------------------------------------------------------
# --- Dataset and Model Helpers ---
# ----------------------------------------------------------------------


def load_shakespeare_dataset(model_preset):
    """Loads and preprocesses the Tiny Shakespeare dataset."""
    logger.info(
        f"Loading and preprocessing Tiny Shakespeare dataset for {model_preset}..."
    )
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(
        example["text"].decode("utf-8") for example in ds.as_numpy_iterator()
    )

    tokenizer = keras_nlp.models.OPTCausalLM.from_preset(
        model_preset
    ).preprocessor.tokenizer
    token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (
        SEQUENCE_LENGTH + 1
    )
    sequences = np.array(token_ids[:num_tokens]).reshape(
        -1, SEQUENCE_LENGTH + 1
    )

    all_data = tf.data.Dataset.from_tensor_slices(sequences)

    num_sequences = sequences.shape[0]
    num_train_samples = int(0.9 * num_sequences)

    train_ds = all_data.take(num_train_samples)
    val_ds = all_data.skip(num_train_samples)

    logger.info(
        f"Dataset ready with {num_train_samples} training and "
        f"{num_sequences - num_train_samples} validation sequences."
    )
    return train_ds, val_ds


def format_for_causal_lm(data):
    """Formats data for KerasNLP's CausalLM, creating features and labels."""
    features = {
        "token_ids": data[:, :-1],
        "padding_mask": tf.ones_like(data[:, :-1], dtype=tf.bool),
    }
    labels = data[:, 1:]
    return features, labels


def get_model_from_preset(preset_name, model_class):
    """Creates a CausalLM model from a KerasNLP preset."""
    logger.info(f"Creating {preset_name} model from KerasNLP preset...")
    model = model_class.from_preset(preset_name, preprocessor=None)
    logger.info(f"Model created with {model.count_params():,} parameters.")
    return model


# ----------------------------------------------------------------------
# --- Plotting Function ---
# ----------------------------------------------------------------------


def plot_training_graphs(tp_history, preset_name):
    """Plots and saves the loss and perplexity graphs for TP training."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"{preset_name} - Tensor Parallel Training", fontsize=16)

    ax1.plot(
        tp_history.history["loss"],
        label="Tensor Parallel - Training Loss",
        color="green",
        linestyle="-",
    )
    ax1.plot(
        tp_history.history["val_loss"],
        label="Tensor Parallel - Validation Loss",
        color="green",
        linestyle="--",
    )
    ax1.set_title("Training and Validation Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(
        tp_history.history["perplexity"],
        label="Tensor Parallel - Training Perplexity",
        color="purple",
        linestyle="-",
    )
    ax2.plot(
        tp_history.history["val_perplexity"],
        label="Tensor Parallel - Validation Perplexity",
        color="purple",
        linestyle="--",
    )
    ax2.set_title("Training and Validation Perplexity")
    ax2.set_ylabel("Perplexity")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    output_filename = f"{preset_name}_tp_verification.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    logger.info(f"\nTraining graph saved to {output_filename}")
    plt.close()


# ----------------------------------------------------------------------
# --- Main Verification Function ---
# ----------------------------------------------------------------------


def run_model_verification(preset_name, model_class):
    """Runs the full training verification test for a given model preset."""

    if TARGET_WORLD_SIZE < 2:
        logger.warning(
            f"SKIPPING {preset_name}: Need at least 2 devices for tensor "
            f"parallelism, found {TARGET_WORLD_SIZE}"
        )
        return "SKIPPED"

    logger.info(f"--- VERIFICATION FOR: {preset_name.upper()} ---")
    start_time_total = time.time()

    model_template = get_model_from_preset(preset_name, model_class)
    initial_weights = model_template.get_weights()
    logger.info("Initial weights saved from template model.")

    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name)

    train_ds = (
        train_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )
    val_ds = (
        val_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )

    total_steps = STEPS_PER_EPOCH * EPOCHS
    total_samples = total_steps * BATCH_SIZE
    total_tokens_processed = total_samples * SEQUENCE_LENGTH
    logger.info(f"Total tokens to process: {total_tokens_processed:,}")

    logger.info("\n--- Training Tensor Parallel (TP) Model ---")

    logger.info(
        f"Initializing TensorParallelKeras with world_size={TARGET_WORLD_SIZE} "
        f"on devices: {[str(d) for d in TARGET_DEVICES]}"
    )

    # The TensorParallelKeras class itself is the model to be trained.
    # It wraps the original model and manages the sharded versions internally.
    tp_model = TensorParallelKeras(
        model=model_template,
        world_size=TARGET_WORLD_SIZE,
        distributed_backend="jax",
        device_ids=TARGET_DEVICES,
    )

    # Set weights on the original model before sharding logic uses them.
    tp_model.original_model.set_weights(initial_weights)
    logger.info("Initial weights set on TP model.")

    # Compiling the TP model automatically sets up the CoordinatedOptimizer
    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras_nlp.metrics.Perplexity(from_logits=True, name="perplexity")
        ],
    )

    tp_start_time = time.time()
    tp_history = tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
    )
    tp_end_time = time.time()
    logger.info("TP model training completed.")

    tp_time = tp_end_time - tp_start_time
    tp_throughput_tps = total_tokens_processed / tp_time if tp_time > 0 else 0

    logger.info("\n--- Performance Metrics ---")
    logger.info(f"TP Training Time:       {tp_time:.2f} s")
    logger.info(f"TP Throughput:          {tp_throughput_tps:,.2f} Tokens/s")

    tp_final_val_loss = tp_history.history["val_loss"][-1]
    logger.info(f"Final Validation Loss: {tp_final_val_loss:.4f}")

    plot_training_graphs(tp_history, preset_name)

    logger.info(
        f"Test for {preset_name} completed in "
        f"{time.time() - start_time_total:.2f}s"
    )
    # A simple pass/fail check: training should not fail and loss should be a valid number.
    return not np.isnan(tp_final_val_loss) and not np.isinf(tp_final_val_loss)


# ----------------------------------------------------------------------
# --- Main Execution ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if TARGET_WORLD_SIZE == 0:
        logger.critical("No JAX devices found. Aborting verification suite.")
        sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info("      TENSOR PARALLELISM VERIFICATION SUITE")
    logger.info("=" * 70)

    results = {}
    total_start_time = time.time()

    for preset, model_class in MODEL_MAPPING.items():
        try:
            result = run_model_verification(preset, model_class)
            if result == "SKIPPED":
                results[preset] = "⚪ SKIPPED"
            else:
                results[preset] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            logger.error(
                f"Test for {preset} failed with an exception: {e}",
                exc_info=True,
            )
            results[preset] = "💥 ERROR"
        logger.info("-" * 70)

    logger.info("\n" + "=" * 70)
    logger.info("🎉 VERIFICATION SUITE COMPLETED!")
    logger.info(
        f"   Total execution time: {time.time() - total_start_time:.2f}s"
    )
    logger.info("\n   --- SUMMARY ---")
    for preset, status in results.items():
        print(f"   - {preset:<18}: {status}")
    logger.info("=" * 70)
