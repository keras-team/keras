#!/usr/bin/env python3
"""
Test script for Keras Hub LiteRT export functionality.

This script loads a Keras Hub model, exports it to LiteRT format,
and verifies numerical accuracy between original and exported models.

Change MODEL_PRESET in load_model() to test different models.
"""

import os
import time
import tempfile
import numpy as np
from pathlib import Path

# Configure environment
print("🔧 Configuring Keras to use the TensorFlow backend...")
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["KAGGLE_KEY"]="20fd7df00ecb83cf98c73dc97029f650"
os.environ["KAGGLE_USERNAME"]="pctablet505"

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Configure GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

import keras
import keras_hub

# Check LiteRT availability
try:
    import ai_edge_litert
    LITERT_AVAILABLE = True
    print("✅ ai_edge_litert is available")
except ImportError:
    try:
        import tensorflow.lite as ai_edge_litert
        LITERT_AVAILABLE = True
        print("✅ Using tensorflow.lite as ai_edge_litert")
    except ImportError:
        LITERT_AVAILABLE = False
        print("❌ LiteRT not available")

def load_model():
    """Load the specified model with random weights."""
    print("\n📦 Loading model...")

    # Change this to test different models
    MODEL_PRESET = "llama3.2_1b"
    MODEL_PRESET = "gemma3_1b"
    # Examples: "gpt2_base_en", "gemma_2b_en", "mistral_7b_en", "phi3_mini_4k_instruct_en"

    model_name = MODEL_PRESET.replace(".", "_").replace("/", "_")

    # # Try to load existing saved model
    # if os.path.exists(saved_model_path):
    #     print(f"✅ Loading existing model from {saved_model_path}")
    #     try:
    #         model = keras.models.load_model(saved_model_path)
    #         print(f"📏 Sequence length: {model.preprocessor.sequence_length}")
    #         model.summary()
    #         return model
    #     except Exception as e:
    #         print(f"⚠️  Failed to load saved model: {e}")

    # Load from preset
    try:
        if "gpt" in MODEL_PRESET.lower():
            model = keras_hub.models.GptCausalLM.from_preset(MODEL_PRESET, load_weights=False)
        elif "gemma" in MODEL_PRESET.lower():
            model = keras_hub.models.Gemma3CausalLM.from_preset(MODEL_PRESET, load_weights=False)
        elif "llama" in MODEL_PRESET.lower():
            model = keras_hub.models.Llama3CausalLM.from_preset(MODEL_PRESET, load_weights=False)
        elif "mistral" in MODEL_PRESET.lower():
            model = keras_hub.models.MistralCausalLM.from_preset(MODEL_PRESET, load_weights=False)
        elif "phi" in MODEL_PRESET.lower():
            model = keras_hub.models.Phi3CausalLM.from_preset(MODEL_PRESET, load_weights=False)
        else:
            # Generic fallback
            import keras_hub.models as models
            for model_class_name in ['GptCausalLM', 'Gemma3CausalLM', 'Llama3CausalLM',
                                   'MistralCausalLM', 'Phi3CausalLM']:
                if hasattr(models, model_class_name):
                    try:
                        model_class = getattr(models, model_class_name)
                        model = model_class.from_preset(MODEL_PRESET, load_weights=False)
                        break
                    except:
                        continue
            else:
                raise ValueError(f"No compatible model class for '{MODEL_PRESET}'")

        model.preprocessor.sequence_length = 128
        print(f"✅ Loaded '{MODEL_PRESET}' with sequence length {model.preprocessor.sequence_length}")


        model.summary()
        return model

    except Exception as e:
        print(f"❌ Failed to load '{MODEL_PRESET}': {e}")
        raise

def create_test_inputs(model):
    """Create test inputs for the model."""
    print("\n🎯 Creating test inputs...")

    # Instead of using preprocessor, create direct inputs that match model expectations
    batch_size = 1  # Use batch size 1 to match the exported model
    sequence_length = 128

    # Create random token IDs (use proper vocab range for Llama3)
    # Llama3 typically has vocab_size around 128,256, so use 1-32000 as safe range
    token_ids = tf.random.uniform(
        shape=(batch_size, sequence_length),
        minval=1,  # Avoid 0 which is usually padding
        maxval=32000,  # Use a reasonable vocab range for Llama3
        dtype=tf.int32
    )

    # Create padding mask (all True for simplicity - no padding)
    padding_mask = tf.ones_like(token_ids, dtype=tf.bool)

    # Create the input dictionary that matches model expectations
    test_inputs = {
        "token_ids": token_ids,
        "padding_mask": padding_mask
    }

    print(f"✅ Test inputs created:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Input keys: {list(test_inputs.keys())}")

    for key, value in test_inputs.items():
        print(f"   {key}: shape={value.shape}, dtype={value.dtype}")

    return test_inputs

def export_to_litert(model, model_name):
    """Export the model to LiteRT format."""
    print(f"\n🚀 Exporting '{model_name}' to LiteRT...")

    litert_model_path = f"{model_name}_test_model.tflite"

    # Check if export already exists
    if os.path.exists(litert_model_path):
        print(f"✅ Found existing LiteRT model: {litert_model_path}")
        return litert_model_path

    try:
        start_time = time.time()
        exported_path = model.export(litert_model_path, "lite_rt")
        end_time = time.time()

        export_time = end_time - start_time
        if os.path.exists(litert_model_path):
            file_size = os.path.getsize(litert_model_path)
            print(f"📊 LiteRT model size: {file_size / (1024*1024):.2f} MB")
            return litert_model_path
        print(f"✅ Export successful!")
        print(f"⏱️  Export time: {export_time:.2f} seconds")
        print(f"� Model size: {model_size_mb:.2f} MB")
        print(f"💾 Saved to: {exported_path}")

        return exported_path

    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise


def load_litert_interpreter(tflite_path):
    """Load the LiteRT interpreter."""
    print("\n🔧 Loading LiteRT interpreter...")

    if not LITERT_AVAILABLE:
        raise ImportError("LiteRT interpreter not available")

    try:
        interpreter = ai_edge_litert.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("✅ LiteRT interpreter loaded")
        print(f"📥 Input tensors: {len(input_details)}")
        for i, detail in enumerate(input_details):
            print(f"   Input {i}: {detail['name']} - {detail['shape']}, {detail['dtype']}")

        print(f"📤 Output tensors: {len(output_details)}")
        for i, detail in enumerate(output_details):
            print(f"   Output {i}: {detail['name']} - {detail['shape']}, {detail['dtype']}")

        return interpreter, input_details, output_details

    except Exception as e:
        print(f"❌ Failed to load LiteRT interpreter: {e}")
        raise


def run_keras_inference(model, inputs):
    """Run inference with the original Keras model."""
    print("\n🧠 Running Keras inference...")

    start_time = time.time()

    try:
        keras_outputs = model(inputs)
    except Exception as e:
        print(f"ℹ️  Dictionary input failed: {e}")
        try:
            keras_outputs = model(inputs["token_ids"], inputs["padding_mask"])
        except Exception as e2:
            print(f"❌ Both input methods failed:")
            print(f"   Dict method: {e}")
            print(f"   Positional method: {e2}")
            raise ValueError("Could not run Keras model inference")

    end_time = time.time()

    print(f"✅ Keras inference completed in {end_time - start_time:.4f} seconds")
    print(f"📊 Output: {keras_outputs.shape}, {keras_outputs.dtype}")

    return keras_outputs


def run_litert_inference(interpreter, input_details, output_details, inputs):
    """Run inference with the LiteRT interpreter."""
    print("\n⚡ Running LiteRT inference...")

    print(f"🔍 Available inputs: {list(inputs.keys())}")
    print(f"🔍 Expected inputs: {[detail['name'] for detail in input_details]}")

    # Set input tensors
    for i, input_detail in enumerate(input_details):
        input_name = input_detail['name']
        input_data = None

        print(f"🔍 Mapping input {i}: {input_name}")

        # Direct name matching
        for key in inputs.keys():
            if key.lower() in input_name.lower() or input_name.lower() in key.lower():
                input_data = inputs[key]
                print(f"   ✅ Mapped by name: {key} -> {input_name}")
                break

        # Pattern matching
        if input_data is None:
            if 'token' in input_name.lower() or 'input_1' == input_name or i == 0:
                if 'token_ids' in inputs:
                    input_data = inputs['token_ids']
                    print(f"   ✅ Mapped by pattern: token_ids -> {input_name}")
            elif 'mask' in input_name.lower() or 'input_2' == input_name or i == 1:
                if 'padding_mask' in inputs:
                    input_data = inputs['padding_mask']
                    print(f"   ✅ Mapped by pattern: padding_mask -> {input_name}")

        # By order
        if input_data is None:
            input_keys = list(inputs.keys())
            if i < len(input_keys):
                input_data = inputs[input_keys[i]]
                print(f"   ✅ Mapped by order: {input_keys[i]} -> {input_name}")

        if input_data is None:
            raise ValueError(f"Cannot map input: {input_name} (index {i})")

        # Ensure correct data type
        expected_dtype = input_detail['dtype']
        if hasattr(input_data, 'dtype') and input_data.dtype != expected_dtype:
            print(f"   🔄 Converting dtype: {input_data.dtype} -> {expected_dtype}")
            input_data = tf.cast(input_data, expected_dtype)

        # Convert to numpy
        if hasattr(input_data, 'numpy'):
            input_numpy = input_data.numpy()
        else:
            input_numpy = np.array(input_data)

        interpreter.set_tensor(input_detail['index'], input_numpy)
        print(f"   ✅ Set input {input_name}: {input_numpy.shape}, {input_numpy.dtype}")

    # Run inference
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    # Get outputs
    litert_outputs = []
    for output_detail in output_details:
        output = interpreter.get_tensor(output_detail['index'])
        litert_outputs.append(output)
        print(f"📤 Output {output_detail['name']}: {output.shape}, {output.dtype}")

    print(f"✅ LiteRT inference completed in {end_time - start_time:.4f} seconds")

    return litert_outputs[0] if len(litert_outputs) == 1 else litert_outputs


def compare_outputs(keras_output, litert_output, tolerance=1e-3):
    """Compare outputs from Keras and LiteRT models."""
    print("\n🔍 Comparing outputs...")

    # Convert to numpy arrays
    keras_np = keras_output.numpy() if hasattr(keras_output, 'numpy') else keras_output
    litert_np = litert_output if isinstance(litert_output, np.ndarray) else np.array(litert_output)

    print(f"📊 Keras output: {keras_np.shape}")
    print(f"📊 LiteRT output: {litert_np.shape}")

    # Check shapes match
    if keras_np.shape != litert_np.shape:
        print(f"❌ Shape mismatch: Keras {keras_np.shape} vs LiteRT {litert_np.shape}")
        return False

    # Calculate differences
    abs_diff = np.abs(keras_np - litert_np)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    rel_diff = np.abs(keras_np - litert_np) / (np.abs(keras_np) + 1e-8)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    print(f"📈 Max absolute difference: {max_abs_diff:.6f}")
    print(f"📈 Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"📈 Max relative difference: {max_rel_diff:.6f}")
    print(f"📈 Mean relative difference: {mean_rel_diff:.6f}")

    outputs_match = max_abs_diff < tolerance

    if outputs_match:
        print(f"✅ Outputs match within tolerance ({tolerance})")
    else:
        print(f"❌ Outputs differ by more than tolerance ({tolerance})")

        print("\n🔍 Sample comparisons:")
        flat_keras = keras_np.flatten()
        flat_litert = litert_np.flatten()

        for i in range(min(10, len(flat_keras))):
            diff = abs(flat_keras[i] - flat_litert[i])
            print(f"   Index {i}: Keras={flat_keras[i]:.6f}, LiteRT={flat_litert[i]:.6f}, diff={diff:.6f}")

    return outputs_match


def main():
    """Main test function."""
    print("🎯 Starting Keras Hub LiteRT Export Test")
    print("=" * 60)

    # Test basic functionality
    print("\n🔍 Testing basic functionality...")
    try:
        print(f"✅ TensorFlow version: {tf.__version__}")
        print(f"✅ Keras version: {keras.__version__}")
        print(f"✅ Keras Hub available: {hasattr(keras_hub, 'models')}")
        print(f"✅ LiteRT available: {LITERT_AVAILABLE}")

        # Test basic TF operations
        print("🧪 Testing basic TensorFlow operations...")
        x = tf.constant([1, 2, 3, 4])
        y = tf.square(x)
        print(f"   tf.square([1,2,3,4]) = {y.numpy()}")
        print("✅ Basic TensorFlow operations work")

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

    try:
        # Load model
        model = load_model()

        # Create test inputs
        test_inputs = create_test_inputs(model)

        # Test Keras inference
        print("\n🧪 Testing Keras inference before export...")
        keras_output = run_keras_inference(model, test_inputs)

        # Export to LiteRT
        model_name = type(model).__name__.lower().replace("causal", "").replace("lm", "")
        export_dir = f"{model_name}"
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, f"{model_name}_model")
        tflite_path = export_to_litert(model, export_path)

        # Load LiteRT interpreter
        interpreter, input_details, output_details = load_litert_interpreter(tflite_path)

        # Run LiteRT inference
        litert_output = run_litert_inference(interpreter, input_details, output_details, test_inputs)

        # Compare outputs
        outputs_match = compare_outputs(keras_output, litert_output)

        print("\n" + "=" * 60)
        if outputs_match:
            print("🎉 SUCCESS: Export test passed! Outputs match between Keras and LiteRT.")
        else:
            print("❌ FAILED: Export test failed! Outputs don't match.")
            print("ℹ️  This might be due to numerical precision differences.")
        print("=" * 60)
        print(f"\n💡 Models saved in:")
        print(f"   📁 TFLite: {export_dir}")


        return outputs_match

    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)