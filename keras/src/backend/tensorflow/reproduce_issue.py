import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import keras.src.ops
import numpy as np

print(f"Has keras.src.ops.allclose: {hasattr(keras.src.ops, 'allclose')}")

try:
    print(f"Has allclose in keras.ops: {hasattr(keras.ops, 'allclose')}")

    x1 = np.array([1.0, 2.0])
    x2 = np.array([1.0, 2.000000001])
    # Try calling directly from src if ops fails
    if hasattr(keras.src.ops, 'allclose'):
        print("Calling keras.src.ops.allclose...")
        result = keras.src.ops.allclose(x1, x2)
        print(f"Result: {result}")
    else:
        print("keras.src.ops.allclose missing")

except Exception as e:
    print(f"Error: {e}")