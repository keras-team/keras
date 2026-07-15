import os

os.environ["KERAS_BACKEND"] = "openvino"

import numpy as np

import keras
import keras.ops as ops

print(f"Using Keras backend: {keras.backend.backend()}")

num_sequences = 4
real = np.ones((1, num_sequences, 1), dtype="float32")
imag = np.zeros((1, num_sequences, 1), dtype="float32")

res = ops.istft(
    (real, imag),
    sequence_length=1,
    sequence_stride=1,
    fft_length=1,
    center=True,
)

print("ISTFT Result shape:", res.shape)

if res.shape[-1] == 0:
    print("\n[BUG REPRODUCED]: Returned an empty tensor!")
    print("Expected non-empty reconstructed signal.")
else:
    print("Returned reconstructed signal of length:", res.shape[-1])
