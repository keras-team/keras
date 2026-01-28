import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import traceback
import sys
sys.path.insert(0, '/c/Users/watta/hive/keras')

import keras
import tensorflow as tf

try:
    ascan = tf.function(keras.ops.associative_scan)
    add = lambda x, y: x + y
    arr = keras.ops.array([1, 2, 3])
    result = ascan(add, arr, axis=0)
    print("Success! Result:", result)
except RecursionError as e:
    print("FAILED: RecursionError")
    traceback.print_exc()
except Exception as e:
    print("ERROR:", type(e).__name__)
    traceback.print_exc()
