""" Utility class for accessing the first epoch time interval.
Credit:
Script modified from TensorFlow Benchmark repo:
https://github.com/tensorflow/benchmarks/blob/keras-benchmarks/scripts/keras_benchmarks/models/timehistory.py
"""
import keras
import time


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
