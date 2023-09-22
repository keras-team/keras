import time

import keras


class BenchmarkMetricsCallback(keras.callbacks.Callback):
    def __init__(self, start_batch=1, stop_batch=None):
        self.start_batch = start_batch
        self.stop_batch = stop_batch

        # Store the throughput of each epoch.
        self.state = {"throughput": []}

    def on_train_batch_begin(self, batch, logs=None):
        if batch == self.start_batch:
            self.state["epoch_begin_time"] = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if batch == self.stop_batch:
            epoch_end_time = time.time()
            throughput = (self.stop_batch - self.start_batch + 1) / (
                epoch_end_time - self.state["epoch_begin_time"]
            )
            self.state["throughput"].append(throughput)
