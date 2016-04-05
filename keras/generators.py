from .engine.training import make_batches
import numpy as np


def time_delay_generator(x, y, w, delays, batch_size):
    index_array = np.arange(x.shape[0])
    tlist = [1, 0] + range(2, len(x.shape) + 1)
    while 1:
        np.random.shuffle(index_array)
        batches = make_batches(x.shape[0], batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            batch_ids = [np.maximum(0, batch_ids-d) for d in range(0, delays)]
            x_batch = x[batch_ids, :].transpose(tlist)
            y_batch = y[batch_ids, :]
            w_batch = w[batch_ids]
            yield (x_batch, y_batch, w_batch)