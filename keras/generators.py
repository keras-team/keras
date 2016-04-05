from .engine.training import make_batches
import numpy as np


def time_delay_generator(x, y, delays, batch_size, weights=None, shuffle=True):
    '''A generator to make it easy to fit time-delay regression models,
    i.e. a model where the value of y depends on past values of x

    # Arguments
    x: input data, as a Numpy array
    y: targets, as a Numpy array.
    delays: number of time-steps to include in model
    weights: Numpy array of weights for the samples
    shuffle: Whether or not to shuffle the data (set True for training)

    # Example
    if X_train is (1000,200), Y_train is (1000,1)
    train_gen = time_delay_generator(X_train, Y_train, delays=10, batch_size=100)

    train_gen is a generator that gives:
    x_batch as size (100,10,200) since each of the 100 samples includes the input
    data at the current and nine previous time steps
    y_batch as size (100,1)
    w_batch as size (100,)

    '''
    index_array = np.arange(x.shape[0])
    tlist = [1, 0] + range(2, np.ndim(x) + 1)
    while 1:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(x.shape[0], batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            batch_ids = [np.maximum(0, batch_ids - d) for d in range(delays)]
            x_batch = x[batch_ids, :].transpose(tlist)
            y_batch = y[batch_ids[0], :]
            if weights is not None:
                w_batch = weights[batch_ids[0], :][:, 0]
            else:
                w_batch = np.ones(x_batch.shape[0])
            w_batch[batch_ids[0] < delays] = 0.
            yield (x_batch, y_batch, w_batch)
