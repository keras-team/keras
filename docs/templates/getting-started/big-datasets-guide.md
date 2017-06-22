# Getting started with large datasets

When using large datasets, they may be too large to fit into memory. When it's the case, you must load your samples just in time to be feed to your network. To maximize the efficiency of your pipeline, it is recommended to do the maximum of work before launching your training.
* Fetch a high-level representation of your dataset.
 * Paths to images, classes, coordinates for boxes, etc.
* Pre-process your dataset
 * Resize your images
 * Create a HDF5 file from it
 * Normalize your input

Doing this, will greatly improve your training time and will help maximize the utilization of your gpu.


## Using fit_generator
The main components to handle large datasets are the `*_generator` methods. Those methods can handle two types of iterators : [generators](#how-to-use-generators) and [Sequences](#how-to-use-sequences).

Both iterators use the same format for the batches. See [model.fit_generator](https://keras.io/models/model/) for more informations.

There is two modes of parallelism available : multi-threaded or multi-processed. This describe the type of workers that will be feeding a queue asynchronously. While processes are faster, they require more memory and may cause issues when used with generators. See [Issues with generators](#issues-with-generators). Also, remember that between epochs, the iterators are not deleted and will still consume memory while evaluating the validation set.

## How to use generators

Generators are really easy to use. If you wishes to learn more about them, here's a [guide](https://wiki.python.org/moin/Generators).

In Keras, generators should return a whole batch at each iteration. Additionally, generators must be infinite so they must cycle through the dataset indefinitely. If you have multiple inputs, outputs, you just need to create a list so if your generator returns `(X,y)`, `X` should be `[x1,...,xn]`.

It is the responsibility of the user to  ensure that their generators are thread-safe since they are not by default.

### Issues with generators
While generators are easy to use, they do not guarantee to keep the order of the samples. For example, the sample *B* could be feed into the queue before the sample *A*. This is problematic when using `predict_generator` because you wouldn't know the order of your samples.

Another issue is that when using multiprocessing, the data may get duplicated. In *Python*, generators cannot be shared between processes, they are copied instead. This means that your data will be seen multiple times through an epoch.


## How to use Sequences

Sequences are new as of Keras 2.0.5 and can be found in `keras.utils`. They have been introduced to resolve the issues with generators. While they are more complex to build, they **guarantee** the ordering of the samples at no cost.

A `Sequence` requires two methods :
```python
class Sequence(object):

    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batches in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError
```

When you've created an object `Sequence`, it may get directly used by `fit_generator`, `evaluate_generator` and `predict_generator`.

An example is available inside the `Sequence`'s documentation.

### Why Sequence requires batches

Sequences were designed to hold batches for the following reasons:
* Mimics the generators behavior
* Allows easier fully convolutional training
 * A batch needs to have the same dimensions
* The user can optimize his pipeline to handle batches

For large dataset, this should not affect your training.
