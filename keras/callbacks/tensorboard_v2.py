"""TensorBoard callback for training visualization.

This is the TF v2 version. A lot of the functionality
from the v1 version isn't currently supported (but will
likely be added back later).

The docstring is left unchanged
to avoid creating confusion on the docs website.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import warnings


class TensorBoard(tf.keras.callbacks.TensorBoard):
    """TensorBoard basic visualizations.

    [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```

    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved. If set to 0, embeddings won't be computed.
            Data to be visualized in TensorBoard's Embedding tab must be passed
            as `embeddings_data`.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/guide/embedding#metadata)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
        embeddings_data: data to be embedded at layers specified in
            `embeddings_layer_names`. Numpy array (if the model has a single
            input) or list of Numpy arrays (if the model has multiple inputs).
            Learn [more about embeddings](
            https://www.tensorflow.org/guide/embedding).
        update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
            the losses and metrics to TensorBoard after each batch. The same
            applies for `'epoch'`. If using an integer, let's say `10000`,
            the callback will write the metrics and losses to TensorBoard every
            10000 samples. Note that writing too frequently to TensorBoard
            can slow down your training.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=None,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch',
                 **kwargs):
        if batch_size is not None:
            warnings.warn('The TensorBoard callback `batch_size` argument '
                          '(for histogram computation) '
                          'is deprecated with TensorFlow 2.0. '
                          'It will be ignored.')
        if write_grads:
            warnings.warn('The TensorBoard callback does not support '
                          'gradients display when using TensorFlow 2.0. '
                          'The `write_grads` argument is ignored.')
        if (embeddings_freq or embeddings_layer_names or
                embeddings_metadata or embeddings_data):
            warnings.warn('The TensorBoard callback does not support '
                          'embeddings display when using TensorFlow 2.0. '
                          'Embeddings-related arguments are ignored.')
        super(TensorBoard, self).__init__(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
            write_graph=write_graph,
            write_images=write_images,
            update_freq=update_freq,
            **kwargs)

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
        model.run_eagerly = False
        super(TensorBoard, self).set_model(model)
