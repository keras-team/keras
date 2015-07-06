## Sequential

Linear stack of layers.

```python
model = keras.models.Sequential()
```
- __Methods__:
    - __add__(layer): Add a layer to the model.
    - __compile__(optimizer, loss, class_mode="categorical"):
        - __Arguments__:
            - __optimizer__: str (name of optimizer) or optimizer object. See [optimizers](optimizers.md).
            - __loss__: str (name of objective function) or objective function. See [objectives](objectives.md).
            - __class_mode__: one of "categorical", "binary". This is only used for computing classification accuracy or using the predict_classes method.
            - __theano_mode__: A `theano.compile.mode.Mode` ([reference](http://deeplearning.net/software/theano/library/compile/mode.html)) instance controlling specifying compilation options.
    - __fit__(X, y, batch_size=128, nb_epoch=100, verbose=1, validation_split=0., validation_data=None, shuffle=True, show_accuracy=False, callbacks=[], class_weight=None, sample_weight=None): Train a model for a fixed number of epochs.
        - __Return__: a history dictionary with a record of training loss values at successive epochs, as well as validation loss values (if applicable), accuracy (if applicable), etc.
        - __Arguments__:
            - __X__: data.
            - __y__: labels.
            - __batch_size__: int. Number of samples per gradient update.
            - __nb_epoch__: int.
            - __verbose__: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
            - __callbacks__: `keras.callbacks.Callback` list. List of callbacks to apply during training. See [callbacks](callbacks.md).
            - __validation_split__: float (0. < x < 1). Fraction of the data to use as held-out validation data.
            - __validation_data__: tuple (X, y) to be used as held-out validation data. Will override validation_split.
            - __shuffle__: boolean. Whether to shuffle the samples at each epoch.
            - __show_accuracy__: boolean. Whether to display class accuracy in the logs to stdout at each epoch.
            - __class_weight__: dictionary mapping classes to a weight value, used for scaling the loss function (during training only).
            - __sample_weight__: list or numpy array with 1:1 mapping to the training samples, used for scaling the loss function (during training only).
    - __evaluate__(X, y, batch_size=128, show_accuracy=False, verbose=1): Show performance of the model over some validation data.
        - __Return__: The loss score over the data, or a `(loss, accuracy)` tuple if `show_accuracy=True`.
        - __Arguments__: Same meaning as fit method above. verbose is used as a binary flag (progress bar or nothing).
    - __predict__(X, batch_size=128, verbose=1):
        - __Return__: An array of predictions for some test data.
        - __Arguments__: Same meaning as fit method above.
    - __predict_classes__(X, batch_size=128, verbose=1): Return an array of class predictions for some test data.
        - __Return__: An array of labels for some test data.
        - __Arguments__: Same meaning as fit method above. verbose is used as a binary flag (progress bar or nothing).
    - __train_on_batch__(X, y, accuracy=False): Single gradient update on one batch.
        - __Return__: loss over the data, or tuple `(loss, accuracy)` if `accuracy=True`.
    - __test_on_batch__(X, y, accuracy=False): Single performance evaluation on one batch.
        - __Return__: loss over the data, or tuple `(loss, accuracy)` if `accuracy=True`.
    - __save_weights__(fname, overwrite=False): Store the weights of all layers to a HDF5 file. If overwrite==False and the file already exists, an exception will be thrown.
    - __load_weights__(fname): Sets the weights of a model, based to weights stored by __save_weights__. You can only __load_weights__ on a savefile from a model with an identical architecture. __load_weights__ can be called either before or after the __compile__ step.

__Examples__:

```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(64, 2, init='uniform'))
model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='sgd')

'''
Demonstration of verbose modes 1 and 2
'''
model.fit(X_train, y_train, nb_epoch=3, batch_size=16, verbose=1)
# outputs
'''
Train on 37800 samples, validate on 4200 samples
Epoch 0
37800/37800 [==============================] - 7s - loss: 0.0385
Epoch 1
37800/37800 [==============================] - 8s - loss: 0.0140
Epoch 2
10960/37800 [=======>......................] - ETA: 4s - loss: 0.0109
'''

model.fit(X_train, y_train, nb_epoch=3, batch_size=16, verbose=2)
# outputs
'''
Train on 37800 samples, validate on 4200 samples
Epoch 0
loss: 0.0190
Epoch 1
loss: 0.0146
Epoch 2
loss: 0.0049
'''

'''
Demonstration of show_accuracy
'''
model.fit(X_train, y_train, nb_epoch=3, batch_size=16, verbose=2, show_accuracy=True)
# outputs
'''
Train on 37800 samples, validate on 4200 samples
Epoch 0
loss: 0.0190 - acc.: 0.8750
Epoch 1
loss: 0.0146 - acc.: 0.8750
Epoch 2
loss: 0.0049 - acc.: 1.0000
'''

'''
Demonstration of validation_split
'''
model.fit(X_train, y_train, nb_epoch=3, batch_size=16, validation_split=0.1, show_accuracy=True, verbose=1)
# outputs
'''
Train on 37800 samples, validate on 4200 samples
Epoch 0
37800/37800 [==============================] - 7s - loss: 0.0385 - acc.: 0.7258 - val. loss: 0.0160 - val. acc.: 0.9136
Epoch 1
37800/37800 [==============================] - 8s - loss: 0.0140 - acc.: 0.9265 - val. loss: 0.0109 - val. acc.: 0.9383
Epoch 2
10960/37800 [=======>......................] - ETA: 4s - loss: 0.0109 - acc.: 0.9420
'''
```

---

## Graph

Arbitrary connection graph. It can have any number of inputs and outputs, with each output trained with its own loss function. The quantity being optimized by a Graph model is the sum of all loss functions over the different outputs.

```python
model = keras.models.Graph()
```
- __Methods__:
    - __add_input__(name, ndim=2, dtype='float'): Add an input with shape dimensionality `ndim`. 
        - __Arguments__:
            - __ndim__: Use `ndim=2` for vector input `(samples, features)`, ndim=3 for temporal input `(samples, time, features)`, ndim=4 for image input `(samples, channels, height, width)`.
            - __dtype__: `float` or `int`. Use `int` if the input is connected to an Embedding layer, `float` otherwise.
    - __add_output__(name, input=None, inputs=[], merge_mode='concat'): Add an output connect to `input` or `inputs`.
        - __Arguments__:
            - __name__: str. unique identifier of the output.
            - __input__: str name of the node that the output is connected to. Only specify *one* of either `input` or `inputs`.
            - __inputs__: list of str names of the node that the output is connected to.
            - __merge_mode__: "sum" or "concat". Only applicable if `inputs` list is specified. Merge mode for the different inputs.
    - __add_node__(layer, name, input=None, inputs=[], merge_mode='concat'): Add an output connect to `input` or `inputs`.
        - __Arguments__:
            - __layer__: Layer instance.
            - __name__: str. unique identifier of the node.
            - __input__: str name of the node/input that the node is connected to. Only specify *one* of either `input` or `inputs`.
            - __inputs__: list of str names of the node that the node is connected to.
            - __merge_mode__: "sum" or "concat". Only applicable if `inputs` list is specified. Merge mode for the different inputs.
    - __compile__(optimizer, loss):
        - __Arguments__:
            - __optimizer__: str (name of optimizer) or optimizer object. See [optimizers](optimizers.md).
            - __loss__: dictionary mapping the name(s) of the output(s) to a loss function (string name of objective function or objective function. See [objectives](objectives.md)).
    - __fit__(data, batch_size=128, nb_epoch=100, verbose=1, validation_split=0., validation_data=None, shuffle=True, callbacks=[]): Train a model for a fixed number of epochs.
        - __Return__: a history dictionary with a record of training loss values at successive epochs, as well as validation loss values (if applicable).
        - __Arguments__:
            - __data__:dictionary mapping input names out outputs names to appropriate numpy arrays. All arrays should contain the same number of samples.
            - __batch_size__: int. Number of samples per gradient update.
            - __nb_epoch__: int.
            - __verbose__: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
            - __callbacks__: `keras.callbacks.Callback` list. List of callbacks to apply during training. See [callbacks](callbacks.md).
            - __validation_split__: float (0. < x < 1). Fraction of the data to use as held-out validation data.
            - __validation_data__: tuple (X, y) to be used as held-out validation data. Will override validation_split.
            - __shuffle__: boolean. Whether to shuffle the samples at each epoch.
    - __evaluate__(data, batch_size=128, verbose=1): Show performance of the model over some validation data.
        - __Return__: The loss score over the data.
        - __Arguments__: Same meaning as fit method above. verbose is used as a binary flag (progress bar or nothing).
    - __predict__(data, batch_size=128, verbose=1):
        - __Return__: A dictionary mapping output names to arrays of predictions over the data.
        - __Arguments__: Same meaning as fit method above. Only inputs need to be specified in `data`.
    - __train_on_batch__(data): Single gradient update on one batch.
        - __Return__: loss over the data.
    - __test_on_batch__(data): Single performance evaluation on one batch.
        - __Return__: loss over the data.
    - __save_weights__(fname, overwrite=False): Store the weights of all layers to a HDF5 file. If `overwrite==False` and the file already exists, an exception will be thrown.
    - __load_weights__(fname): Sets the weights of a model, based to weights stored by __save_weights__. You can only __load_weights__ on a savefile from a model with an identical architecture. __load_weights__ can be called either before or after the __compile__ step.


__Examples__:

```python
# graph model with one input and two outputs
graph = Graph()
graph.add_input(name='input', ndim=2)
graph.add_node(Dense(32, 16), name='dense1', input='input')
graph.add_node(Dense(32, 4), name='dense2', input='input')
graph.add_node(Dense(16, 4), name='dense3', input='dense1')
graph.add_output(name='output1', input='dense2')
graph.add_output(name='output2', input='dense3')

graph.compile('rmsprop', {'output1':'mse', 'output2':'mse'})
history = graph.fit({'input':X_train, 'output1':y_train, 'output2':y2_train}, nb_epoch=10)

```

```python
# graph model with two inputs and one output
graph = Graph()
graph.add_input(name='input1', ndim=2)
graph.add_input(name='input2', ndim=2)
graph.add_node(Dense(32, 16), name='dense1', input='input1')
graph.add_node(Dense(32, 4), name='dense2', input='input2')
graph.add_node(Dense(16, 4), name='dense3', input='dense1')
graph.add_output(name='output', inputs=['dense2', 'dense3'], merge_mode='sum')
graph.compile('rmsprop', {'output':'mse'})

history = graph.fit({'input1':X_train, 'input2':X2_train, 'output':y_train}, nb_epoch=10)
predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}

```