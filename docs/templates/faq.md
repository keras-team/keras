# Keras FAQ: Frequently Asked Keras Questions

[How can I run Keras on GPU?](#how-can-i-run-keras-on-gpu)

[How can I save a Keras model?](#how-can-i-save-a-keras-model)

[Why is the training loss much higher than the testing loss?](#why-is-the-training-loss-much-higher-than-the-testing-loss)

[How can I visualize the output of an intermediate layer?](#how-can-i-visualize-the-output-of-an-intermediate-layer)

[Isn't there a bug with Merge or Graph related to input concatenation?](#isnt-there-a-bug-with-merge-or-graph-related-to-input-concatenation)

[How can I use Keras with datasets that don't fit in memory?](#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)

[How can I interrupt training when the validation loss isn't decreasing anymore?](#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore)

[How is the validation split computed?](#how-is-the-validation-split-computed)

[Is the data shuffled during training?](#is-the-data-shuffled-during-training)

[How can I record the training / validation loss / accuracy at each epoch?](#how-can-i-record-the-training-validation-loss-accuracy-at-each-epoch)

[How can I use stateful RNNs?](#how-can-i-use-stateful-rnns)

---

### How can I run Keras on GPU?

Method 1: use Theano flags.
```bash
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

The name 'gpu' might have to be changed depending on your device's identifier (e.g. `gpu0`, `gpu1`, etc).

Method 2: set up your `.theanorc`: [Instructions](http://deeplearning.net/software/theano/library/config.html)

Method 3: manually set `theano.config.device`, `theano.config.floatX` at the beginning of your code:
```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

---

### How can I save a Keras model?

*It is not recommended to use pickle or cPickle to save a Keras model.*

If you only need to save the architecture of a model, and not its weights, you can do:

```python
# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
```

You can then build a fresh model from this data:

```python
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML
model = model_from_yaml(yaml_string)
```

If you need to save the weights of a model, you can do so in HDF5:
```python
model.save_weights('my_model_weights.h5')
```

Assuming you have code for instantiating your model, you can then load the weights you saved into a model with the same architecture:

```python
model.load_weights('my_model_weights.h5')
```

This leads us to a way to save and reconstruct models from only serialized data:
```python
json_string = model.to_json()
open('my_model_architecture.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5')

# elsewhere...
model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')
```

---

### Why is the training loss much higher than the testing loss?

A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing time.

Besides, the training loss is the average of the losses over each batch of training data. Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches. On the other hand, the testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss.

---

### How can I visualize the output of an intermediate layer?

You can build a Theano function that will return the output of a certain layer given a certain input, for example:

```python
# with a Sequential model
get_3rd_layer_output = theano.function([model.layers[0].input],
                                       model.layers[3].get_output(train=False))
layer_output = get_3rd_layer_output(X)

# with a Graph model
get_conv_layer_output = theano.function([model.inputs[i].input for i in model.input_order],
                                        model.nodes['conv'].get_output(train=False),
                                        on_unused_input='ignore')
conv_output = get_conv_layer_output([input_data_dict[i] for i in model.input_order])
```

---

### Isn't there a bug with Merge or Graph related to input concatenation?

Yes, there was a known bug with tensor concatenation in Theano that was fixed early 2015.
Please upgrade to the latest version of Theano:

```bash
sudo pip install git+git://github.com/Theano/Theano.git
```

---

### How can I use Keras with datasets that don't fit in memory?

You can do batch training using `model.train_on_batch(X, y)` and `model.test_on_batch(X, y)`. See the [models documentation](models.md).

You can also see batch training in action in our [CIFAR10 example](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py).

---

### How can I interrupt training when the validation loss isn't decreasing anymore?

You can use an `EarlyStopping` callback:

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
```

Find out more in the [callbacks documentation](callbacks.md).

---

### How is the validation split computed?

If you set the `validation_split` argument in `model.fit` to e.g. 0.1, then the validation data used will be the *last 10%* of the data. If you set it to 0.25, it will be the last 25% of the data, etc.


---

### Is the data shuffled during training?

Yes, if the `shuffle` argument in `model.fit` is set to `True` (which is the default), the training data will be randomly shuffled at each epoch.

Validation data isn't shuffled.

---


### How can I record the training / validation loss / accuracy at each epoch?

The `model.fit` method returns an `History` callback, which has a `history` attribute containing the lists of successive losses / accuracies.

```python
hist = model.fit(X, y, validation_split=0.2)
print(hist.history)
```

---

### How can I use stateful RNNs?

Making a RNN stateful means that the states for the samples of each batch will be reused as initial states for the samples in the next batch.

When using stateful RNNs, it is therefore assumed that:

- all batches have the same number of samples
- If `X1` and `X2` are successive batches of samples, then `X2[i]` is the follow-up sequence to `X1[i]`, for every `i`.

To use statefulness in RNNs, you need to:

- explicitly specify the batch size you are using, by passing a `batch_input_shape` argument to the first layer in your model. It should be a tuple of integers, e.g. `(32, 10, 16)` for a 32-samples batch of sequences of 10 timesteps with 16 features per timestep.
- set `stateful=True` in your RNN layer(s).

To reset the states accumulated:

- use `model.reset_states()` to reset the states of all layers in the model
- use `layer.reset_states()` to reset the states of a specific stateful RNN layer

Example:

```python

X  # this is our input data, of shape (32, 21, 16)
# we will feed it to our model in sequences of length 10

model = Sequential()
model.add(LSTM(32, batch_input_shape=(32, 10, 16), stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(X[:, :10, :], np.reshape(X[:, 10, :], (32, 16)))

# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(X[:, 10:20, :], np.reshape(X[:, 20, :], (32, 16)))

# let's reset the states of the LSTM layer:
model.reset_states()

# another way to do it in this case:
model.layers[0].reset_states()
```

Notes that the methods `predict`, `fit`, `train_on_batch`, `predict_classes`, etc. will *all* update the states of the stateful layers in a model. This allows you to do not only stateful training, but also stateful prediction.

