## Usage of callbacks

A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. You can pass a list of callback (as the keyword argument `callbacks`) to the `.fit()` method of the `Sequential` model. The relevant methods of the callbacks will then be called at each stage of the training. 

---

## Base class

```python
keras.callbacks.Callback()
```
- __Properties__:
    - __params__: dict. Training parameters (eg. verbosity, batch size, number of epochs...).
    - __model__: `keras.models.Model`. Reference of the model being trained.
- __Methods__:
    - __on_train_begin__(logs={}): Method called at the beginning of training.
    - __on_train_end__(logs={}): Method called at the end of training.
    - __on_epoch_begin__(epoch, logs={}): Method called at the beginning of epoch `epoch`.
    - __on_epoch_end__(epoch, logs={}): Method called at the end of epoch `epoch`.
    - __on_batch_begin__(batch, logs={}): Method called at the beginning of batch `batch`.
    - __on_batch_end__(batch, logs={}): Method called at the end of batch `batch`.

The `logs` dictionary will contain keys for quantities relevant to the current batch or epoch. Currently, the `.fit()` method of the `Sequential` model class will include the following quantities in the `logs` that it passes to its callbacks:
- __on_epoch_end__: logs optionally include `val_loss` (if validation is enabled in `fit`), and `val_accuracy` (if validation and accuracy monitoring are enabled).
- __on_batch_begin__: logs include `size`, the number of samples in the current batch.
- __on_batch_end__: logs include `loss`, and optionally `accuracy` (if accuracy monitoring is enabled).

---

## Available callbacks

```python
keras.callbacks.ModelCheckpoint(filepath, verbose=0, save_best_only=False)
```

Save the model after every epoch. If `save_best_only=True`, the latest best model according to the validation loss will not be overwritten. 


```python
keras.callbacks.EarlyStopping(patience=0, verbose=0)
```

Stop training after no improvement of the validation loss is seen for `patience` epochs.

```python
keras.callbacks.Plotter(save_to_filepath=None, show_plot_window=True, linestyles=None, linestyles_first_epoch=None, show_regressions=True, poly_forward_perc=0.1, poly_backward_perc=0.2, poly_n_forward_min=10, poly_n_backward_min=20, poly_degree=1)
```

Creates a plot which shows the changes in loss and accuracy (for both training and validation datasets) over time.
The plot is updated at the end of each epoch. Setting the parameter `save_to_filepath` allows to save the generated plot. An older file with the same name will be overwritten. Some image viewers may support automatic reloading of that file upon any change (i.e. at the end of each epoch).
Setting `show_plot_window` to `False` will hide the plot window and only save to the file.

---


## Create a callback

You can create a custom callback by extending the base class `keras.callbacks.Callback`. A callback has access to its associated model through the class property `self.model`.

Here's a simple example saving a list of losses over each batch during training:
```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

---

### Example to record the loss history

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(784, 10, init='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, callbacks=[history])

print history.losses
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''
```

---

### Example to checkpoint models

```python
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(784, 10, init='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])

```

