## Usage of callbacks

A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training. You can pass a list of callback (as the keyword argument `callbacks`) to the `.fit()` method of the `Sequential` model. The relevant methods of the callbacks will then be called at each stage of the training. 

---

## Base class

```python
keras.callbacks.Callback()
```
- __Properties__:
    - __params__: dict. Training parameters (eg. verbosity, batch size, number of epochs...).
- __Methods__:
    - __on_train_begin__(data): Method called at the beginning of training.
    - __on_train_end__(data): Method called at the end of training.
    - __on_epoch_begin__(data): Method called at the beginning of epoch `epoch`.
    - __on_epoch_end__(data): Method called at the end of epoch `epoch`.
    - __on_batch_begin__(data): Method called at the beginning of batch `batch`.
    - __on_batch_end__(data): Method called at the end of batch `batch`.

The `data` instances are all of subtypes of `Message` which contain the quantities relevant to the current batch or epoch.

---


## Create a callback

You can create a custom callback by extending the base class `keras.callbacks.Callback`. A callback has access to its associated model through the class property `self.model`.

---

### Example to record the loss history

```python
from keras.messages import *

class LossHistory(keras.callbacks.Callback):
    def __init__(self, stream=None):
        super(LossHistory, self).__init__()
        if stream:
            self.set_event_source(stream)

    def set_event_source(self, stream):
        stream.filter(lambda x: isinstance(x, TrainBegin)).subscribe(lambda x: self.on_train_begin(x))
        stream.filter(lambda x: isinstance(x, BatchEnd)).subscribe(lambda x: self.on_batch_end(x))

    def on_train_begin(self):
        self.losses = []

    def on_batch_end(self, data):
        self.losses.append(data.loss)

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

