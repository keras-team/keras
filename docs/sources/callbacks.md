## Usage of callbacks

A callback is a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training.

---

## Base class

```python
keras.callbacks.Callback()
```
- __Properties__:
    - __params__: dict. Training parameters (eg. verbosity, batch size, number of epochs...).
    - __model__: `keras.models.Model`. Reference of the model being trained.
- __Methods__:
    - __on_train_begin__(): Method called at the beginning of training.
    - __on_train_end__(): Method called at the end of training.
    - __on_epoch_begin__(epoch): Method called at the beginning of epoch `epoch`.
    - __on_epoch_end__(epoch): Method called at the end of epoch `epoch`.
    - __on_batch_begin__(batch): Method called at the beginning of batch `batch`.
    - __on_batch_end__(batch): Method called at the end of batch `batch`.

---


## Create a callback

You can create a custom callback by extending the base class `keras.callbacks.Callback`. A callback has access to its associated model through the class property `self.model`. Two properties of models will be of particular interest to callbacks: `self.model.epoch_history` and `self.model.batch_history`.

Here's a simple example saving a list of losses over each batch during training:
```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self):
        self.losses = []

    def on_batch_end(self, batch):
        self.losses.append(self.model.batch_history.loss[-1])
```

---

### Example

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self):
        self.losses = []

    def on_batch_end(self, batch):
        self.losses.append(self.model.batch_history.loss[-1])

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