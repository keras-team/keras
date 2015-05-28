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
    - __on_epoch_end__(epoch, val_loss, val_acc): Method called at the end of epoch `epoch`, with validation loss `val_loss` and accuracy `val_acc` (if applicable).
    - __on_batch_begin__(batch): Method called at the beginning of batch `batch`.
    - __on_batch_end__(batch, indices, loss, accuracy): Method called at the end of batch `batch`, with loss `loss` and accuracy `accuracy` (if applicable).


### Example

```python
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import History

model = Sequential()
model.add(Dense(784, 10, init='uniform'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = History()
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, callbacks=[history])

print history.losses
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''
```