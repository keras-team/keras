from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge
from keras.utils import np_utils
import numpy as np

nb_classes = 10
batch_size = 128
nb_epoch = 1

max_train_samples = 5000
max_test_samples = 1000

np.random.seed(1337) # for reproducibility

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000,784)[:max_train_samples]
X_test = X_test.reshape(10000,784)[:max_test_samples]
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)[:max_train_samples]
Y_test = np_utils.to_categorical(y_test, nb_classes)[:max_test_samples]

#########################
# sequential model test #
#########################
print('Test sequential')
model = Sequential()
model.add(Dense(784, 50))
model.add(Activation('relu'))
model.add(Dense(50, 10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=(X_test, Y_test))
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(X_test, Y_test))
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

score = model.evaluate(X_train, Y_train, verbose=0)
print('score:', score)
if score < 0.25:
    raise Exception('Score too low, learning issue.')
preds = model.predict(X_test, verbose=0)
classes = model.predict_classes(X_test, verbose=0)

model.get_config(verbose=1)

###################
# merge test: sum #
###################
print('Test merge: sum')
left = Sequential()
left.add(Dense(784, 50))
left.add(Activation('relu'))

right = Sequential()
right.add(Dense(784, 50))
right.add(Activation('relu'))

model = Sequential()
model.add(Merge([left, right], mode='sum'))

model.add(Dense(50, 10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test], Y_test))
model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test], Y_test))
model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

score = model.evaluate([X_train, X_train], Y_train, verbose=0)
print('score:', score)
if score < 0.22:
    raise Exception('Score too low, learning issue.')
preds = model.predict([X_test, X_test], verbose=0)
classes = model.predict_classes([X_test, X_test], verbose=0)

model.get_config(verbose=1)

###################
# merge test: concat #
###################
print('Test merge: concat')
left = Sequential()
left.add(Dense(784, 50))
left.add(Activation('relu'))

right = Sequential()
right.add(Dense(784, 50))
right.add(Activation('relu'))

model = Sequential()
model.add(Merge([left, right], mode='concat'))

model.add(Dense(50*2, 10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test], Y_test))
model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test], Y_test))
model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
model.fit([X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

score = model.evaluate([X_train, X_train], Y_train, verbose=0)
print('score:', score)
if score < 0.22:
    raise Exception('Score too low, learning issue.')
preds = model.predict([X_test, X_test], verbose=0)
classes = model.predict_classes([X_test, X_test], verbose=0)

model.get_config(verbose=1)

##########################
# test merge recursivity #
##########################
print('Test merge recursivity')

left = Sequential()
left.add(Dense(784, 50))
left.add(Activation('relu'))

right = Sequential()
right.add(Dense(784, 50))
right.add(Activation('relu'))

righter = Sequential()
righter.add(Dense(784, 50))
righter.add(Activation('relu'))

intermediate = Sequential()
intermediate.add(Merge([left, right], mode='sum'))
intermediate.add(Dense(50, 50))
intermediate.add(Activation('relu'))

model = Sequential()
model.add(Merge([intermediate, righter], mode='sum'))

model.add(Dense(50, 10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit([X_train, X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=([X_test, X_test, X_test], Y_test))
model.fit([X_train, X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=([X_test, X_test, X_test], Y_test))
model.fit([X_train, X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
model.fit([X_train, X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
model.fit([X_train, X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
model.fit([X_train, X_train, X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

score = model.evaluate([X_train, X_train, X_train], Y_train, verbose=0)
print('score:', score)
if score < 0.19:
    raise Exception('Score too low, learning issue.')
preds = model.predict([X_test, X_test, X_test], verbose=0)
classes = model.predict_classes([X_test, X_test, X_test], verbose=0)

model.get_config(verbose=1)

model.save_weights('temp.h5')
model.load_weights('temp.h5')

score = model.evaluate([X_train, X_train, X_train], Y_train, verbose=0)
print('score:', score)

######################
# test merge overlap #
######################
print('Test merge overlap')
left = Sequential()
left.add(Dense(784, 50))
left.add(Activation('relu'))

model = Sequential()
model.add(Merge([left, left], mode='sum'))

model.add(Dense(50, 10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=(X_test, Y_test))
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(X_test, Y_test))
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_split=0.1)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_split=0.1)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)

score = model.evaluate(X_train, Y_train, verbose=0)
print('score:', score)
if score < 0.22:
    raise Exception('Score too low, learning issue.')
preds = model.predict(X_test, verbose=0)
classes = model.predict_classes(X_test, verbose=0)

model.get_config(verbose=1)
