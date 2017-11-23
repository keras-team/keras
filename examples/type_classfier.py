from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.models import model_from_json
import numpy as np
# set parameters:
max_features = 5001
maxlen = 100
num_classes = 6
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10


x_train=np.loadtxt("x_train.txt",dtype=int)
y_train=np.loadtxt("y_train.txt",dtype=int)

indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

print('Loading data...')
#x_train=np.loadtxt("x_train.txt",dtype=int)
#y_train=np.loadtxt("y_train.txt",dtype=int)
x_test=x_train[20000:]
y_test=y_train[20000:]
x_train=x_train[:20000]
y_train=y_train[:20000]
#x_test=x_train
#y_test=y_train
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(x_train[:1])
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.5))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

model_json = model.to_json()
with open("type_predict.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("type_predict.h5")
json_file = open('type_predict.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("type_predict.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print(score)