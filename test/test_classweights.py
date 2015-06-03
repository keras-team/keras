from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge
from keras.utils import np_utils
import numpy as np

nb_classes = 10
batch_size = 128
nb_epoch = 15

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
y_train = y_train[:max_train_samples]
y_test = y_test[:max_test_samples]
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

def createMNISTModel():
    model = Sequential()
    model.add(Dense(784, 50))
    model.add(Activation('relu'))
    model.add(Dense(50, 10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

model_classweights_fit = createMNISTModel()
model_fit = createMNISTModel()

model_classweights_train = createMNISTModel()
model_train = createMNISTModel()

high_weight = 100
class_weight = {0:1,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:high_weight}

########################
#test different methods#
########################

# fit
model_classweights_fit.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=3, validation_data=(X_test, Y_test), class_weight=class_weight)
model_fit.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=3, validation_data=(X_test, Y_test))
# train
model_classweights_train.train(X_train, Y_train, class_weight=class_weight)
model_train.train(X_train, Y_train)

print('Classification accuracies on test set:')
for nb in range(nb_classes):
    testIdcs = np.where(y_test == np.array(nb))[0]
    X_temp = X_test[testIdcs, :]
    Y_temp = Y_test[testIdcs,:]
    # eval model which was trained with fit()
    score_cw = model_classweights_fit.evaluate(X_temp, Y_temp, show_accuracy=True, verbose=0)
    score = model_fit.evaluate(X_temp, Y_temp, show_accuracy=True, verbose=0)
    # eval model which was trained with train()
    score_cw_train = model_classweights_train.evaluate(X_temp, Y_temp, show_accuracy=True, verbose=0)
    score_train = model_train.evaluate(X_temp, Y_temp, show_accuracy=True, verbose=0)
    # print test accuracies for class weighted model vs. uniform weights
    print("Digit %d: class_weight = %d -> %.3f \t class_weight = %d -> %.3f" % (nb, class_weight[nb], score_cw[1], 1, score[1]))
    if class_weight[nb] == high_weight and (score_cw[1] <= score[1] or score_cw_train[1] <= score_train[1]):
        raise Exception('Class weights are not implemented correctly')


