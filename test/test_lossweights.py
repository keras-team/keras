from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout
from keras.optimizers import SGD
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

############################
# categorical crossentropy #
############################

print("Testing fit methods with and without classweights")
# fit
model_classweights_fit.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=(X_test, Y_test), class_weight=class_weight)
model_fit.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=0, validation_data=(X_test, Y_test))
print("Testing train methods with and without classweights")
# train
model_classweights_train.train(X_train, Y_train, class_weight=class_weight)
model_train.train(X_train, Y_train)

print('MNIST Classification accuracies on test set for fitted models:')
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

####################################################
# test cases for all remaining objective functions #
####################################################
batch_size = 64
nb_epoch = 10

np.random.seed(1337) # for reproducibility

def generateData(n_samples, n_dim):
    A_feats = np.random.randn(n_samples, n_dim)
    B_feats = np.random.randn(n_samples, n_dim)
    A_label = np.zeros((n_samples,1))
    B_label = np.ones((n_samples,1))
    X = np.vstack((A_feats, B_feats))
    y = np.vstack((A_label, B_label)).squeeze()
    return X, y

n_dim = 100
X_train, y_train = generateData(1000, n_dim)
X_test, y_test = generateData(5000, n_dim)

def createModel(ls, n_dim, activation="sigmoid"):
    model = Sequential()
    model.add(Dense(n_dim, 50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, 1))
    model.add(Activation(activation))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=ls, optimizer=sgd, class_mode="binary")
    return model

verbosity = 0
cw = {0: 1.5, 1: 1}
# binary crossentropy
model = createModel('binary_crossentropy', n_dim)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=verbosity, validation_data=(X_test, y_test), class_weight=cw)
res = model.predict(X_test, verbose=verbosity).round()
neg_preds, pos_preds = (1.0*np.sum(res == 0)/res.shape[0], 1.0*np.sum(res == 1)/res.shape[0])
assert(neg_preds > pos_preds)
print("binary crossentropy: %0.2f VS %0.2f" % (neg_preds, pos_preds))

# MAE
model = createModel('mean_absolute_error', n_dim)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=verbosity, validation_data=(X_test, y_test), class_weight=cw)
res = model.predict(X_test, verbose=verbosity).round()
neg_preds, pos_preds = (1.0*np.sum(res == 0)/res.shape[0], 1.0*np.sum(res == 1)/res.shape[0])
assert(neg_preds > pos_preds)
print("MAE: %0.2f VS %0.2f" % (neg_preds, pos_preds))

# MSE
model = createModel('mean_squared_error', n_dim)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=verbosity, validation_data=(X_test, y_test), class_weight=cw)
res = model.predict(X_test, verbose=verbosity).round()
neg_preds, pos_preds = (1.0*np.sum(res == 0)/res.shape[0], 1.0*np.sum(res == 1)/res.shape[0])
assert(neg_preds > pos_preds)
print("MSE: %0.2f VS %0.2f" % (neg_preds, pos_preds))

# hinge losses, map labels
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
cw = {-1: 1.5, 1: 1}

# hinge
model = createModel('hinge', n_dim, "tanh")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=verbosity, validation_data=(X_test, y_test), class_weight=cw)
res = model.predict(X_test, verbose=verbosity)
res[res < 0] = -1
res[res >= 0] = 1
neg_preds, pos_preds = (1.0*np.sum(res == -1)/res.shape[0], 1.0*np.sum(res == 1)/res.shape[0])
assert(neg_preds > pos_preds)
print("hinge: %0.2f VS %0.2f" % (neg_preds, pos_preds))

# squared hinge
model = createModel('squared_hinge', n_dim, "tanh")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=verbosity, validation_data=(X_test, y_test), class_weight=cw)
res = model.predict(X_test, verbose=verbosity)
res[res < 0] = -1
res[res >= 0] = 1
neg_preds, pos_preds = (1.0*np.sum(res == -1)/res.shape[0], 1.0*np.sum(res == 1)/res.shape[0])
assert(neg_preds > pos_preds)
print("sqr hinge: %0.2f VS %0.2f" % (neg_preds, pos_preds))

