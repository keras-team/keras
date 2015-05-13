from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np

X = np.array([
    [0.0, 0.1],
    [0.6, 0.8],
    [0.1, 0.2],
    [0.9, 0.8],
    [0.9, 0.7],
    [0.3, 0.0],
    [0.2, 0.2],
    [0.5, 0.9]
    ])

y = (X[:,0] + X[:,1] > 0.5).astype("int32")

model = Sequential()
model.add(Dense(2, 3, init='uniform', activation='tanh'))
model.add(Dense(3, 1, init='uniform', activation='sigmoid'))
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, class_mode="binary")
model.fit(X, y, validation_split=0.5, shuffle=True, nb_epoch=50, verbose=2)

print("y")
print(y)
print("yhat")
print(model.predict_classes(X))
