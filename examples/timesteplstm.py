"""
This example shows how to use TimeStepLSTM in model.

This particular example does the exact same thing as an LSTM with "return_states = True" because
input to the RNN was not changed as the states were evaluated but hopefully this will give you an
idea of what you can do with this new Layer
"""

from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, TimeStepLSTM, LSTM, Masking, Reshape
from keras.layers.merge import concatenate
import keras.backend as K
import numpy as np

np.random.seed(111)

# create model
i1 = Input(shape=(10, 16))
lstm1 = TimeStepLSTM(32)
lstm2 = LSTM(32, return_sequences=True)(i1)
predict_layer = Dense(1, activation='sigmoid')

predictions = []
# evaluate states individually
for t in range(K.int_shape(i1)[1]):
    if t > 0:
        # i1 can be updated here before calling RNN at next timepoint
        ht = lstm1(i1, prev_state=ht[-2:], timepoint=t)
    else:
        ht = lstm1(i1, timepoint=t)

    # can act on the individual states here before continuing in RNN
    # in this example, prediction based on the current state was made
    prediction = predict_layer(ht[0])
    predictions.append(Reshape((1,1))(prediction))

out1 = concatenate(predictions, axis=1)
out2 = TimeDistributed(Dense(1, activation='sigmoid'))(lstm2)

model1 = Model(i1, out1)
model1.compile(optimizer='adam', loss='binary_crossentropy')
model2 = Model(i1, out2)
model2.compile(optimizer='adam', loss='binary_crossentropy')

# train and compare outputs from both LSTM networks
train1 = np.ones((3, 10, 16))
train2 = np.zeros((2, 10, 16))
labels1 = np.zeros((3, 10, 1))
labels2 = np.ones((2, 10, 1))

train_data = np.concatenate([train1, train2], axis=0)
train_labels = np.concatenate([labels1, labels2], axis=0)

test_data1 = np.ones((1, 10, 16))
test_data2 = np.zeros((1, 10, 16))

model1.fit(train_data, train_labels, epochs=1000, batch_size=1)
model2.fit(train_data, train_labels, epochs=1000, batch_size=1)

preds11 = model1.predict(test_data1)
preds12 = model1.predict(test_data2)
preds21 = model2.predict(test_data1)
preds22 = model2.predict(test_data2)

print('Preds11: ', preds11)
print('Preds12: ', preds12)
print('Preds21: ', preds21)
print('Preds22: ', preds22)
