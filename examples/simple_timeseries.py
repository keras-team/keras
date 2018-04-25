#
# TimeseriesGenerator demo
#

from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

# time
t = np.linspace(0, 20*np.pi, num=1000)
# input signal
x = np.sin(np.cos(3*t))
# output signal
y = np.sin(np.cos(6*t+4))

# define recurrent model
from keras.models import Model
from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense

inputs = Input(batch_shape=(None, None, 1))
l = SimpleRNN(100, return_sequences=True)(inputs)
l = Dense(100, activation='tanh')(l)
preds = Dense(1, activation='linear')(l)
model = Model(inputs=inputs, outputs=preds)
model.compile(loss='mean_squared_error', optimizer='Nadam')

# fit model to serie
xx = np.expand_dims(x, axis=-1)
g = TimeseriesGenerator(xx, y, hlength=100, target_seq=True, shuffle=True)
model.fit_generator(g, steps_per_epoch=len(g), epochs=20, shuffle=True)

# plot prediction
x2 = np.reshape(x, (1, x.shape[0], 1))
z = model.predict(x2)

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 12))
    plt.title('Phase representation')
    plt.plot(x, y.flatten(), color='black')
    plt.plot(x, z.flatten(), dashes=[8, 1], label='prediction', color='orange')
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid()
    plt.show()
except ImportError:
    print("install `mathplotlib` for graphical output.")
    pass
