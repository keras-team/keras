import os
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import TimeSeriesSequence

f = open("/home/srjoglekar246/weather.csv", encoding="ISO-8859-1")
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    try:
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values
    except:
        pass

targets = float_data[:, 1]

mean = float_data[:20000].mean(axis=0)
float_data -= mean
std = float_data[:20000].std(axis=0)
float_data /= std

lookback = 100
step = 2
delay = 10
batch_size = 32

train_gen = TimeSeriesSequence(float_data, targets,
        lookback=lookback, step=step, delay=delay, 
        min_index=0, max_index=15000,
        batch_size=batch_size)

val_gen = TimeSeriesSequence(float_data, targets,
        lookback=lookback, step=step, delay=delay,
        min_index=15001, max_index=20000,
        batch_size=batch_size)

val_steps = int((20000 - 15001 - lookback) // batch_size)

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
        steps_per_epoch=100,
        epochs=20)
        validation_data=val_gen,
        validation_steps=val_steps)

