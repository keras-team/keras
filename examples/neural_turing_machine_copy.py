from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt

from theano import function

from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.utils import generic_utils

from keras.layers.ntm import NeuralTuringMachine as NTM

"""
Copy Problem defined in Graves et. al [0]

Training data is made of sequences with length 1 to 20. 
Test data are sequences of length 100.
The model is tested every 500 weight updates.
After about 3500 updates, the accuracy jumps from around 50% to >90%.

Estimated compile time: 12 min
Estimated time to train Neural Turing Machine and 3 layer LSTM on an NVidia GTX 680: 2h

[0]: http://arxiv.org/pdf/1410.5401v2.pdf
"""

batch_size = 100

h_dim = 128
n_slots = 128
m_length = 20
input_dim = 8
lr = 1e-3
clipvalue = 10

##### Neural Turing Machine ######

ntm = NTM(h_dim, n_slots=n_slots, m_length=m_length, shift_range=3,
          inner_rnn='lstm', return_sequences=True, input_dim=input_dim)
model = Sequential()
model.add(ntm)
model.add(TimeDistributedDense(input_dim))
model.add(Activation('sigmoid'))

sgd = Adam(lr=lr, clipvalue=clipvalue)
model.compile(loss='binary_crossentropy', optimizer=sgd)

# LSTM - Run this for comparison

sgd2 = Adam(lr=lr, clipvalue=clipvalue)
lstm = Sequential()
lstm.add(LSTM(input_dim=input_dim, output_dim=h_dim*2, return_sequences=True))
lstm.add(LSTM(output_dim=h_dim*2, return_sequences=True))
lstm.add(LSTM(output_dim=h_dim*2, return_sequences=True))
lstm.add(TimeDistributedDense(input_dim))
lstm.add(Activation('sigmoid'))

lstm.compile(loss='binary_crossentropy', optimizer=sgd)

###### DATASET ########

def get_sample(batch_size=128, n_bits=8, max_size=20, min_size=1):
    # generate samples with random length
    inp = np.zeros((batch_size, 2*max_size-1, n_bits))
    out = np.zeros((batch_size, 2*max_size-1, n_bits))
    sw = np.zeros((batch_size, 2*max_size-1, 1))
    for i in range(batch_size):
        t = np.random.randint(low=min_size, high=max_size)
        x = np.random.uniform(size=(t, n_bits)) > .5
        for j,f in enumerate(x.sum(axis=-1)): # remove fake flags
            if f>=n_bits:
                x[j, :] = 0.
        del_flag = np.ones((1, n_bits))
        inp[i, :t+1] = np.concatenate([x, del_flag], axis=0)
        out[i, t+1:(2*t+1)] = x
        sw[i, t+1:(2*t+1)] = 1
    return inp, out, sw

def show_pattern(inp, out, sw, file_name='ntm_output.png'):
    ''' Helper function to visualize results '''
    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(inp>.5)
    plt.subplot(132)
    plt.imshow(out>.5)
    plt.subplot(133)
    plt.imshow(sw>.5)
    plt.savefig(file_name)
    plt.close()

# Show data example:
inp, out, sw = get_sample(1, 8, 20)

plt.subplot(131)
plt.title('input')
plt.imshow(inp[0], cmap='gray')
plt.subplot(132)
plt.title('desired')
plt.imshow(out[0], cmap='gray')
plt.subplot(133)
plt.title('sample_weight')
plt.imshow(sw[0], cmap='gray')

# training uses sequences of length 1 to 20. Test uses series of length 100.
def test_model(model, file_name, min_size=100):
    I, V, sw = get_sample(batch_size=500, n_bits=input_dim, max_size=min_size+1, min_size=min_size)
    Y = np.asarray(model.predict(I, batch_size=100) > .5).astype('float64')
    acc = (V[:, -min_size:, :] == Y[:, -min_size:, :]).mean() * 100
    show_pattern(Y[0], V[0], sw[0], file_name)
    return acc

##### TRAIN ######
nb_epoch = 4000
progbar = generic_utils.Progbar(nb_epoch)
for e in range(nb_epoch):
    I, V, sw = get_sample(n_bits=input_dim, max_size=20, min_size=1, batch_size=100)

    loss1 = model.train_on_batch(I, V, sample_weight=sw)
    loss2 = lstm.train_on_batch(I, V, sample_weight=sw)

    progbar.add(1, values=[("NTM", loss1), ("LSTM", loss2)])

    if e % 500 == 0:
        print("")
        acc1 = test_model(model, 'ntm.png')
        acc2 = test_model(lstm, 'lstm.png')
        print("NTM  test acc: {}".format(acc1))
        print("LSTM test acc: {}".format(acc2))

##### VISUALIZATION #####
X = model.get_input()
Y = ntm.get_full_output()[0:3]  # (memory over time, read_vectors, write_vectors)
F = function([X], Y, allow_input_downcast=True)

inp, out, sw = get_sample(1, 8, 21, 20)
mem, read, write = F(inp.astype('float32'))
Y = model.predict(inp)

plt.figure(figsize=(15, 12))

plt.subplot(221)
plt.imshow(write[0])
plt.xlabel('memory location')
plt.ylabel('time')
plt.title('write')

plt.subplot(222)
plt.imshow(read[0])
plt.title('read')

plt.subplot(223)
plt.title('desired')
plt.imshow(out[0])

plt.subplot(224)
plt.imshow(Y[0]>.5)
plt.title('output')

plt.figure(figsize=(15, 10))
plt.subplot(325)
plt.ylabel('time')
plt.xlabel('location')
plt.title('memory evolving in time (avg value per location)')
plt.imshow(mem[0].mean(axis=-1))
