from __future__ import print_function
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.advanced_activations import HierarchicalSoftmax
import time

n_classes = 100000
n_examples = 1000
n_dimensions = 500
noise_features = np.random.random((n_examples, n_dimensions))

m_in = Input(shape = (n_dimensions,))
label_input = Input(shape = (1,), dtype = 'int32')
hsm = HierarchicalSoftmax(n_classes)([m_in, label_input])
hsm_model = Model(input = [m_in, label_input], output = hsm)
hsm_model.compile(loss = 'mie',
                  optimizer = 'adam')
start = time.time()
hsm_model.fit([noise_features, np.arange(n_examples)],
              np.zeros(noise_features.shape[0]), nb_epoch = 1)
print("HSM took: {}".format(time.time()-start))


normal = Dense(n_classes, activation = "softmax")(m_in)
normal_model = Model(input = m_in, output = normal)
normal_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
labels = np.hstack([np.eye(n_examples, dtype = np.int),
                    np.zeros([n_examples, n_classes-n_examples])])
start = time.time()
normal_model.fit(noise_features, labels, nb_epoch = 1)
print("SM took: {}".format(time.time()-start))
