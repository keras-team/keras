import h5py
import numpy as np
import os
import struct
from collections import defaultdict

from keras import backend as K
from keras.models import Model, load_model

BACKEND = K.backend()
DIM_ORDERING = K.image_dim_ordering()

def load_model_if_exists(prefix=None, out_dir=None):
    '''
    Load a model from path out_dir/prefix_BACKEND_model.h5
    if it exists.

    :param prefix:
    :param out_dir:
    :return:
    '''
    if prefix is None:
        prefix = 'keras_model'
    if not prefix.endswith('_'):
        prefix += '_'
    if out_dir is None:
        out_dir = os.getcwd()
    model_path = os.path.join(out_dir, prefix + 'model.h5')
    if os.path.isfile(model_path):
        return load_model(model_path)
    return None


def save_model_details(model, prefix=None, out_dir=None):
    '''
    Save model configuration to out_dir/prefix_BACKEND_config.json,
    model weights to out_dir/prefix_BACKEND_weights.h5, and
    full model to out_dir/prefix_BACKEND_model.h5.

    :param model:
    :param prefix:
    :param out_dir:
    :return:
    '''
    if prefix is None:
        prefix = 'keras_model'
    if not prefix.endswith('_'):
        prefix += '_'
    if out_dir is None:
        out_dir = os.getcwd()
    with open(os.path.join(out_dir, prefix + 'config.json'), 'w') as f:
        f.write(model.to_json(indent=1) + '\n')
    # TODO: save out YAML as well -- seems to be broken currently
#    with open(os.path.join(out_dir, prefix + 'config.yaml'), 'w') as f:
#        f.write(model.to_yaml() + '\n')
    model.save(os.path.join(out_dir, prefix + 'model.h5'), True)
    model.save_weights(os.path.join(out_dir, prefix + 'weights.h5'), True)


def save_model_output(model, X, Y, nb_examples=None, prefix=None, out_dir=None):
    '''
    Save data necessary to fully test a loaded model to an HDF5 archive: sample
    data (X, Y), model predictions, scores in given metrics, and per-layer activations.

    :param model:
    :param X:
    :param Y:
    :param nb_examples:
    :param prefix:
    :param out_dir:
    :return:
    '''
    if prefix is None:
        prefix = 'keras_model'
    if not prefix.endswith('_'):
        prefix += '_'
    if out_dir is None:
        out_dir = os.getcwd()

    if nb_examples is not None:
        idx = np.random.choice(X.shape[0], nb_examples, replace=True)
        X = X[idx]
        Y = Y[idx]
        batch_size = nb_examples
    else:
        batch_size = 128

    print('saving inputs and ouputs')
    f = h5py.File(os.path.join(out_dir, prefix + 'inputs_and_outputs.h5'), 'w')
    f.create_dataset('X', X.shape, dtype='f', data=X)
    f.create_dataset('Y', Y.shape, dtype='f', data=Y)
    Yhat = model.predict(X, batch_size=batch_size)
    f.create_dataset('Yhat', Yhat.shape, dtype='f', data=Yhat)

    print('saving scores')
    Sgroup = f.create_group('scores')
    scores = model.evaluate(X, Y, verbose=0)
    if type(scores) is list:
        for score_name, score in zip(model.metrics_names, scores):
            Sgroup.create_dataset(score_name, (1,), dtype='f', data=np.array([score]))
    else:
        Sgroup.create_dataset(model.metrics_names[0], (1,), dtype='f', data=np.array([scores]))

    Agroup = f.create_group('activations')
    for layer in model.layers:
        print('saving activations for layer ' + layer.name)
        m = Model(input=model.inputs, output=layer.output)
        A = m.predict(X, batch_size)
        if len(A.shape) == 4 and DIM_ORDERING == 'tf':
            A = np.rollaxis(A, 3, 1)
        Agroup.create_dataset(layer.name, A.shape, dtype='f', data=A)

    f.close()
