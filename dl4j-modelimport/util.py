import h5py
import json
import numpy as np
import os

from keras import backend as K
from keras.models import Model, load_model

BACKEND = K.backend()
DIM_ORDERING = K.image_dim_ordering()

def load_model_if_exists(prefix=None, out_dir=None):
    '''
    Load a model from path out_dir/prefix_DIM_ORDERING_model.h5
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
    Save model configuration to out_dir/prefix_DIM_ORDERING_config.json,
    model weights to out_dir/prefix_DIM_ORDERING_weights.h5, and
    full model to out_dir/prefix_DIM_ORDERING_model.h5.

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


def save_model_output(model, inputs, outputs, nb_examples=None, prefix=None, out_dir=None):
    '''
    Save data necessary to fully test a loaded model to an HDF5 archive: sample
    data (inputs, outputs), model predictions, scores in given metrics, and per-layer activations.

    :param model:
    :param inputs:
    :param outputs:
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

    if type(inputs) is np.ndarray:
        inputs = {model.input_names[0]: inputs}
    elif type(inputs) is list or type(inputs) is set:
        inputs = dict(zip(model.input_names, inputs))
    elif type(inputs) is not dict:
        raise ValueError('Invalid container type ' + type(inputs) + ' for inputs')
    if type(outputs) is np.ndarray:
        outputs = {model.output_names[0]: outputs}
    elif type(outputs) is list or type(outputs) is set:
        outputs = dict(zip(model.output_names, outputs))
    elif type(outputs) is not dict:
        raise ValueError('Invalid container type ' + type(outputs) + ' for outputs')
    if nb_examples is not None:
        idx = np.random.choice(inputs.values()[0].shape[0], nb_examples, replace=False)
        inputs = dict([ (k, v[idx]) for (k, v) in inputs.iteritems() ])
        outputs = dict([ (k, v[idx]) for (k, v) in outputs.iteritems() ])
        batch_size = nb_examples
    else:
        batch_size = 128

    print('creating HDF5 archive')
    f = h5py.File(os.path.join(out_dir, prefix + 'inputs_and_outputs.h5'), 'w')

    print('saving inputs')
    attr = { 'inputs': [ nm for nm in model.input_names ]}
    f.attrs['inputs'] = json.dumps(attr)
    grp = f.create_group('inputs')
    for (k, v) in inputs.iteritems():
        grp.create_dataset(k, v.shape, dtype='f', data=v)

    print('saving outputs')
    attr = { 'outputs': [ nm for nm in model.output_names ]}
    f.attrs['outputs'] = json.dumps(attr)
    grp = f.create_group('outputs')
    for (k, v) in outputs.iteritems():
        grp.create_dataset(k, v.shape, dtype='f', data=v)

    print('saving predictions')
    predictions = model.predict([ inputs[nm] for nm in model.input_names ], batch_size=batch_size)
    if type(predictions) is np.ndarray:
        predictions = [ predictions ]
    grp = f.create_group('predictions')
    for (k, v) in zip(model.output_names, predictions):
        grp.create_dataset(k, v.shape, dtype='f', data=v)

    print('saving scores')
    scores = model.evaluate([ inputs[nm] for nm in model.input_names ],
                            [ outputs[nm] for nm in model.output_names ],
                            verbose=0)
    grp = f.create_group('scores')
    if type(scores) is list:
        grp.create_dataset(model.metrics_names[0], (1,), dtype='f', data=np.array([scores[0]]))
        score_map = dict([ (nm, grp.create_group(nm)) for nm in model.output_names ])
        for score_name, score in zip(model.metrics_names[1:], scores[1:]):
            for output_name in model.output_names:
                if output_name in score_name:
                    nm = score_name.replace(output_name + '_', '')
                    grp = score_map[output_name]
                    grp.create_dataset(nm, (1,), dtype='f', data=np.array([score]))
                    break
    else:
        grp.create_dataset(model.metrics_names[0], (1,), dtype='f', data=np.array([scores]))

    grp = f.create_group('activations')
    for layer in model.layers:
        print('saving activations for layer ' + layer.name)
        m = Model(input=model.inputs, output=layer.output)
        a = m.predict([ inputs[nm] for nm in model.input_names], batch_size)
        grp.create_dataset(layer.name, a.shape, dtype='f', data=a)

    f.close()
