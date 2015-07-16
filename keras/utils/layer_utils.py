import inspect
import numpy as np

from ..layers.advanced_activations import LeakyReLU, PReLU
from ..layers.core import Dense, Merge, Dropout, Activation, Reshape, Flatten, RepeatVector, Layer
from ..layers.core import ActivityRegularization, TimeDistributedDense, AutoEncoder, MaxoutDense
from ..layers.embeddings import Embedding, WordContextProduct
from ..layers.noise import GaussianNoise, GaussianDropout
from ..layers.normalization import BatchNormalization
from ..layers.recurrent import SimpleRNN, SimpleDeepRNN, GRU, LSTM, JZS1, JZS2, JZS3
from ..layers import containers

from .. import regularizers
from .. import constraints


def container_from_config(layer_dict):
    name = layer_dict.get('name')
    hasParams = False

    if name == 'Merge':
        mode = layer_dict.get('mode')
        layers = layer_dict.get('layers')
        layer_list = []
        for layer in layers:
            init_layer = container_from_config(layer)
            layer_list.append(init_layer)
        merge_layer = Merge(layer_list, mode)
        return merge_layer

    elif name == 'Sequential':
        layers = layer_dict.get('layers')
        layer_list = []
        for layer in layers:
            init_layer = container_from_config(layer)
            layer_list.append(init_layer)
        seq_layer = containers.Sequential(layer_list)
        return seq_layer

    elif name == 'Graph':
        graph_layer = containers.Graph()
        inputs = layer_dict.get('input_config')

        for input in inputs:
            graph_layer.add_input(**input)

        nodes = layer_dict.get('node_config')
        for node in nodes:
            layer = container_from_config(layer_dict['nodes'].get(node['name']))
            node['layer'] = layer
            graph_layer.add_node(**node)

        outputs = layer_dict.get('output_config')
        for output in outputs:
            graph_layer.add_output(**output)
        return graph_layer

    else: # The case in which layer_dict represents an "atomic" layer
        layer_dict.pop('name')
        if 'parameters' in layer_dict:
            params = layer_dict.get('parameters')
            layer_dict.pop('parameters')
            hasParams = True

        for k, v in layer_dict.items():
        	# For now, this can only happen for regularizers and constraints
            if isinstance(v, dict):
                vname = v.get('name')
                v.pop('name')
                if vname in [x for x,y in inspect.getmembers(constraints, predicate=inspect.isclass)]:
                	layer_dict[k] = constraints.get(vname, v)
                if vname in [x for x,y in inspect.getmembers(regularizers, predicate=inspect.isclass)]:
                	layer_dict[k] = regularizers.get(vname, v)
                
        base_layer = get_layer(name, layer_dict)
        if hasParams:
            shaped_params = []
            for param in params:
                data = np.asarray(param.get('data'))
                shape = tuple(param.get('shape'))
                shaped_params.append(data.reshape(shape))
            base_layer.set_weights(shaped_params)
        return base_layer


from .generic_utils import get_from_module
def get_layer(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'layer', instantiate=True, kwargs=kwargs)