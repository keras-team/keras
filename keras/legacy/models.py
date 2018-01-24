from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .layers import Merge


def needs_legacy_support(model):
    return isinstance(model.layers[0], Merge)


def legacy_sequential_layers(model):
    layers = []
    if model.layers:
        if isinstance(model.layers[0], Merge):
            merge = model.layers[0]
            for layer in merge.layers:
                if hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        if sublayer not in layers:
                            layers.append(sublayer)
                else:
                    if layer not in layers:
                        layers.append(layer)
        else:
            if model.layers[0] not in layers:
                layers.append(model.layers[0])
        for layer in model.layers[1:]:
            if layer not in layers:
                layers.append(layer)
    return layers
