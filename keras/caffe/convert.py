import caffe_pb2 as caffe
import google.protobuf
from converters import model_from_config


def caffe_to_keras(prototext=None, caffemodel=None, phase='train'):
        '''
            prototext: model description file in caffe
            caffemodel: stored weights file
            phase: train or test
            Usage:
                model_data = caffe_to_keras(prototext='VGG16.prototxt',
                                            caffemodel='VGG16_700iter.caffemodel')
                graph = model_data.get('network') # loaded with with weights is caffemodel is provided, else randomly initialized
                inputs = model_data.get('inputs')
                outputs = model_data.get('outputs')
                weights = model_data.get('weights') # useful for embedding networks
        '''
        model_data = {}

        if prototext:
            config = caffe.NetParameter()
            google.protobuf.text_format.Merge(open(prototext).read(), config)
        elif caffemodel:
            config = caffe.NetParameter()
            config.MergeFromString(open(caffemodel, 'rb').read())

        if len(config.layers) != 0:
            # prototext V1
            layers = config.layers[:]
        elif len(config.layer) != 0:
            # prototext V2
            layers = config.layer[:]
        else:
            raise Exception('could not load any layers from prototext')

        network, inputs, outputs = model_from_config(layers,
                                                     0 if phase == 'train' else 1,
                                                     config.input_dim[1:])
        model_data['network'] = network
        model_data['inputs'] = inputs
        model_data['outputs'] = outputs
        return model_data
