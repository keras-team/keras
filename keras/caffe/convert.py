import caffe_pb2 as caffe
import google.protobuf
from converters import model_from_config, model_from_param, convert_weights, convert_solver, convert_meanfile

class CaffeToKeras(object):
	def __init__(self, prototext=None, caffemodel=None, solver=None, mean=None, phase='train'):
		'''
			prototext: model description file in caffe
			caffemodel: stored weights file
			solver: solver file
			mean: mean data file
			phase: train or test

			Usage:
				model = CaffeToKeras(prototext='VGG16.prototxt', caffemodel='VGG16_700iter.caffemodel')
				graph = model('network') # loaded with with weights is caffemodel is provided, else randomly initialized
				inputs = model('inputs')
				outputs = model('outputs')
				weights = model('weights') # useful for embedding networks
		'''
		if phase == 'train':
			self.phase = 0
		else:
			self.phase = 1

		if prototext is not None:
			config = caffe.NetParameter()
			google.protobuf.text_format.Merge(open(prototext).read(), config)

			self.input_dim = config.input_dim
			self.config_layers = config.layers[:]
			self.network, self.inputs, self.outputs = model_from_config(self.config_layers, self.phase, self.input_dim[1:])
		
		if caffemodel is not None:
			param = caffe.NetParameter()
			param.MergeFromString(open(caffemodel,'rb').read())
			self.param_layers = param.layers[:]
			
		if hasattr(self, 'network'):
			# network already created with prototext
			if caffemodel is not None:
				# see if weights have to be loaded 
				self.weights = convert_weights(self.param_layers)
				for layer_weights in self.weights:
					self.network.nodes[layer_weights] = self.weights[layer_weights]
		else:
			self.network, self.inputs, self.outputs = model_from_param(self.param_layers)

		if solver is not None:
			# TODO
			self.solver = convert_solver(solver)

		if mean is not None:
			# TODO
			self.mean = convert_meanfile(mean)

	def __call__(self, item):
		if item == 'network':
			return self.network
		elif item == 'solver':
			return self.solver
		elif item == 'mean':
			return self.mean
		elif item == 'weights':
			return self.weights
		elif item == 'inputs':
			return self.inputs
		elif item == 'outputs':
			return self.outputs
		else:
			print 'Invalid request'