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
			self.layers = config.layers[:]
			self.network = model_from_config(self.layers, self.phase, self.input_dim[1:])

		if caffemodel is not None:
			param = caffe.NetParameter()
			param.MergeFromString(open(caffemodel,'rb').read())

			self.weights = convert_weights(param.layers)

			if prototext is None:
				self.layers = param.layers[:]
				self.network = model_from_param(self.layers)
			else:
				self.network.copy_weights(self.weights)

		if solver is not None:
			# parse and save
			self.solver = convert_solver(solver)

		if mean is not None:
			# parse and save
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
		else:
			print 'Invalid request'