import .caffe_pb2 as caffe
import google.protobuf
from converters import ConvertModel, ConvertWeights, ConvertSolver, ConvertMeanFile
from caffe_utils import get_input_graph, output_dim

class CaffeToKeras(object):
	def __init__(prototext=None, caffemodel=None, solver=None, mean=None, phase='train'):
		'''
			prototext: model description file in caffe
			caffemodel: stored weights file
			solver: solver file
			mean: mean data file
			phase: train or test

			Usage:
				model = CaffeToKeras(prototext='VGG16.prototxt', caffemodel='VGG16_700iter.caffemodel')
				graph = model('network')
				weights = model('weights') # useful for embedding networks
		'''
		if phase = 'train':
			self.phase = 0
		else:
			self.phase = 1

		if prototext is not None:
			config = caffe.NetParameter()
			google.protobuf.text_format.Merge(open(prototext).read(), config)

			self.input_dim = config.input_dim
			self.layers = config.layers[:]
			self.network = ConvertModel(self.layers, self.phase, self.input_dim)

		if caffemodel is not None:
			param = caffe.NetParameter()
			param.MergeFromString(open(caffemodel,'rb').read())

			if prototext is None:
				self.layers = param.layers[:]
				self.network = ConvertModel(self.layers, self.phase, self.input_dim)

			self.weights = ConvertWeights(param.layers)
			self.network.copy_weights(self.weights)

		if solverfile is not None:
			# parse and save
			self.solver = ConvertSolver(solver)

		if meanfile is not None:
			#parse and save
			self.mean = ConvertMeanFile(mean)

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