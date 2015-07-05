import .caffe_pb2 as caffe
import google.protobuf
from converters import ConvertModel, ConvertWeights, ConvertSolver, ConvertMeanFile
from caffe_utils import get_input_graph, output_dim

class CaffeToKeras(object):
	def __init__(prototext=None, caffemodel=None, solver=None, mean=None, phase='train'):
		if phase = 'train':
			self.phase = 0
		else:
			self.phase = 1

		if prototext is not None:
			config = caffe.NetParameter()
			google.protobuf.text_format.Merge(open(prototext).read(), config)

			self.input_dim = config.input_dim
			self.layers = config.layers[:]
			self.model = ConvertModel(self.layers, self.phase, self.input_dim)

		if caffemodel is not None:
			param = caffe.NetParameter()
			param.MergeFromString(open(caffemodel,'rb').read())

			if prototext is None:
				self.layers = param.layers[:]
				self.model = ConvertModel(self.layers, self.phase, self.input_dim)

			self.weights = ConvertWeights(param.layers)
			self.model.copy_weights(self.weights)

		if solverfile is not None:
			# parse and save
			self.solver = ConvertSolver(solver)

		if meanfile is not None:
			#parse and save
			self.mean = ConvertMeanFile(mean)

	def __call__(self, item):
		if item == 'model':
			return self.model
		elif item == 'solver':
			return self.solver
		elif item == 'mean':
			return self.mean
		elif item == 'weights':
			return self.weights
		else:
			print 'Invalid request'