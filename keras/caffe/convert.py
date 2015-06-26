from __future__ import absolute_import
from __future__ import print_function

from converter import ConvertModel, ConvertWeights, ConvertSolver, ConvertMeanFile

import google.protobuf

class CaffeToKeras(object):
	def __init__(prototext=None, caffemodel=None, solver=None, imagemean=None, phase='train'):

		self.model = None
		self.weights = None
		self.solver = None
		self.mean = None

		self.layers = None
		self.weights = None

		if phase = 'train':
			self.phase = 0
		else:
			self.phase = 1

		if prototext is not None:
			config = caffe.NetParameter()
			google.protobuf.text_format.Merge(open(prototext).read(), config)

			self.layers = config.layers[:]

			self.model = ConvertModel(self.layers)

		if caffemodel is not None:
			param = caffe.NetParameter()
			param.MergeFromString(open(caffemodel,'rb').read())

			if self.model is None:
				self.layers = param.layers[:]
				model = ConvertModel(self.layers)
				result = [model]

			self.weights = ConvertWeights(param.layers) # dictionary of layer_name : weights
			result[0].load_weights(weights)

		if solverfile is not None:
			keras_solver = ConvertSolver(solverfile)
			self.solver = keras_solver

		if meanfile is not None:
			mean = ConvertMeanFile(meanfile)
			self.mean = mean

	def get_model():
		if self.model is not None:
			self.model.describe()
			return self.model

	def get_weights():
		if self.weights is not None:
			return self.weights

	def get_solver():
		if self.solver is not None:
			print('Learning Rate: ', self.solver.lr)
			print('Momentum: ', self.solver.momentum)
			print('\tNestrov: ', self.solver.nestrov)
			return self.solver

	def get_mean():
		if self.mean is not None:
			return self.mean