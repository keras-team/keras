from __future__ import absolute_import
from __future__ import print_function
import pytest
import os
import numpy as np
np.random.seed(1337)

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, merge, Lambda


def test_ops_1():

	def add(x, y=3.14):
		return x + y

	def sub(x, y=3.14):
		return x - y

	def mul(x, y=3.14):
		return x * y

	def div(x, y=3.14):
		return x / y

	def radd(x, y=3.14):
		return y + x

	def rsub(x, y=3.14):
		return y - x

	def rmul(x, y=3.14):
		return y * x

	def rdiv(x, y=3.14):
		return y / x

	functions = [add, sub, mul, div, radd, rsub, rmul, rdiv]

	X1 = np.random.uniform(1, 2, (7, 10))
	X2 = np.random.uniform(1, 2, (7, 10))

	for func in functions:
		a = Input((10,))
		assert a.__class__._keras_operators_supported
		b = Dense(10)(a)
		c = Lambda(func)(b)
		d = Dense(10)(c)
		model1 = Model(input=a, output=d)
		model1.compile(loss='mse', optimizer='sgd')
		a = Input((10,))
		b = Dense(10)(a)
		c = func(b)
		d = Dense(10)(c)
		model2 = Model(input=a, output=d)
		model2.compile(loss='mse', optimizer='sgd')
		model2.set_weights(model1.get_weights())
		Y1 = model1.predict(X1)
		Y2 = model2.predict(X1)
		assert np.all(Y1 == Y2)
		a1 = Input((10,))
		b1 = Dense(10)(a1)
		a2 = Input((10,))
		b2 = Dense(10)(a2)
		c = merge([b1, b2], mode=lambda x:func(x[0], x[1]), output_shape=lambda s:s[0])
		d = Dense(10)(c)
		model1 = Model(input=[a1, a2], output=d)
		model1.compile(loss='mse', optimizer='sgd')
		a1 = Input((10,))
		b1 = Dense(10)(a1)
		a2 = Input((10,))
		b2 = Dense(10)(a2)
		c = func(b1, b2)
		d = Dense(10)(c)
		model2 = Model(input=[a1, a2], output=d)
		model2.compile(loss='mse', optimizer='sgd')
		model2.set_weights(model1.get_weights())
		Y1 = model1.predict([X1, X2])
		Y2 = model2.predict([X1, X2])
		assert np.all(Y1 == Y2)


if __name__ == '__main__':
	pytest.main([__file__])
