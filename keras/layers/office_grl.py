from __future__ import print_function
import keras.backend as K
import keras
from keras import Input, Model
from keras.datasets import cifar10
from keras.layers.transfer import MMD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.engine import Layer
import os

from keras.applications.resnet50 import ResNet50
import tensorflow as tf

# Keras implementation of a gradient reversal layer for the Tensorflow backend
# Author: Michele Tonutti  (https://github.com/michetonu/gradient_reversal_keras_tf)              
def reverse_gradient(X, hp_lambda):
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1
    grad_name = "GradientReversal%d" % reverse_gradient.num_calls
    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]
    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)
        return y

class GradientReversal(Layer):
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda
    def build(self, input_shape):
        self.trainable_weights = []
    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#Dataset Input
x_source，y_source = 	
x_target, y_target = 
domain_input = 

# GRL_net
def grl_net(source_shape,target_shape):
    # base CNN net
    base_net = ResNet50(weights='imagenet', include_top=False)
    for layer in base_net.layers:
        layer.trainable = False

	# Actual input of final net
    source_input = Input(shape=source_shape, name='source_input')
	target_input = Input(shape=target_shape, name='target_input')

	# Feature extractor
    all_input = keras.layers.concatenate([source_input,target_input])
	cnn_feature = base_net(all_input)
	bottleneck_output = Dense(256,activation='relu')(cnn_feature)
	bottleneck_source = bottleneck_output[0:32,:,:,:]	#slice bottleneck_output
	class_output = Dense(31,activation='softmax',name='class_output')(bottleneck_source) 
    
    # Domain Discriminator
	grl = GradientReversal(1.0,name='grl')(bottleneck_output)
	domain_fc1 = Dense(1024,activation='relu')(grl)
	domain_fc1 = Dropout(0.5)(domain_fc1)
	domain_fc2 = Dense(1024,activation='relu')(domain_fc2)
	domain_fc2 = Dropout(0.5)(domain_fc2)
	domain_output = Dense(2,activation='softmax',name='domain_output')(domain_fc2)

	model = Model(inputs=[source_input,target_input],outputs=[class_output,domain_output])

	model.compile(optimizer=keras.optimizers.SGD(lr=0.003, momentum=0.9, nesterov=True),
		loss={'class_output':'categorical_crossentropy','domain_output':'categorical_crossentropy'},
		loss_weights={'class_output':1.0,'domain_output':1.0})
	return model

# Train GRL_net
def train_grl_net(x_source,x_target,y_source,domain_input):
	source_shape = x_source.shape[1:]
	target_shape = x_target.shape[1:]
    model = grl_net(source_shape,target_shape)
    model.fit({'source_input':x_source,'target_input':x_target},
        {'class_output':y_source,'domain_output':domain_input},
        epochs=50, batch_size=32)

					

