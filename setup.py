#!/usr/bin/env python

from distutils.core import setup

setup(name='Keras',
      version='0.0.1',
      description='Theano-based Deep Learning',
      author='Francois Chollet',
      author_email='francois.chollet@gmail.com',
      url='https://github.com/fchollet/keras',
      license='MIT',
      packages=[
        'keras', 
        'keras.layers', 
        'keras.preprocessing', 
        'keras.datasets', 
        'keras.utils',
      ],
      # TODO: dependencies
)