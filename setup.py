from setuptools import setup
from setuptools import find_packages

import os
from six.moves.urllib.request import urlretrieve

# First, compile Caffe protobuf Python file
datadir = os.path.expanduser(os.path.join('~', '.keras', 'data'))
if not os.path.exists(datadir):
    os.makedirs(datadir)

caffe_source = os.path.join(datadir, 'caffe.proto')
caffe_destination = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'keras', 'caffe')
urlretrieve('https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto', caffe_source)
os.system('protoc --proto_path="' + datadir + '" --python_out="' + caffe_destination + '" "' + caffe_source + '"')

setup(name='Keras',
      version='0.1.2',
      description='Theano-based Deep Learning library',
      author='Francois Chollet',
      author_email='francois.chollet@gmail.com',
      url='https://github.com/fchollet/keras',
      download_url='https://github.com/fchollet/keras/tarball/0.1.2',
      license='MIT',
      install_requires=['theano', 'pyyaml', 'h5py'],
      packages=find_packages())
