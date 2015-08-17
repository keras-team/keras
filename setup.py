from setuptools import setup
from setuptools import find_packages


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
