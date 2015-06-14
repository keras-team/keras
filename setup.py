from setuptools import setup
from setuptools import find_packages

setup(name = 'Keras',
      version = '0.1.0',
      description = 'Theano-based Deep Learning library',
      long_description = open('README.md').read(),
      author = 'Francois Chollet',
      author_email = 'francois.chollet@gmail.com',
      url = 'https://github.com/fchollet/keras',
      download_url = 'https://github.com/fchollet/keras/tarball/0.1.0',
      license = 'MIT',
      install_requires = ['theano'],
      packages = find_packages(),
)