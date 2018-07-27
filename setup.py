from setuptools import setup
from setuptools import find_packages

long_description = '''
Keras is a high-level neural networks API,
written in Python and capable of running on top of
TensorFlow, CNTK, or Theano.

Use Keras if you need a deep learning library that:

- Allows for easy and fast prototyping
  (through user friendliness, modularity, and extensibility).
- Supports both convolutional networks and recurrent networks,
  as well as combinations of the two.
- Runs seamlessly on CPU and GPU.

Read the documentation at: https://keras.io/

For a detailed overview of what makes Keras special, see:
https://keras.io/why-use-keras/

Keras is compatible with Python 2.7-3.6
and is distributed under the MIT license.
'''

setup(name='Keras',
      version='2.2.1',
      description='Deep Learning for humans',
      long_description=long_description,
      author='Francois Chollet',
      author_email='francois.chollet@gmail.com',
      url='https://github.com/keras-team/keras',
      download_url='https://github.com/keras-team/keras/tarball/2.2.1',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'pyyaml',
                        'h5py',
                        'keras_applications==1.0.4',
                        'keras_preprocessing==1.0.2'],
      extras_require={
          'visualize': ['pydot>=1.2.4'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov',
                    'pandas',
                    'requests'],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
