from setuptools import setup
from setuptools import find_packages

long_description = '''
Keras is a high-level neural networks API for Python.

Read the documentation at: https://keras.io/

Keras is compatible with Python 3.6+
and is distributed under the MIT license.
'''

setup(name='Keras',
      version='2.4.3',
      description='Deep Learning for humans',
      long_description=long_description,
      author='Francois Chollet',
      author_email='francois.chollet@gmail.com',
      url='https://github.com/keras-team/keras',
      download_url='https://github.com/keras-team/keras/tarball/2.4.3',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'pyyaml',
                        'h5py'],
      extras_require={
          'visualize': ['pydot>=1.2.4'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'flaky',
                    'pytest-cov',
                    'pandas',
                    'requests',
                    'markdown'],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
