from setuptools import setup
from setuptools import find_packages


setup(name='Keras',
      version='2.0.0-tf',
      description='Deep Learning for Python',
      author='Francois Chollet',
      author_email='francois.chollet@gmail.com',
      url='https://github.com/fchollet/keras',
      license='MIT',
      install_requires=['tensorflow', 'pyyaml', 'six'],
      extras_require={
          'h5py': ['h5py'],
          'visualize': ['pydot-ng'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist'],
      },
      packages=find_packages())
