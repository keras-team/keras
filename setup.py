from setuptools import setup
from setuptools import find_packages

setup(name='Keras',
      version='0.0.1',
      description='Theano-based Deep Learning',
      author='Francois Chollet',
      author_email='francois.chollet@gmail.com',
      url='https://github.com/fchollet/keras',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'pillow'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Science/Research'
      ]
)
