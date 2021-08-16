# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow Keras.

TensorFlow Keras is an implementation of the Keras API that uses TensorFlow as
a backend.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import setuptools

DOCLINES = __doc__.split('\n')

# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
_VERSION = '2.7.0'

REQUIRED_PACKAGES = [
    # We depend on TensorFlow's declared pip dependencies.
    # Add a new dep there if one is needed.
]

project_name = 'keras'
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)


setuptools.setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
    description='Deep learning for humans.',
    long_description='\n'.join(DOCLINES[2:]),
    url='https://keras.io/',
    download_url='https://github.com/keras-team/keras/tags',
    author='Keras team',
    author_email='keras-users@googlegroups.com',
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # PyPI package information.
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords=['keras', 'tensorflow', 'machine learning', 'deep learning'],
)
