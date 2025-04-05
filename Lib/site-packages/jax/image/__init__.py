# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image manipulation functions.

More image manipulation functions can be found in libraries built on top of
JAX, such as `PIX`_.

.. _PIX: https://github.com/deepmind/dm_pix
"""

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.image.scale import (
  resize as resize,
  ResizeMethod as ResizeMethod,
  scale_and_translate as scale_and_translate,
)
