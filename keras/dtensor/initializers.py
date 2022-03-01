# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""DTensor specific Keras initializers."""

import inspect
import sys


from keras import backend
from keras.dtensor import utils
from keras.initializers import initializers_v2
from keras.utils import generic_utils
from keras.utils import tf_inspect


class Initializer(initializers_v2.Initializer):
  """DTensor specific initializer.

  Note that the initializer will take an extra argument in `__call__` , which is
  the `layout` for the init value.
  """

  def __call__(self, shape, dtype=None, layout=None, **kwargs):
    raise NotImplementedError('Initializer subclasses must implement the '
                              '`__call__()` method.')


def _ensure_keras_seeded():
  """Make sure the keras.backend global seed generator is set.

  This is important for DTensor use case to ensure that each client are
  initialized with same seed for tf.random.Generator, so that the value created
  are in sync among all the clients.
  """
  if not getattr(backend._SEED_GENERATOR, 'generator', None):  # pylint:disable=protected-access
    raise ValueError('When using DTensor APIs, you need to set the global seed '
                     'before using any Keras initializers. Please make sure '
                     'to call `tf.keras.utils.set_random_seed()` in your code.')


class Zeros(initializers_v2.Zeros):

  def __call__(self, shape, dtype=None, **kwargs):
    layout = kwargs.pop('layout', None)
    fn = super(Zeros, self).__call__
    return utils.call_with_layout(fn, layout, shape=shape, dtype=dtype)


class Ones(initializers_v2.Ones):

  def __call__(self, shape, dtype=None, **kwargs):
    layout = kwargs.pop('layout', None)
    fn = super(Ones, self).__call__
    return utils.call_with_layout(fn, layout, shape=shape, dtype=dtype)


class Constant(initializers_v2.Constant):

  def __call__(self, shape, dtype=None, **kwargs):
    layout = kwargs.pop('layout', None)
    fn = super(Constant, self).__call__
    return utils.call_with_layout(fn, layout, shape=shape, dtype=dtype)


# pylint:disable=missing-class-docstring
class RandomUniform(initializers_v2.RandomUniform):

  def __init__(self, minval=-0.05, maxval=0.05, seed=None):
    super().__init__(minval=minval, maxval=maxval, seed=seed)
    # Make sure to use the tf.random.Generator which doesn't rely on the
    # stateful random ops.
    # TODO(scottzhu): Remove this once the backend.use_generator_for_rng is
    # default to True.
    self._random_generator._force_generator = True

  def __call__(self, shape, dtype=None, **kwargs):
    layout = kwargs.pop('layout', None)
    if layout:
      _ensure_keras_seeded()
    fn = super(RandomUniform, self).__call__
    return utils.call_with_layout(fn, layout, shape=shape, dtype=dtype)


class RandomNormal(initializers_v2.RandomNormal):

  def __init__(self, mean=0.0, stddev=0.05, seed=None):
    super().__init__(mean=mean, stddev=stddev, seed=seed)
    self._random_generator._force_generator = True

  def __call__(self, shape, dtype=None, **kwargs):
    layout = kwargs.pop('layout', None)
    if layout:
      _ensure_keras_seeded()
    fn = super(RandomNormal, self).__call__
    return utils.call_with_layout(fn, layout, shape=shape, dtype=dtype)


class TruncatedNormal(initializers_v2.TruncatedNormal):

  def __init__(self, mean=0.0, stddev=0.05, seed=None):
    super().__init__(mean=mean, stddev=stddev, seed=seed)
    self._random_generator._force_generator = True

  def __call__(self, shape, dtype=None, **kwargs):
    layout = kwargs.pop('layout', None)
    if layout:
      _ensure_keras_seeded()
    fn = super(TruncatedNormal, self).__call__
    return utils.call_with_layout(fn, layout, shape=shape, dtype=dtype)


class Identity(initializers_v2.Identity):

  def __call__(self, shape, dtype=None, **kwargs):
    layout = kwargs.pop('layout', None)
    fn = super(Identity, self).__call__
    return utils.call_with_layout(fn, layout, shape=shape, dtype=dtype)


class Orthogonal(initializers_v2.Orthogonal):

  def __init__(self, gain=1.0, seed=None):
    super().__init__(gain=gain, seed=seed)
    self._random_generator._force_generator = True

  def __call__(self, shape, dtype=None, **kwargs):
    layout = kwargs.pop('layout', None)
    if layout:
      _ensure_keras_seeded()
    fn = super(Orthogonal, self).__call__
    return utils.call_with_layout(fn, layout, shape=shape, dtype=dtype)


class VarianceScaling(initializers_v2.VarianceScaling):

  def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal',
               seed=None):
    super().__init__(scale=scale, mode=mode, distribution=distribution,
                     seed=seed)
    self._random_generator._force_generator = True

  def __call__(self, shape, dtype=None, **kwargs):
    layout = kwargs.pop('layout', None)
    if layout:
      _ensure_keras_seeded()
    fn = super(VarianceScaling, self).__call__
    return utils.call_with_layout(fn, layout, shape=shape, dtype=dtype)


class GlorotUniform(VarianceScaling):

  def __init__(self, seed=None):
    super().__init__(
        scale=1.0, mode='fan_avg', distribution='uniform', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


class GlorotNormal(VarianceScaling):

  def __init__(self, seed=None):
    super().__init__(
        scale=1.0, mode='fan_avg', distribution='truncated_normal', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


class LecunNormal(VarianceScaling):

  def __init__(self, seed=None):
    super().__init__(
        scale=1., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


class LecunUniform(VarianceScaling):

  def __init__(self, seed=None):
    super().__init__(
        scale=1., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


class HeNormal(VarianceScaling):

  def __init__(self, seed=None):
    super().__init__(
        scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


class HeUniform(VarianceScaling):

  def __init__(self, seed=None):
    super().__init__(
        scale=2., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


# Populate all initializers with their string names.
_ALL_INITIALIZERS = {}
for name, obj in inspect.getmembers(sys.modules[__name__]):
  if inspect.isclass(obj) and issubclass(obj, initializers_v2.Initializer):
    _ALL_INITIALIZERS[name] = obj
    alternative_name = generic_utils.to_snake_case(name)
    _ALL_INITIALIZERS[alternative_name] = obj


def serialize(initializer):
  return generic_utils.serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
  """Return an `Initializer` object from its config."""
  return generic_utils.deserialize_keras_object(
      config,
      module_objects=_ALL_INITIALIZERS,
      custom_objects=custom_objects,
      printable_module_name='initializer')


def get(identifier):
  """Retrieve an initializer by the identifier."""
  # This function is copied from keras, and we only want to inject the logic for
  # `deserialize()`.
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    identifier = str(identifier)
    return deserialize(identifier)
  elif callable(identifier):
    if tf_inspect.isclass(identifier):
      identifier = identifier()
    return identifier
  else:
    raise ValueError('Could not interpret initializer identifier: ' +
                     str(identifier))
