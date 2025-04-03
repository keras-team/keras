# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Flexibly compose gradient transformations."""

from collections.abc import Callable, Hashable, Mapping
from typing import NamedTuple, Union

import jax
from optax._src import base
from optax._src import wrappers


def chain(
    *args: base.GradientTransformation,
) -> base.GradientTransformationExtraArgs:
  """Applies a list of chainable update transformations.

  This function creates a new :func:`optax.GradientTransformation` that applies
  a sequence of gradient transformations in order. The ``init`` function of the
  new transformation constructs the optimizer state by concatenating the states
  of the individual transforms, while the ``update`` function applies the
  updates in the given order.

  Args:
    *args: a sequence of chainable (init_fn, update_fn) tuples.

  Returns:
    A :class:`GradientTransformationExtraArgs`, created by chaining the input
    transformations. Note that independent of the argument types, the resulting
    transformation always supports extra args. Any extra arguments passed to the
    returned transformation will be passed only to those transformations in the
    chain that support extra args.

  Examples:

    A transform that scales by -0.1 the adam update:

      >>> import optax
      >>> transform1 = optax.scale_by_adam()
      >>> transform2 = optax.scale(-0.1)
      >>> chained_transform = optax.chain(transform1, transform2)
      >>> params = {'a': 1.0}
      >>> state = chained_transform.init(params)
      >>> updates = {'a': -0.5}
      >>> updates, new_state = chained_transform.update(updates, state, params)
  """

  transforms = [base.with_extra_args_support(t) for t in args]
  init_fns, update_fns = zip(*transforms)

  def init_fn(params):
    return tuple(fn(params) for fn in init_fns)

  def update_fn(updates, state, params=None, **extra_args):
    if len(update_fns) != len(state):
      raise ValueError(
          'The number of updates and states has to be the same in '
          'chain! Make sure you have called init first!'
      )

    new_state = []
    for s, fn in zip(state, update_fns):
      updates, new_s = fn(updates, s, params, **extra_args)
      new_state.append(new_s)
    return updates, tuple(new_state)

  # We opt to always return the GradientTransformationExtraArgs type here,
  # instead of selecting the return type based on the arguments, since it works
  # much better with the currently available type checkers. It also means that
  # users will not get unexpected signature errors if they remove all of the
  # transformations in a chain accepting extra args.
  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def named_chain(
    *transforms: tuple[str, base.GradientTransformation]
) -> base.GradientTransformationExtraArgs:
  """Chains optax gradient transformations.

  A variant of :func:`optax.chain` that allows to name each transformation.

  Here the ``transforms`` are ``(name, transformation)`` pairs, constituted of a
  string ``name`` and an associated transformation ``transformation``. The
  gradient transformation must be an instance of :class:`GradientTransformation`
  or :class:`GradientTransformationExtraArgs`.

  Each ``name`` is used as key for the state of the corresponding transformation
  within the ``named_chain`` state. Thus the state of the transformation
  with a given ``name`` can be easily retrieved as ``opt_state[name]``.

  Args:
    *transforms: an arbitrary number of ``(name, tx)`` pairs, constituted of a
      string ``name`` and an associated transformation ``tx``. The latter is a
      :class:`GradientTransformation` or
      :class:`GradientTransformationExtraArgs`.

  Returns:
    A single (init_fn, update_fn) tuple.

  Examples:

    >>> # tx1 is a GradientTransformation with no extra_args.
    >>> # tx2 is a GradientTransformationExtraArgs that requires `loss`.
    >>> # tx3 is a GradientTransformationExtraArgs that requires `temperature`.
    >>> tx = named_chain(('one', tx1), ('two', tx2), ('three', tx3))
    >>> extra_args={'loss': 0.3, 'temperature': 0.01}
    >>> tx.init(params)
    >>> tx.update(grads, state, params, **extra_args)
  """

  names = [name for name, _ in transforms]

  if len(names) != len(set(names)):
    raise ValueError(
        f'Named transformations must have unique names, but got {names}'
    )

  transforms = [
      (name, base.with_extra_args_support(t)) for name, t in transforms
  ]

  def init_fn(params):
    states = {}
    for name, tx in transforms:
      states[name] = tx.init(params)
    return states

  def update_fn(updates, state, params=None, **extra_args):
    new_state = {}
    for name, tx in transforms:
      updates, new_state[name] = tx.update(
          updates, state[name], params, **extra_args
      )
    return updates, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


class PartitionState(NamedTuple):
  inner_states: Mapping[Hashable, base.OptState]


def partition(
    transforms: Mapping[Hashable, base.GradientTransformation],
    param_labels: Union[base.PyTree, Callable[[base.PyTree], base.PyTree]],
    *,
    mask_compatible_extra_args: bool = False,
) -> base.GradientTransformationExtraArgs:
  """Partitions params and applies a different transformation to each subset.

  Sometimes you may want to apply different transformations to different
  parameters. For example, you may want to apply Adam to the weights of a
  neural network, but SGD to the biases. This function allows you to do that.

  Args:
    transforms: A mapping from labels to transformations. Each transformation
      will be only be applied to parameters with the same label.
    param_labels: A PyTree that is the same shape or a prefix of the
      parameters/updates (or a function that returns one given the parameters as
      input). The leaves of this PyTree correspond to the keys of the transforms
      (therefore the values at the leaves must be a subset of the keys).
    mask_compatible_extra_args: Whether to also apply the same masking to
      extra_arg fields with the same tree structure as params/updates.

  Returns:
    A :func:`optax.GradientTransformationExtraArgs` that implements an ``init``
    and ``update`` function.

  Examples:

    Below is an example where we apply Adam to the weights and SGD to the biases
    of a 2-layer neural network::

      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp

      >>> def map_nested_fn(fn):
      ...   '''Recursively apply `fn` to key-value pairs of a nested dict.'''
      ...   def map_fn(nested_dict):
      ...     return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
      ...             for k, v in nested_dict.items()}
      ...   return map_fn

      >>> params = {'linear_1': {'w': jnp.zeros((5, 6)), 'b': jnp.zeros(5)},
      ...           'linear_2': {'w': jnp.zeros((6, 1)), 'b': jnp.zeros(1)}}
      >>> gradients = jax.tree.map(jnp.ones_like, params)  # dummy gradients

      >>> label_fn = map_nested_fn(lambda k, _: k)
      >>> tx = optax.multi_transform(
      ...     {'w': optax.adam(1.0), 'b': optax.sgd(1.0)}, label_fn)
      >>> state = tx.init(params)
      >>> updates, new_state = tx.update(gradients, state, params)
      >>> new_params = optax.apply_updates(params, updates)

    Instead of providing a ``label_fn``, you may provide a PyTree of labels
    directly.  Also, this PyTree may be a prefix of the parameters PyTree. This
    is demonstrated in the GAN pseudocode below::

      >>> generator_params = ...
      >>> discriminator_params = ...
      >>> all_params = (generator_params, discriminator_params)
      >>> param_labels = ('generator', 'discriminator')

      >>> tx = optax.multi_transform(
      >>>     {'generator': optax.adam(0.1), 'discriminator': optax.adam(0.5)},
      >>>     param_labels)

    If you would like to not optimize some parameters, you may wrap
    :func:`optax.multi_transform` with :func:`optax.masked`.
  """

  transforms = {
      k: base.with_extra_args_support(v) for k, v in transforms.items()
  }

  def make_mask(labels, group):
    return jax.tree.map(lambda label: label == group, labels)

  def init_fn(params):
    labels = param_labels(params) if callable(param_labels) else param_labels

    label_set = set(jax.tree.leaves(labels))
    if not label_set.issubset(transforms.keys()):
      raise ValueError(
          'Some parameters have no corresponding transformation.\n'
          f'Parameter labels: {list(sorted(label_set))} \n'
          f'Transforms keys: {list(sorted(transforms.keys()))} \n'
      )

    inner_states = {
        group: wrappers.masked(
            tx,
            make_mask(labels, group),
            mask_compatible_extra_args=mask_compatible_extra_args,
        ).init(params)
        for group, tx in transforms.items()
    }
    return PartitionState(inner_states)

  def update_fn(updates, state, params=None, **extra_args):
    labels = param_labels(updates) if callable(param_labels) else param_labels
    new_inner_state = {}
    for group, tx in transforms.items():
      masked_tx = wrappers.masked(
          tx,
          make_mask(labels, group),
          mask_compatible_extra_args=mask_compatible_extra_args,
      )
      updates, new_inner_state[group] = masked_tx.update(
          updates, state.inner_states[group], params, **extra_args
      )
    return updates, PartitionState(new_inner_state)

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
