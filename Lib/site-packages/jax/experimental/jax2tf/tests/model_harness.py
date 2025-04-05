# Copyright 2022 The JAX Authors.
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
"""All the models to convert."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
from typing import Any
import re

import numpy as np
import jraph

from jax.experimental.jax2tf.tests.flax_models import actor_critic
from jax.experimental.jax2tf.tests.flax_models import bilstm_classifier
from jax.experimental.jax2tf.tests.flax_models import cnn
from jax.experimental.jax2tf.tests.flax_models import gnn
from jax.experimental.jax2tf.tests.flax_models import resnet
from jax.experimental.jax2tf.tests.flax_models import seq2seq_lstm
from jax.experimental.jax2tf.tests.flax_models import transformer_lm1b as lm1b
from jax.experimental.jax2tf.tests.flax_models import transformer_nlp_seq as nlp_seq
from jax.experimental.jax2tf.tests.flax_models import transformer_wmt as wmt
from jax.experimental.jax2tf.tests.flax_models import vae

import jax
from jax import random

import tensorflow as tf


@dataclasses.dataclass
class ModelHarness:
  name: str
  apply: Callable[..., Any]
  variables: dict[str, Any]
  inputs: Sequence[np.ndarray]
  rtol: float = 1e-4
  polymorphic_shapes: Sequence[str | None] | None = None
  tensor_spec: Sequence[tf.TensorSpec] | None = None

  def __post_init__(self):
    # When providing polymorphic shapes, tensor_spec should be provided as well.
    assert bool(self.polymorphic_shapes) == bool(self.tensor_spec)

  @property
  def tf_input_signature(self):
    def _to_tensorspec(x):
      return tf.TensorSpec(x.shape, tf.dtypes.as_dtype(x.dtype))

    if self.tensor_spec:
      return self.tensor_spec
    else:
      return jax.tree_util.tree_map(_to_tensorspec, self.inputs)

  def apply_with_vars(self, *args, **kwargs):
    return self.apply(self.variables, *args, **kwargs)


##### All harnesses in this file.
ALL_HARNESSES: dict[str, Callable[[str], ModelHarness]] = {}


def _make_harness(harness_fn, name, poly_shapes=None, tensor_specs=None):
  """Partially apply harness in order to create variables lazily.

  Note: quotes and commas are stripped from `name` to ensure they can be passed
        through the command-line.
  """
  if poly_shapes:
    name += "_" + re.sub(r"(?:'|\"|,)", "", str(poly_shapes))
  if tensor_specs:
    tensor_specs = [tf.TensorSpec(spec, dtype) for spec, dtype in tensor_specs]
  partial_fn = functools.partial(
      harness_fn,
      name=name,
      polymorphic_shapes=poly_shapes,
      tensor_spec=tensor_specs)
  if name in ALL_HARNESSES:
    raise ValueError(f"Harness {name} exists already")
  ALL_HARNESSES[name] = partial_fn


######################## Model Harness Definitions #############################


def _actor_critic_harness(name, **kwargs):
  model = actor_critic.ActorCritic(num_outputs=8)
  x = np.zeros((1, 84, 84, 4), np.float32)
  variables = model.init(random.PRNGKey(0), x)
  return ModelHarness(name, model.apply, variables, [x], **kwargs)


def _bilstm_harness(name, **kwargs):
  model = bilstm_classifier.TextClassifier(
      # TODO(marcvanzee): This fails when
      # `embedding_size != hidden_size`. I suppose some arrays are
      # concatenated with incompatible shapes, which could mean
      # something is going wrong in the translation.
      embedding_size=3,
      hidden_size=1,
      vocab_size=13,
      output_size=1,
      dropout_rate=0.,
      word_dropout_rate=0.)
  x = np.array([[2, 4, 3], [2, 6, 3]], np.int32)
  lengths = np.array([2, 3], np.int32)
  variables = model.init(random.PRNGKey(0), x, lengths, deterministic=True)
  apply = functools.partial(model.apply, deterministic=True)
  return ModelHarness(name, apply, variables, [x, lengths], **kwargs)


def _cnn_harness(name, **kwargs):
  model = cnn.CNN()
  x = np.zeros((1, 28, 28, 1), np.float32)
  variables = model.init(random.PRNGKey(0), x)
  return ModelHarness(name, model.apply, variables, [x], **kwargs)


def _get_gnn_graphs():
  n_node = np.arange(3, 11)
  n_edge = np.arange(4, 12)
  total_n_node = np.sum(n_node)
  total_n_edge = np.sum(n_edge)
  n_graph = n_node.shape[0]
  feature_dim = 10
  graphs = jraph.GraphsTuple(
      n_node=n_node,
      n_edge=n_edge,
      senders=np.zeros(total_n_edge, dtype=np.int32),
      receivers=np.ones(total_n_edge, dtype=np.int32),
      nodes=np.ones((total_n_node, feature_dim)),
      edges=np.zeros((total_n_edge, feature_dim)),
      globals=np.zeros((n_graph, feature_dim)),
  )
  return graphs


def _gnn_harness(name, **kwargs):
  # Setting taken from flax/examples/ogbg_molpcba/models_test.py.
  rngs = {
      'params': random.PRNGKey(0),
      'dropout': random.PRNGKey(1),
  }
  graphs = _get_gnn_graphs()
  model = gnn.GraphNet(
      latent_size=5,
      num_mlp_layers=2,
      message_passing_steps=2,
      output_globals_size=15,
      use_edge_model=True)
  variables = model.init(rngs, graphs)
  return ModelHarness(name, model.apply, variables, [graphs], rtol=2e-4,
                      **kwargs)


def _gnn_conv_harness(name, **kwargs):
  # Setting taken from flax/examples/ogbg_molpcba/models_test.py.
  rngs = {
      'params': random.PRNGKey(0),
      'dropout': random.PRNGKey(1),
  }
  graphs = _get_gnn_graphs()
  model = gnn.GraphConvNet(
      latent_size=5,
      num_mlp_layers=2,
      message_passing_steps=2,
      output_globals_size=5)
  variables = model.init(rngs, graphs)
  return ModelHarness(name, model.apply, variables, [graphs], **kwargs)


def _resnet50_harness(name, **kwargs):
  model = resnet.ResNet50(num_classes=2, dtype=np.float32)
  x = np.zeros((8, 16, 16, 3), np.float32)
  variables = model.init(random.PRNGKey(0), x)
  apply = functools.partial(model.apply, train=False, mutable=False)
  return ModelHarness(name, apply, variables, [x], **kwargs)


def _seq2seq_lstm_harness(name, **kwargs):
  model = seq2seq_lstm.Seq2seq(teacher_force=True, hidden_size=2, vocab_size=4)
  encoder_inputs = np.zeros((1, 2, 4), np.float32)  # [batch, inp_len, vocab]
  decoder_inputs = np.zeros((1, 3, 4), np.float32)  # [batch, outp_len, vocab]
  rngs = {
      'params': random.PRNGKey(0),
      'lstm': random.PRNGKey(1),
  }
  xs = [encoder_inputs, decoder_inputs]
  variables = model.init(rngs, *xs)
  apply = functools.partial(model.apply, rngs={'lstm': random.PRNGKey(2)})
  return ModelHarness(name, apply, variables, xs, **kwargs)


def _min_transformer_kwargs():
  return dict(
      vocab_size=8,
      output_vocab_size=8,
      emb_dim = 4,
      num_heads= 1,
      num_layers = 1,
      qkv_dim= 2,
      mlp_dim = 2,
      max_len = 2,
      dropout_rate = 0.,
      attention_dropout_rate = 0.)


def _full_transformer_kwargs():
  kwargs = dict(
      decode = True,
      deterministic = True,
      logits_via_embedding=False,
      share_embeddings=False)
  return {**kwargs, **_min_transformer_kwargs()}


def _transformer_lm1b_harness(name, **kwargs):
  config = lm1b.TransformerConfig(**_full_transformer_kwargs())
  model = lm1b.TransformerLM(config=config)
  x = np.zeros((2, 1), np.float32)
  rng1, rng2 = random.split(random.PRNGKey(0))
  variables = model.init(rng1, x)

  def apply(*args):
    # Don't return the new state (containing the cache).
    output, _ = model.apply(*args, rngs={'cache': rng2}, mutable=['cache'])
    return output

  return ModelHarness(name, apply, variables, [x], **kwargs)


def _transformer_nlp_seq_harness(name, **kwargs):
  config = nlp_seq.TransformerConfig(**_min_transformer_kwargs())
  model = nlp_seq.Transformer(config=config)
  x = np.zeros((2, 1), np.float32)
  variables = model.init(random.PRNGKey(0), x, train=False)
  apply = functools.partial(model.apply, train=False)
  return ModelHarness(name, apply, variables, [x], **kwargs)


def _transformer_wmt_harness(name, **kwargs):
  config = wmt.TransformerConfig(**_full_transformer_kwargs())
  model = wmt.Transformer(config=config)
  x = np.zeros((2, 1), np.float32)
  variables = model.init(random.PRNGKey(0), x, x)

  def apply(*args):
    # Don't return the new state (containing the cache).
    output, _ = model.apply(*args, mutable=['cache'])
    return output

  return ModelHarness(name, apply, variables, [x, x], **kwargs)


def _vae_harness(name, **kwargs):
  model = vae.VAE(latents=3)
  x = np.zeros((1, 8, 8, 3), np.float32)
  rng1, rng2 = random.split(random.PRNGKey(0))
  variables = model.init(rng1, x, rng2)
  generate = lambda v, x: model.apply(v, x, method=model.generate)
  return ModelHarness(name, generate, variables, [x], **kwargs)


####################### Model Harness Construction #############################


# actor_critic input spec: [((1, 84, 84, 4), np.float32)].
for poly_shapes, tensor_specs in [
    (None, None),  # No polymorphism.
    # batch polymorphism.
    (["(b, ...)"], [((None, 84, 84, 4), tf.float32)]),
    # Dependent shapes for spatial dims.
    # TODO(marcvanzee): Figure out the right multiple for these dimensions.
    (["(_, 4*b, 4*b, _)"], [((1, None, None, 4), tf.float32)]),
]:
  _make_harness(
      harness_fn=_actor_critic_harness,
      name="flax/actor_critic",
      poly_shapes=poly_shapes,
      tensor_specs=tensor_specs)

# bilstm input specs: [((2, 3), np.int32), ((2,), np.int32)] = [inputs, lengths]
for poly_shapes, tensor_specs in [  # type: ignore
    (None, None),
    # batch polymorphism
    (["(b, _)", "(_,)"], [((None, 3), tf.int32), ((2,), tf.int32)]),
    # dynamic input lengths
    (["(_, _)", "(b,)"], [((2, 3), tf.int32), ((None,), tf.int32)]),
]:
  _make_harness(
      harness_fn=_bilstm_harness,
      name="flax/bilstm",
      poly_shapes=poly_shapes,
      tensor_specs=tensor_specs)

# cnn input spec: [((1, 28, 28, 1), np.float32)].
for poly_shapes, tensor_specs in [
    (None, None),  # No polymorphism.
    # batch polymorphism.
    (["(b, ...)"], [((None, 28, 28, 1), tf.float32)]),
    # Dependent shapes for spatial dims.
    # TODO(marcvanzee): Figure out the right multiple for these dimensions.
    (["(_, b, b, _)"], [((1, None, None, 1), tf.float32)]),
]:
  _make_harness(
      harness_fn=_cnn_harness,
      name="flax/cnn",
      poly_shapes=poly_shapes,
      tensor_specs=tensor_specs)

# We do not support polymorphism for the GNN examples since they use GraphTuples
# as input rather than regular arrays.
_make_harness(harness_fn=_gnn_harness, name="flax/gnn")
_make_harness(harness_fn=_gnn_conv_harness, name="flax/gnn_conv")

# resnet50 input spec: [((8, 16, 16, 3), np.float32)]
for poly_shapes, tensor_specs in [
    (None, None),  # No polymorphism.
    # batch polymorphism.
    (["(b, ...)"], [((None, 16, 16, 3), tf.float32)]),
    # Dependent shapes for spatial dims.
    # TODO(marcvanzee): Figure out the right multiple for these dimensions.
    (["(_, 4*b, 4*b, _)"], [((8, None, None, 3), tf.float32)]),
]:
  _make_harness(
      harness_fn=_resnet50_harness,
      name="flax/resnet50",
      poly_shapes=poly_shapes,
      tensor_specs=tensor_specs)


# seq2seq input specs (we use the same input and output lengths for now):
# [
#   ((1, 2, 4), np.float32),  # encoder inp: [batch, max_input_len, vocab_size]
#   ((1, 3, 4), np.float32),  # decoder_inp: [batch, max_output_len, vocab_size]
# ]
for poly_shapes, tensor_specs in [  # type: ignore
    (None, None),
    # batch polymorphism
    (
        ["(b, _, _)",                "(b, _, _)"],
        [((None, 2, 4), tf.float32), ((None, 3, 4), tf.float32)],
    ),
    # dynamic input lengths
    (
        ["(_, b, _)",                "(_, _, _)"],
        [((1, None, 4), tf.float32), ((1, 3, 4), tf.float32)],
    ),
    # dynamic output lengths
    (
        ["(_, _, _)",                 "(_, b, _)"],
        [((1, 2, 4), tf.float32),     ((1, None, 4), tf.float32)],
    ),
]:
  _make_harness(
      harness_fn=_seq2seq_lstm_harness,
      name="flax/seq2seq_lstm",
      poly_shapes=poly_shapes,
      tensor_specs=tensor_specs)

# lm1b/nlp_seq input spec: [((2, 1), np.float32)]  [batch, seq_len]
for poly_shapes, tensor_specs in [  # type: ignore
    (None, None),
    # batch polymorphism.
    (["(b, _)"], [((None, 1), tf.float32)]),
]:
  for name, harness_fn in [
      ("flax/lm1b", _transformer_lm1b_harness),
      ("flax/nlp_seq", _transformer_nlp_seq_harness)
  ]:
    _make_harness(
        harness_fn=harness_fn,
        name=name,
        poly_shapes=poly_shapes,
        tensor_specs=tensor_specs)

# wmt input spec (both inputs have the same shape):
# [
#   ((1, 2), np.float32),  # inputs:  [batch, max_target_len]
#   ((1, 2), np.float32),  # targets: [batch, max_target_len]
# ]
for poly_shapes, tensor_specs in [  # type: ignore
    (None, None),
    # batch polymorphism.
    (["(b, _)"] * 2, [((None, 1), tf.float32)] * 2),
    # dynamic lengths.
    (["(_, b)"] * 2, [((1, None), tf.float32)] * 2),
]:
  _make_harness(
      harness_fn=_transformer_wmt_harness,
      name="flax/wmt",
      poly_shapes=poly_shapes,
      tensor_specs=tensor_specs)

# vae input spec: [((1, 8, 8, 3), np.float32)].
for poly_shapes, tensor_specs in [
    (None, None),  # No polymorphism.
    # batch polymorphism.
    (["(b, ...)"], [((None, 8, 8, 3), tf.float32)]),
    # Dependent shapes for spatial dims.
    # TODO(marcvanzee): Figure out the right multiple for these dimensions.
    (["(_, b, b, _)"], [((1, None, None, 3), tf.float32)]),
]:
  _make_harness(
      harness_fn=_vae_harness,
      name="flax/vae",
      poly_shapes=poly_shapes,
      tensor_specs=tensor_specs)
