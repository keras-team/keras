# Copyright 2023 The JAX Authors.
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

from functools import partial
import itertools

from jax._src import config
from jax._src import core
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.dispatch import apply_primitive
from jax._src.tree_util import tree_flatten, tree_unflatten
from jax._src.interpreters import batching
from jax._src.util import safe_zip
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import dialects, ir

_next_shard_group_id = itertools.count()

def shard_alike(x, y):
  """Shards x and y alike."""
  x_flat, x_tree = tree_flatten(x)
  y_flat, y_tree = tree_flatten(y)

  if x_tree != y_tree:
    raise ValueError('Trees should be equal. '
                     f'Got x_tree: {x_tree}, y_tree: {y_tree}')

  for x_, y_ in safe_zip(x_flat, y_flat):
    x_aval = core.shaped_abstractify(x_)
    y_aval = core.shaped_abstractify(y_)
    if x_aval.shape != y_aval.shape:
      raise ValueError(
          'The leaves shapes of `x` and `y` should match. Got `x` leaf shape:'
          f' {x_aval.shape} and `y` leaf shape: {y_aval.shape}. File an issue at'
          ' https://github.com/jax-ml/jax/issues if you want this feature.')

  outs = [shard_alike_p.bind(x_, y_) for x_, y_ in safe_zip(x_flat, y_flat)]
  x_out_flat, y_out_flat = zip(*outs)
  return tree_unflatten(x_tree, x_out_flat), tree_unflatten(y_tree, y_out_flat)


shard_alike_p = core.Primitive('shard_alike')
shard_alike_p.multiple_results = True
shard_alike_p.def_impl(partial(apply_primitive, shard_alike_p))
shard_alike_p.def_abstract_eval(lambda x, y: (x, y))

def shard_alike_transpose(ct, **kwargs):
  x_ct, y_ct = ct
  if type(x_ct) is ad.Zero or type(y_ct) is ad.Zero:
    return x_ct, y_ct
  else:
    return shard_alike(x_ct, y_ct)
ad.deflinear(shard_alike_p, shard_alike_transpose)


def _shard_alike_batcher(batched_args, batch_dims):
  x, y = batched_args
  xd, yd = batch_dims
  if xd == yd:
    return shard_alike(x, y), (xd, yd)
  elif xd is batching.not_mapped:
    x = batching.broadcast(x, y.shape[yd], yd)
    return shard_alike(x, y), (yd, yd)
  elif yd is batching.not_mapped:
    y = batching.broadcast(y, x.shape[xd], xd)
    return shard_alike(x, y), (xd, xd)
  else:
    y = batching.moveaxis(y, yd, xd)
    return shard_alike(x, y), (xd, xd)
batching.primitive_batchers[shard_alike_p] = _shard_alike_batcher


def _group_shard(
    ctx,
    x: ir.Value,
    y: ir.Value,
    x_aval_out: core.AbstractValue,
    y_aval_out: core.AbstractValue,
) -> tuple[ir.Value, ir.Value]:
  shard_group_id = next(_next_shard_group_id)

  if config.use_shardy_partitioner.value:
    dialects.sdy.ShardingGroupOp(x, shard_group_id)
    dialects.sdy.ShardingGroupOp(y, shard_group_id)
    return x, y

  unknown_op_sharding = xc.OpSharding()
  unknown_op_sharding.type = xc.OpSharding.Type.UNKNOWN
  unknown_op_sharding.is_shard_group = True
  unknown_op_sharding.shard_group_id = shard_group_id
  unknown_op_sharding.shard_group_type = xc.OpSharding.ShardGroupType.AS

  x = mlir.wrap_with_sharding_op(ctx, x, x_aval_out, unknown_op_sharding,
                                 has_side_effect=True)
  y = mlir.wrap_with_sharding_op(ctx, y, y_aval_out, unknown_op_sharding,
                                 has_side_effect=True)
  return x, y


def shard_alike_lowering(ctx, x, y):
  return _group_shard(ctx, x, y, *ctx.avals_out)
mlir.register_lowering(shard_alike_p, shard_alike_lowering)
