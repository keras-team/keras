# Copyright 2024 The JAX Authors.
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

"""Implements SdyShardingRule."""

from collections import OrderedDict

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import sdy


# A single character replacement for ... to simplify parsing.
BATCHING: str = "…"

# A prefix for names of batching dimension factors, used for expanding the
# leading ... into factors.
_BATCHING_DIM_FACTOR_PREFIX = "?"

# A Jax value in general corresponds to an ir.Type or a tuple of ir.Types.
IrTypes = ir.Type | tuple[ir.Type, ...]

def _check_factor(factor:str):
  """Validates a factor.

  A factor is a string starting with a letter and containing only letters,
  digits, or underscores.
  """
  if not factor[0].isalpha():
    raise ValueError(f"Factor names have to start with a letter, but got '{factor[0]}'")
  for char in factor[1:]:
    if char != "_" and not char.isdigit() and not char.isalpha():
      raise ValueError(f"Unknown character '{char}'")

class CompoundFactor(tuple):
  """Describes the factors for a compound factor.

  A compound factor should contain at least two factors, e.g.
  * CompoundFactor('b', 'c').
  """
  def __init__(self, *factors):
    if len(factors) < 2:
      raise ValueError("A compound factor should contain at least two factors")
    for factor in factors:
      if not isinstance(factor, str):
        raise ValueError(f"Each element of CompoundFactor must be a str, but got {type(factor)}")
      if factor == BATCHING:
        raise ValueError("Ellipsis can't be used in a compound factor")
      else:
        _check_factor(factor)

  def __new__(cls, *factors):
    return tuple.__new__(CompoundFactor, factors)


class ArrayMapping(tuple):
  """Describes the factors for an operand or result.

  Each element is either a factor or a CompoundFactor. A leading element can
  also be BATCHING, which represents batching dimensions. examples:
  * ArrayMapping('a')
  * ArrayMapping('b', 'c')
  * ArrayMapping(CompoundFactor('b', 'c'), 'd')
  * ArrayMapping(BATCHING, CompoundFactor('b', 'c'), 'd')
  """
  def __init__(self, *dim_mappings):
    for i, d in enumerate(dim_mappings):
      if not isinstance(d, str) and not isinstance(d, CompoundFactor):
        raise ValueError(
            "Each element of ArrayMapping must be a str or CompoundFactor, but"
            f" got {type(d)}")
      if isinstance(d, str):
        if d == BATCHING:
          if i != 0:
            raise ValueError("Ellipsis can only be used at the beginning of a dimension")
        else:
          _check_factor(d)

  def __new__(cls, *dim_mappings):
    return tuple.__new__(ArrayMapping, dim_mappings)


class SdyShardingRule:
  """Represents a Shardy sharding rule.

  An SdyShardingRule contains the ArrayMappings for operands and results, and an
  optional list of factor sizes. A factor is a name used in the ArrayMappings.
  If a factor is only used in CompoundFactors, its size must be specified.
  """
  operand_mappings: tuple[ArrayMapping, ...]
  result_mappings: tuple[ArrayMapping, ...]
  factor_sizes: dict[str, int]

  def __init__(self, operand_mappings: tuple[ArrayMapping, ...],
               result_mappings: tuple[ArrayMapping, ...], **factor_sizes):
    # Find all factors and mark whether their size can be inferred.
    factors_inferrable = dict()
    for value in operand_mappings + result_mappings:
      for dim in value:
        if isinstance(dim, str):
          factors_inferrable[dim] = True
        else:
          for factor in dim:
            if factor not in factors_inferrable.keys():
              factors_inferrable[factor] = False

    # Check that factors in factor_sizes are used in the rule.
    for factor in factor_sizes:
      if factor not in factors_inferrable:
        raise ValueError(
          f"Factor {factor} is not used in the rule, but size is provided")

    # Check that factors that are used for a whole dimension aren't in
    # factor_sizes and factors that are never used for a whole dimension are
    # in factor_sizes.
    for factor, inferrable in factors_inferrable.items():
      if factor not in factor_sizes and not inferrable:
        raise ValueError(
          f"Factor {factor} is only used in compound factors; must specify"
          " its size")
      if factor in factor_sizes and inferrable:
        raise ValueError(
          f"Factor {factor} represents a whole dimension; do not specify its"
          " size")

    self.operand_mappings = operand_mappings
    self.result_mappings = result_mappings
    self.factor_sizes = factor_sizes

  def __str__(self):
    return f"SdyShardingRule({self.operand_mappings}, {self.result_mappings}, {self.factor_sizes})"


def _get_batching_dim_factor_name(batch_dim_order : int):
  """Constructs a factor name for a batching dimension.

  We expand the leading ... into factors representing the batching dimensions
  to support building the MLIR representation for the sharding rule. For this
  reason, we construct a factor name that won't be used by users for the
  batching dimensions.
  """
  return f"{_BATCHING_DIM_FACTOR_PREFIX}{batch_dim_order}"

def _parse_values(
    rule: str,
) -> tuple[ArrayMapping, ...]:
  """Parses the LHS or RHS of an Einsum notation like string.

  Converts each operand or result in the Einsum notation like string to a tuple
  of ArrayMapping. This very closely follows how einops parses their rules in
  einops/parsing.py.

  Args:
    rule: The Einsum notation for the operands or results of an operation.

  Returns:
    The tuple of ArrayMapping.

  Raises:
    ValueError: If the rule is not balanced or contains unknown characters.
  """

  # Remove unnecessary spaces in the rule to simplify the parsing process.
  words = rule.split()
  rule = " ".join(words)

  # Similar to einops rules, an empty LHS/RHS has a single scalar value.
  if not rule:
    return (ArrayMapping(),)

  all_values = []
  # Represent all dimensions of an value. When an value[0]==BATCHING, the
  # value may have 0 or more leading dimensions.
  value = []
  current_factor = None
  # A value of None indicates the current dimension is not a compound dimension,
  # while a value of [] indicates that we have just started parsing a compound
  # dimension.
  current_compound_dim: list[str] | None = None

  def add_factor(x):
    if current_compound_dim is None:
      value.append(x)
    else:
      current_compound_dim.append(x)

  for char in rule:
    if char == BATCHING:
      if (current_factor is not None or current_compound_dim is not None
          or value):
        raise ValueError(
            "Ellipsis can only be used at the beginning of a dimension")
      add_factor(BATCHING)
      continue
    if char in "(), ":
      if current_factor is not None:
        add_factor(current_factor)
        current_factor = None
      if char == "(":
        if current_compound_dim is not None:
          raise ValueError(
              "Compound factors should be one level, nested brackets are not"
              " allowed")
        current_compound_dim = []
      elif char == ")":
        if current_compound_dim is None:
          raise ValueError("Brackets are not balanced")
        if len(current_compound_dim) <= 1:
          raise ValueError("Brackets should contain at least two factors")
        value.append(CompoundFactor(*current_compound_dim))
        current_compound_dim = None
      elif char == ",":
        all_values.append(ArrayMapping(*value))
        value = []
    elif char == "_" or char.isdigit() or char.isalpha():
      if current_factor is None:
        if str.isdigit(char):
          raise ValueError(f"Factor names have to start with a letter, but got '{char}'")
        current_factor = char
      else:
        current_factor += char
    else:
      raise ValueError(f"Unknown character '{char}'")

  if current_compound_dim is not None:
    raise ValueError(f"Brackets are not balanced in rule: '{rule}'")
  if current_factor is not None:
    add_factor(current_factor)
  all_values.append(ArrayMapping(*value))

  return tuple(all_values)

def str_to_sdy_sharding_rule(rule: str, **factor_sizes) -> SdyShardingRule:
  """Constructs a SdyShardingRule object from the Einsum notation like string.

  This is done by verifying that the input Einsum notation like string and
  with optional factor sizes represents a valid sharding rule and converting
  it to an internal representation.

  Args:
    rule: The Einsum notation like string for an operation.
    **factor_sizes: The optional factor sizes.

  Raises:
    ValueError: If there is any problem with the rule or factor_sizes.
  """
  if not isinstance(rule, str):
    raise TypeError(f"rule must be a str, but got {type(rule)}")
  if not all(isinstance(size, int) for size in factor_sizes.values()):
    raise TypeError(
        f"factor_sizes must be a dict of str to int, but got {factor_sizes}")

  # Replace ... with a single char to simplify parsing.
  if BATCHING in rule:
    raise ValueError(f"Unknown character '{BATCHING}'")
  if "." in rule:
    rule = rule.replace("...", BATCHING)
    if "." in rule:
      raise ValueError("Character '.' must be used inside ellipsis '...'")

  try:
    operands, results = rule.split("->")
  except ValueError as e:
    raise ValueError(f"There is no -> in rule: '{rule}'") from e

  operand_mappings = _parse_values(operands)
  result_mappings = _parse_values(results)

  return SdyShardingRule(operand_mappings, result_mappings, **factor_sizes)


def sdy_sharding_rule_to_mlir(
  rule: SdyShardingRule,
  operand_types: list[IrTypes],
  result_types: list[IrTypes],) -> ir.Attribute:
  """Builds the MLIR representation for the sharding rule.

  This is done by verifying that the rule is consistent with the types of
  the operation and converting the Einsum notation like string to
  OpShardingRuleAttr.
  """
  if len(rule.operand_mappings) != len(operand_types):
    raise ValueError(
      f"Sharding rule has {len(rule.operand_mappings)} operands, but the operation"
      f" has {len(operand_types)} operands")
  if len(rule.result_mappings) != len(result_types):
    raise ValueError(
      f"Sharding rule has {len(rule.result_mappings)} results, but the operation"
      f" has {len(result_types)} results")
  if not all(isinstance(t, ir.Type) for t in operand_types + result_types):
    raise TypeError(
        f"operand_types and result_types must be a list of ir.Type, but got"
        f" {operand_types} and {result_types}")

  factors_to_indices_sizes: OrderedDict[str, list[int]] = OrderedDict()
  types = operand_types + result_types
  UNKNOWN = -1  # Representation for unknown factor size or factor index.

  def get_message_for_value(i):
    if i >= len(operand_types):
      return f"{i - len(operand_types)}th result"
    else:
      return f"{i}th operand"

  def get_rank_for_value(i):
    return ir.ShapedType(types[i]).rank

  def get_size_for_value_dim(i, j):
    return ir.ShapedType(types[i]).shape[j]

  def add_factor(factor, size):
    """Adds a factor to factors_to_indices_sizes.

    `size` may be a dimensions size, a user specified factor size, or UNKNOWN
    if a factor is first used as in a compound factor and then used for a
    whole dimension.
    """
    factor_index, factor_size = factors_to_indices_sizes.get(factor, [UNKNOWN, UNKNOWN])
    if factor_index != UNKNOWN:
      # Not the first time seeing the factor.
      if size != UNKNOWN and factor_size != UNKNOWN and factor_size != size:
        factor_or_batching_dim = (
          f"Factor {factor}" if _BATCHING_DIM_FACTOR_PREFIX not in factor
            else f"Batching dimension {factor[1:]}")
        raise ValueError(
          f"{factor_or_batching_dim} corresponds to two sizes:"
          f" {factor_size} and {size}")
      if size != UNKNOWN and factor_size == UNKNOWN:
        factors_to_indices_sizes[factor] = [factor_index, size]
    else:
      # First time seeing the factor.
      factor_index = len(factors_to_indices_sizes)
      factors_to_indices_sizes[factor] = [factor_index, size]

  def add_batching_dim_factor(batch_dim_order, factor_size):
    ellipsis_batch_dim_name = _get_batching_dim_factor_name(batch_dim_order)
    add_factor(ellipsis_batch_dim_name, factor_size)

  def build_dim_mapping_for_compound_factors(i, j, factors):
    accumulated_size = 1
    all_indices = []
    for factor in factors:
      factor_index, factor_size = factors_to_indices_sizes[factor]
      accumulated_size *= factor_size
      all_indices.append(factor_index)

    dim_size = get_size_for_value_dim(i, j)
    if accumulated_size != dim_size:
      raise ValueError(
          f"{get_message_for_value(i)} actual size {dim_size} doesn't match"
          f" the size {accumulated_size} derived from the compound factors"
          f" {factors}")

    return sdy.DimMappingAttr.get(factor_indices=all_indices)

  # Add factors and their sizes in the order they appear in the rule,
  # including the batching dimensions represented by ellipsis.
  ellipsis_rank = None
  for i, mapping in enumerate(rule.operand_mappings + rule.result_mappings):
    value = tuple(mapping)
    if value and value[0] == BATCHING:
      has_batching = True
      value = value[1:]
    else:
      has_batching = False
    rule_rank = len(value)
    op_rank = get_rank_for_value(i)
    # The number of dimensions represented by ellipsis.
    current_batching_rank = 0
    if has_batching and op_rank >= rule_rank:
      current_batching_rank = op_rank - rule_rank
    if has_batching:
      if ellipsis_rank is None:
        ellipsis_rank = current_batching_rank
      elif ellipsis_rank != current_batching_rank:
        raise ValueError(
          "Ellipsis represents different number of leading dimensions"
          f" {ellipsis_rank} and {current_batching_rank}")
    rule_rank += current_batching_rank
    if rule_rank != op_rank:
      msg = get_message_for_value(i)
      raise ValueError(
        f"Sharding rule {msg} has rank {rule_rank}, but the operation"
        f" {msg} has rank {op_rank}")

    for j in range(current_batching_rank):
      add_batching_dim_factor(j, get_size_for_value_dim(i, j))

    for j, dim in enumerate(value):
      if isinstance(dim, str):
        add_factor(dim, get_size_for_value_dim(i, j + current_batching_rank))
      else:
        for factor in dim:
          add_factor(factor, rule.factor_sizes.get(factor, UNKNOWN))

  # Build the tensor mappings for each operand and result.
  tensor_mappings = []
  for i, mapping in enumerate(rule.operand_mappings + rule.result_mappings):
    value = tuple(mapping)
    dim_mappings = []

    if value and value[0] == BATCHING:
      value = value[1:]
      if ellipsis_rank is None:
        current_batching_rank = 0
      else:
        current_batching_rank = ellipsis_rank
    else:
      current_batching_rank = 0

    for j in range(current_batching_rank):
      dim_mappings.append(
        sdy.DimMappingAttr.get(factor_indices=[
          factors_to_indices_sizes[_get_batching_dim_factor_name(j)][0]]))

    for j, dim in enumerate(value):
      if isinstance(dim, str):
        dim_mappings.append(
          sdy.DimMappingAttr.get(
            factor_indices=[factors_to_indices_sizes[dim][0]]))
      else:
        dim_mappings.append(
          build_dim_mapping_for_compound_factors(
            i, j + current_batching_rank, dim))

    tensor_mappings.append(
      sdy.TensorMappingAttr.get(dim_mappings=dim_mappings))

  return sdy.OpShardingRuleAttr.get(
      factor_sizes=[item[1] for item in factors_to_indices_sizes.values()],
      operand_mappings=tensor_mappings[0:len(operand_types)],
      result_mappings=tensor_mappings[len(operand_types):])
