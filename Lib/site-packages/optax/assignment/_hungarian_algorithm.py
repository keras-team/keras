# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""The Hungarian algorithm for the linear assignment problem."""

import functools

import jax
import jax.numpy as jnp


def hungarian_algorithm(cost_matrix):
  r"""The Hungarian algorithm for the linear assignment problem.

  In `this problem <https://en.wikipedia.org/wiki/Linear_assignment_problem>`_,
  we are given an :math:`n \times m` cost matrix. The goal is to compute an
  assignment, i.e. a set of pairs of rows and columns, in such a way that:

  - At most one column is assigned to each row.
  - At most one row is assigned to each column.
  - The total number of assignments is :math:`\min(n, m)`.
  - The assignment minimizes the sum of costs.

  Equivalently, given a weighted complete bipartite graph, the problem is to
  find a maximum-cardinality matching that minimizes the sum of the weights of
  the edges included in the matching.

  Formally, the problem is as follows. Given :math:`C \in \mathbb{R}^{n \times m
  }`, solve the following `integer linear program <https://en.wikipedia.org/wiki
  /Integer_linear_program>`_:

  .. math::

    \begin{align*}
        \text{minimize} \quad & \sum_{i \in [n]} \sum_{j \in [m]} C_{ij} X_{ij}
        \\ \text{subject to} \quad
        & X_{ij} \in \{0, 1\} & \forall i \in [n], j \in [m] \\
        & \sum_{i \in [n]} X_{ij} \leq 1 & \forall j \in [m] \\
        & \sum_{j \in [m]} X_{ij} \leq 1 & \forall i \in [n] \\
        & \sum_{i \in [n]} \sum_{j \in [m]} X_{ij} = \min(n, m)
    \end{align*}

  The `Hungarian algorithm <https://en.wikipedia.org/wiki/Hungarian_algorithm>`_
  is a cubic-time algorithm that solves this problem.

  This implementation of the Hungarian algorithm is based on the pseudocode
  presented in pages 1685-1686 of the IEEE paper cited below.

  Args:
    cost_matrix: A matrix of costs.

  Returns:
    A pair ``(i, j)`` where ``i`` is an array of row indices and ``j`` is an
    array of column indices.
    The cost of the assignment is ``cost_matrix[i, j].sum()``.

  Examples:
    >>> import optax
    >>> from jax import numpy as jnp
    >>> cost = jnp.array(
    ...  [
    ...    [8, 4, 7],
    ...    [5, 2, 3],
    ...    [9, 6, 7],
    ...    [9, 4, 8],
    ...  ])
    >>> i, j = optax.assignment.hungarian_algorithm(cost)
    >>> print("cost:", cost[i, j].sum())
    cost: 15
    >>> print("i:", i)
    i: [0 1 3]
    >>> print("j:", j)
    j: [0 2 1]
    >>> cost = jnp.array(
    ...  [
    ...    [90, 80, 75, 70],
    ...    [35, 85, 55, 65],
    ...    [125, 95, 90, 95],
    ...    [45, 110, 95, 115],
    ...    [50, 100, 90, 100],
    ...  ])
    >>> i, j = optax.assignment.hungarian_algorithm(cost)
    >>> print("cost:", cost[i, j].sum())
    cost: 265
    >>> print("i:", i)
    i: [0 1 2 3]
    >>> print("j:", j)
    j: [3 2 1 0]

  References:
    David F. Crouse, `On implementing 2D rectangular assignment algorithms
    <https://ieeexplore.ieee.org/document/7738348>`_, 2016
  """

  if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
    return jnp.zeros(0, int), jnp.zeros(0, int)

  transpose = cost_matrix.shape[1] < cost_matrix.shape[0]

  if transpose:
    cost_matrix = cost_matrix.T

  cost_matrix = cost_matrix.astype(float)
  u = jnp.zeros(cost_matrix.shape[0], cost_matrix.dtype)
  v = jnp.zeros(cost_matrix.shape[1], cost_matrix.dtype)

  path = jnp.full(cost_matrix.shape[1], -1)
  col4row = jnp.full(cost_matrix.shape[0], -1)
  row4col = jnp.full(cost_matrix.shape[1], -1)

  init = cost_matrix, u, v, path, row4col, col4row
  cost_matrix, _, _, _, _, col4row = jax.lax.fori_loop(
      0, cost_matrix.shape[0], _lsa_body, init
  )

  if transpose:
    i = col4row.argsort()
    return col4row[i], i
  else:
    return jnp.arange(cost_matrix.shape[0]), col4row


def _find_short_augpath_while_body_inner_for(it, val):
  """Inner loop of the main loop to find augmenting path."""
  (
      remaining,
      min_value,
      costs,
      i,
      u,
      v,
      shortest_path_costs,
      path,
      lowest,
      row4col,
      index,
  ) = val

  j = remaining[it]

  r = min_value + costs[i, j] - u[i] - v[j]

  path = path.at[j].set(jnp.where(r < shortest_path_costs[j], i, path[j]))

  shortest_path_costs = shortest_path_costs.at[j].min(r)

  index = jnp.where(
      (shortest_path_costs[j] < lowest)
      | ((shortest_path_costs[j] == lowest) & (row4col[j] == -1)),
      it,
      index,
  )

  lowest = jnp.minimum(lowest, shortest_path_costs[j])

  return (
      remaining,
      min_value,
      costs,
      i,
      u,
      v,
      shortest_path_costs,
      path,
      lowest,
      row4col,
      index,
  )


def _find_short_augpath_while_body_tail(val):
  """Post-processing to find augmenting path."""
  remaining, index, row4col, sink, i, sc, num_remaining = val

  j = remaining[index]
  pred = row4col[j] == -1
  sink = jnp.where(pred, j, sink)
  i = jnp.where(pred, i, row4col[j])

  sc = sc.at[j].set(True)
  num_remaining -= 1
  remaining = remaining.at[index].set(remaining[num_remaining])

  return remaining, sink, i, sc, num_remaining


def _find_short_augpath_while_body(val):
  """Main loop to find augmenting path."""
  (
      costs,
      u,
      v,
      path,
      row4col,
      current_row,
      min_value,
      num_remaining,
      remaining,
      sr,
      sc,
      shortest_path_costs,
      sink,
  ) = val

  index = -1
  lowest = jnp.inf
  sr = sr.at[current_row].set(True)

  init = (
      remaining,
      min_value,
      costs,
      current_row,
      u,
      v,
      shortest_path_costs,
      path,
      lowest,
      row4col,
      index,
  )
  output = jax.lax.fori_loop(
      0, num_remaining, _find_short_augpath_while_body_inner_for, init
  )
  (
      remaining,
      _,
      costs,
      current_row,
      u,
      v,
      shortest_path_costs,
      path,
      lowest,
      row4col,
      index,
  ) = output

  min_value = lowest
  # infeasible costs matrix
  sink = jnp.where(min_value == jnp.inf, -1, sink)

  state = remaining, index, row4col, sink, current_row, sc, num_remaining
  (remaining, sink, current_row, sc, num_remaining) = jax.tree.map(
      functools.partial(jnp.where, sink == -1),
      _find_short_augpath_while_body_tail(state),
      (remaining, sink, current_row, sc, num_remaining),
  )

  return (
      costs,
      u,
      v,
      path,
      row4col,
      current_row,
      min_value,
      num_remaining,
      remaining,
      sr,
      sc,
      shortest_path_costs,
      sink,
  )


def _find_augmenting_path(costs, u, v, path, row4col, current_row):
  """Finds an augmenting path."""
  min_value = 0
  num_remaining = costs.shape[1]
  remaining = jnp.arange(costs.shape[1])[::-1]

  sr = jnp.zeros(costs.shape[0], bool)
  sc = jnp.zeros(costs.shape[1], bool)

  shortest_path_costs = jnp.full(costs.shape[1], jnp.inf)

  sink = -1

  init = (
      costs,
      u,
      v,
      path,
      row4col,
      current_row,
      min_value,
      num_remaining,
      remaining,
      sr,
      sc,
      shortest_path_costs,
      sink,
  )
  output = jax.lax.while_loop(
      lambda val: val[-1] == -1, _find_short_augpath_while_body, init
  )
  (
      _,
      _,
      _,
      path,
      _,
      _,
      min_value,
      _,
      _,
      sr,
      sc,
      shortest_path_costs,
      sink,
  ) = output

  return sink, min_value, sr, sc, shortest_path_costs, path


def _lsa_body(current_row, val):
  """Main loop in the Hungarian algorithm."""
  costs, u, v, path, row4col, col4row = val

  sink, min_value, sr, sc, shortest_path_costs, path = _find_augmenting_path(
      costs, u, v, path, row4col, current_row
  )

  u = u.at[current_row].add(min_value)

  mask = sr & (jnp.arange(costs.shape[0]) != current_row)
  u = jnp.where(mask, u + min_value - shortest_path_costs[col4row], u)

  v = jnp.where(sc, v + shortest_path_costs - min_value, v)

  def augment(carry):
    sink, row4col, col4row, _ = carry
    i = path[sink]
    row4col = row4col.at[sink].set(i)
    col4row, sink = col4row.at[i].set(sink), col4row[i]
    breakvar = i == current_row
    return sink, row4col, col4row, breakvar

  _, row4col, col4row, _ = jax.lax.while_loop(
      lambda val: ~val[-1], augment, (sink, row4col, col4row, False)
  )

  return costs, u, v, path, row4col, col4row
