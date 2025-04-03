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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.numpy.linalg import (
  cholesky as cholesky,
  cond as cond,
  cross as cross,
  det as det,
  diagonal as diagonal,
  eig as eig,
  eigh as eigh,
  eigvals as eigvals,
  eigvalsh as eigvalsh,
  inv as inv,
  lstsq as lstsq,
  matmul as matmul,
  matrix_norm as matrix_norm,
  matrix_power as matrix_power,
  matrix_rank as matrix_rank,
  matrix_transpose as matrix_transpose,
  multi_dot as multi_dot,
  norm as norm,
  outer as outer,
  pinv as pinv,
  qr as qr,
  slogdet as slogdet,
  solve as solve,
  svd as svd,
  svdvals as svdvals,
  tensordot as tensordot,
  tensorinv as tensorinv,
  tensorsolve as tensorsolve,
  trace as trace,
  vector_norm as vector_norm,
  vecdot as vecdot,
)
