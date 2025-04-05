# Copyright 2021 The JAX Authors.
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

"""
.. currentmodule:: jax.experimental.sparse

The :mod:`jax.experimental.sparse` module includes experimental support for sparse matrix
operations in JAX. It is under active development, and the API is subject to change. The
primary interfaces made available are the :class:`BCOO` sparse array type, and the
:func:`sparsify` transform.

Batched-coordinate (BCOO) sparse matrices
-----------------------------------------
The main high-level sparse object currently available in JAX is the :class:`BCOO`,
or *batched coordinate* sparse array, which offers a compressed storage format compatible
with JAX transformations, in particular JIT (e.g. :func:`jax.jit`), batching
(e.g. :func:`jax.vmap`) and autodiff (e.g. :func:`jax.grad`).

Here is an example of creating a sparse array from a dense array:

    >>> from jax.experimental import sparse
    >>> import jax.numpy as jnp
    >>> import numpy as np

    >>> M = jnp.array([[0., 1., 0., 2.],
    ...                [3., 0., 0., 0.],
    ...                [0., 0., 4., 0.]])

    >>> M_sp = sparse.BCOO.fromdense(M)

    >>> M_sp
    BCOO(float32[3, 4], nse=4)

Convert back to a dense array with the ``todense()`` method:

    >>> M_sp.todense()
    Array([[0., 1., 0., 2.],
           [3., 0., 0., 0.],
           [0., 0., 4., 0.]], dtype=float32)

The BCOO format is a somewhat modified version of the standard COO format, and the dense
representation can be seen in the ``data`` and ``indices`` attributes:

    >>> M_sp.data  # Explicitly stored data
    Array([1., 2., 3., 4.], dtype=float32)

    >>> M_sp.indices # Indices of the stored data
    Array([[0, 1],
           [0, 3],
           [1, 0],
           [2, 2]], dtype=int32)

BCOO objects have familiar array-like attributes, as well as sparse-specific attributes:

    >>> M_sp.ndim
    2

    >>> M_sp.shape
    (3, 4)

    >>> M_sp.dtype
    dtype('float32')

    >>> M_sp.nse  # "number of specified elements"
    4

BCOO objects also implement a number of array-like methods, to allow you to use them
directly within jax programs. For example, here we compute the transposed matrix-vector
product:

    >>> y = jnp.array([3., 6., 5.])

    >>> M_sp.T @ y
    Array([18.,  3., 20.,  6.], dtype=float32)

    >>> M.T @ y  # Compare to dense version
    Array([18.,  3., 20.,  6.], dtype=float32)

BCOO objects are designed to be compatible with JAX transforms, including :func:`jax.jit`,
:func:`jax.vmap`, :func:`jax.grad`, and others. For example:

    >>> from jax import grad, jit

    >>> def f(y):
    ...   return (M_sp.T @ y).sum()
    ...
    >>> jit(grad(f))(y)
    Array([3., 3., 4.], dtype=float32)

Note, however, that under normal circumstances :mod:`jax.numpy` and :mod:`jax.lax` functions
do not know how to handle sparse matrices, so attempting to compute things like
``jnp.dot(M_sp.T, y)`` will result in an error (however, see the next section).

Sparsify transform
------------------
An overarching goal of the JAX sparse implementation is to provide a means to switch from
dense to sparse computation seamlessly, without having to modify the dense implementation.
This sparse experiment accomplishes this through the :func:`sparsify` transform.

Consider this function, which computes a more complicated result from a matrix and a vector input:

    >>> def f(M, v):
    ...   return 2 * jnp.dot(jnp.log1p(M.T), v) + 1
    ...
    >>> f(M, y)
    Array([17.635532,  5.158883, 17.09438 ,  7.591674], dtype=float32)

Were we to pass a sparse matrix to this directly, it would result in an error, because ``jnp``
functions do not recognize sparse inputs. However, with :func:`sparsify`, we get a version of
this function that does accept sparse matrices:

    >>> f_sp = sparse.sparsify(f)

    >>> f_sp(M_sp, y)
    Array([17.635532,  5.158883, 17.09438 ,  7.591674], dtype=float32)

Support for :func:`sparsify` includes a large number of the most common primitives, including:

- generalized (batched) matrix products & einstein summations (:obj:`~jax.lax.dot_general_p`)
- zero-preserving elementwise binary operations (e.g. :obj:`~jax.lax.add_p`, :obj:`~jax.lax.mul_p`, etc.)
- zero-preserving elementwise unary operations (e.g. :obj:`~jax.lax.abs_p`, :obj:`jax.lax.neg_p`, etc.)
- summation reductions (:obj:`~jax.lax.reduce_sum_p`)
- general indexing operations (:obj:`~jax.lax.slice_p`, `lax.dynamic_slice_p`, `lax.gather_p`)
- concatenation and stacking (:obj:`~jax.lax.concatenate_p`)
- transposition & reshaping ((:obj:`~jax.lax.transpose_p`, :obj:`~jax.lax.reshape_p`,
  :obj:`~jax.lax.squeeze_p`, :obj:`~jax.lax.broadcast_in_dim_p`)
- some higher-order functions (:obj:`~jax.lax.cond_p`, :obj:`~jax.lax.while_p`, :obj:`~jax.lax.scan_p`)
- some simple 1D convolutions (:obj:`~jax.lax.conv_general_dilated_p`)

Nearly any :mod:`jax.numpy` function that lowers to these supported primitives can be used
within a sparsify transform to operate on sparse arrays. This set of primitives is enough
to enable relatively sophisticated sparse workflows, as the next section will show.

Example: sparse logistic regression
-----------------------------------
As an example of a more complicated sparse workflow, let's consider a simple logistic regression
implemented in JAX. Notice that the following implementation has no reference to sparsity:

    >>> import functools
    >>> from sklearn.datasets import make_classification
    >>> from jax.scipy import optimize

    >>> def sigmoid(x):
    ...   return 0.5 * (jnp.tanh(x / 2) + 1)
    ...
    >>> def y_model(params, X):
    ...   return sigmoid(jnp.dot(X, params[1:]) + params[0])
    ...
    >>> def loss(params, X, y):
    ...   y_hat = y_model(params, X)
    ...   return -jnp.mean(y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat))
    ...
    >>> def fit_logreg(X, y):
    ...   params = jnp.zeros(X.shape[1] + 1)
    ...   result = optimize.minimize(functools.partial(loss, X=X, y=y),
    ...                              x0=params, method='BFGS')
    ...   return result.x

    >>> X, y = make_classification(n_classes=2, random_state=1701)
    >>> params_dense = fit_logreg(X, y)
    >>> print(params_dense)  # doctest: +SKIP
    [-0.7298445   0.29893667  1.0248291  -0.44436368  0.8785025  -0.7724008
     -0.62893456  0.2934014   0.82974285  0.16838408 -0.39774987 -0.5071844
      0.2028872   0.5227761  -0.3739224  -0.7104083   2.4212713   0.6310087
     -0.67060554  0.03139788 -0.05359547]

This returns the best-fit parameters of a dense logistic regression problem.
To fit the same model on sparse data, we can apply the :func:`sparsify` transform:

    >>> Xsp = sparse.BCOO.fromdense(X)  # Sparse version of the input
    >>> fit_logreg_sp = sparse.sparsify(fit_logreg)  # Sparse-transformed fit function
    >>> params_sparse = fit_logreg_sp(Xsp, y)
    >>> print(params_sparse)  # doctest: +SKIP
    [-0.72971725  0.29878938  1.0246326  -0.44430563  0.8784217  -0.77225566
     -0.6288222   0.29335397  0.8293481   0.16820715 -0.39764675 -0.5069753
      0.202579    0.522672   -0.3740134  -0.7102678   2.4209507   0.6310593
     -0.670236    0.03132951 -0.05356663]
"""

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax.experimental.sparse.ad import (
    jacfwd as jacfwd,
    jacobian as jacobian,
    jacrev as jacrev,
    grad as grad,
    value_and_grad as value_and_grad,
)
from jax.experimental.sparse.bcoo import (
    bcoo_broadcast_in_dim as bcoo_broadcast_in_dim,
    bcoo_concatenate as bcoo_concatenate,
    bcoo_conv_general_dilated as bcoo_conv_general_dilated,
    bcoo_dot_general as bcoo_dot_general,
    bcoo_dot_general_p as bcoo_dot_general_p,
    bcoo_dot_general_sampled as bcoo_dot_general_sampled,
    bcoo_dot_general_sampled_p as bcoo_dot_general_sampled_p,
    bcoo_dynamic_slice as bcoo_dynamic_slice,
    bcoo_extract as bcoo_extract,
    bcoo_extract_p as bcoo_extract_p,
    bcoo_fromdense as bcoo_fromdense,
    bcoo_fromdense_p as bcoo_fromdense_p,
    bcoo_gather as bcoo_gather,
    bcoo_multiply_dense as bcoo_multiply_dense,
    bcoo_multiply_sparse as bcoo_multiply_sparse,
    bcoo_update_layout as bcoo_update_layout,
    bcoo_reduce_sum as bcoo_reduce_sum,
    bcoo_reshape as bcoo_reshape,
    bcoo_rev as bcoo_rev,
    bcoo_slice as bcoo_slice,
    bcoo_sort_indices as bcoo_sort_indices,
    bcoo_sort_indices_p as bcoo_sort_indices_p,
    bcoo_spdot_general_p as bcoo_spdot_general_p,
    bcoo_squeeze as bcoo_squeeze,
    bcoo_sum_duplicates as bcoo_sum_duplicates,
    bcoo_sum_duplicates_p as bcoo_sum_duplicates_p,
    bcoo_todense as bcoo_todense,
    bcoo_todense_p as bcoo_todense_p,
    bcoo_transpose as bcoo_transpose,
    bcoo_transpose_p as bcoo_transpose_p,
    BCOO as BCOO,
)

from jax.experimental.sparse.bcsr import (
    bcsr_broadcast_in_dim as bcsr_broadcast_in_dim,
    bcsr_concatenate as bcsr_concatenate,
    bcsr_dot_general as bcsr_dot_general,
    bcsr_dot_general_p as bcsr_dot_general_p,
    bcsr_extract as bcsr_extract,
    bcsr_extract_p as bcsr_extract_p,
    bcsr_fromdense as bcsr_fromdense,
    bcsr_fromdense_p as bcsr_fromdense_p,
    bcsr_sum_duplicates as bcsr_sum_duplicates,
    bcsr_todense as bcsr_todense,
    bcsr_todense_p as bcsr_todense_p,
    BCSR as BCSR,
)

from jax.experimental.sparse._base import (
    JAXSparse as JAXSparse
)

from jax.experimental.sparse.api import (
    empty as empty,
    eye as eye,
    todense as todense,
    todense_p as todense_p,
)

from jax.experimental.sparse.util import (
    CuSparseEfficiencyWarning as CuSparseEfficiencyWarning,
    SparseEfficiencyError as SparseEfficiencyError,
    SparseEfficiencyWarning as SparseEfficiencyWarning,
)

from jax.experimental.sparse.coo import (
    coo_fromdense as coo_fromdense,
    coo_fromdense_p as coo_fromdense_p,
    coo_matmat as coo_matmat,
    coo_matmat_p as coo_matmat_p,
    coo_matvec as coo_matvec,
    coo_matvec_p as coo_matvec_p,
    coo_todense as coo_todense,
    coo_todense_p as coo_todense_p,
    COO as COO,
)

from jax.experimental.sparse.csr import (
    csr_fromdense as csr_fromdense,
    csr_fromdense_p as csr_fromdense_p,
    csr_matmat as csr_matmat,
    csr_matmat_p as csr_matmat_p,
    csr_matvec as csr_matvec,
    csr_matvec_p as csr_matvec_p,
    csr_todense as csr_todense,
    csr_todense_p as csr_todense_p,
    CSC as CSC,
    CSR as CSR,
)

from jax.experimental.sparse.random import random_bcoo as random_bcoo
from jax.experimental.sparse.transform import (
    sparsify as sparsify,
    SparseTracer as SparseTracer,
)

from jax.experimental.sparse import linalg as linalg
