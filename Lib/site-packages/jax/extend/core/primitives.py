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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.ad_util import stop_gradient_p as stop_gradient_p

from jax._src.core import (
  call_p as call_p,
  closed_call_p as closed_call_p
)

from jax._src.custom_derivatives import (
  custom_jvp_call_p as custom_jvp_call_p,
  custom_jvp_call_jaxpr_p as custom_jvp_call_jaxpr_p,
  custom_vjp_call_p as custom_vjp_call_p,
  custom_vjp_call_jaxpr_p as custom_vjp_call_jaxpr_p,
)

from jax._src.dispatch import device_put_p as device_put_p

from jax._src.interpreters.ad import (
  add_jaxvals_p as add_jaxvals_p,
  custom_lin_p as custom_lin_p,
  zeros_like_p as zeros_like_p,
)

from jax._src.interpreters.pxla import xla_pmap_p as xla_pmap_p

from jax._src.lax.lax import (
  abs_p as abs_p,
  acos_p as acos_p,
  acosh_p as acosh_p,
  add_p as add_p,
  after_all_p as after_all_p,
  and_p as and_p,
  argmax_p as argmax_p,
  argmin_p as argmin_p,
  asin_p as asin_p,
  asinh_p as asinh_p,
  atan_p as atan_p,
  atan2_p as atan2_p,
  atanh_p as atanh_p,
  bitcast_convert_type_p as bitcast_convert_type_p,
  broadcast_in_dim_p as broadcast_in_dim_p,
  cbrt_p as cbrt_p,
  ceil_p as ceil_p,
  clamp_p as clamp_p,
  clz_p as clz_p,
  complex_p as complex_p,
  concatenate_p as concatenate_p,
  conj_p as conj_p,
  convert_element_type_p as convert_element_type_p,
  copy_p as copy_p,
  cos_p as cos_p,
  cosh_p as cosh_p,
  create_token_p as create_token_p,
  div_p as div_p,
  dot_general_p as dot_general_p,
  eq_p as eq_p,
  eq_to_p as eq_to_p,
  exp_p as exp_p,
  exp2_p as exp2_p,
  expm1_p as expm1_p,
  floor_p as floor_p,
  ge_p as ge_p,
  gt_p as gt_p,
  imag_p as imag_p,
  infeed_p as infeed_p,
  integer_pow_p as integer_pow_p,
  iota_p as iota_p,
  is_finite_p as is_finite_p,
  le_p as le_p,
  le_to_p as le_to_p,
  log1p_p as log1p_p,
  log_p as log_p,
  logistic_p as logistic_p,
  lt_p as lt_p,
  lt_to_p as lt_to_p,
  max_p as max_p,
  min_p as min_p,
  mul_p as mul_p,
  ne_p as ne_p,
  neg_p as neg_p,
  nextafter_p as nextafter_p,
  not_p as not_p,
  or_p as or_p,
  outfeed_p as outfeed_p,
  pad_p as pad_p,
  population_count_p as population_count_p,
  pow_p as pow_p,
  real_p as real_p,
  reduce_and_p as reduce_and_p,
  reduce_max_p as reduce_max_p,
  reduce_min_p as reduce_min_p,
  reduce_or_p as reduce_or_p,
  reduce_p as reduce_p,
  reduce_precision_p as reduce_precision_p,
  reduce_prod_p as reduce_prod_p,
  reduce_sum_p as reduce_sum_p,
  reduce_xor_p as reduce_xor_p,
  rem_p as rem_p,
  reshape_p as reshape_p,
  rev_p as rev_p,
  rng_bit_generator_p as rng_bit_generator_p,
  rng_uniform_p as rng_uniform_p,
  round_p as round_p,
  rsqrt_p as rsqrt_p,
  select_n_p as select_n_p,
  shift_left_p as shift_left_p,
  shift_right_arithmetic_p as shift_right_arithmetic_p,
  shift_right_logical_p as shift_right_logical_p,
  sign_p as sign_p,
  sin_p as sin_p,
  sinh_p as sinh_p,
  sort_p as sort_p,
  sqrt_p as sqrt_p,
  square_p as square_p,
  squeeze_p as squeeze_p,
  sub_p as sub_p,
  tan_p as tan_p,
  tanh_p as tanh_p,
  top_k_p as top_k_p,
  transpose_p as transpose_p,
  xor_p as xor_p,
)

from jax._src.lax.special import (
  bessel_i0e_p as bessel_i0e_p,
  bessel_i1e_p as bessel_i1e_p,
  digamma_p as digamma_p,
  erfc_p as erfc_p,
  erf_inv_p as erf_inv_p,
  erf_p as erf_p,
  igammac_p as igammac_p,
  igamma_grad_a_p as igamma_grad_a_p,
  igamma_p as igamma_p,
  lgamma_p as lgamma_p,
  polygamma_p as polygamma_p,
  random_gamma_grad_p as random_gamma_grad_p,
  regularized_incomplete_beta_p as regularized_incomplete_beta_p,
  zeta_p as zeta_p,
)

from jax._src.lax.slicing import (
  dynamic_slice_p as dynamic_slice_p,
  dynamic_update_slice_p as dynamic_update_slice_p,
  gather_p as gather_p,
  scatter_add_p as scatter_add_p,
  scatter_max_p as scatter_max_p,
  scatter_min_p as scatter_min_p,
  scatter_mul_p as scatter_mul_p,
  scatter_p as scatter_p,
  slice_p as slice_p,
)

from jax._src.lax.convolution import (
  conv_general_dilated_p as conv_general_dilated_p,
)

from jax._src.lax.windowed_reductions import (
  reduce_window_max_p as reduce_window_max_p,
  reduce_window_min_p as reduce_window_min_p,
  reduce_window_p as reduce_window_p,
  reduce_window_sum_p as reduce_window_sum_p,
  select_and_gather_add_p as select_and_gather_add_p,
  select_and_scatter_p as select_and_scatter_p,
  select_and_scatter_add_p as select_and_scatter_add_p,
)

from jax._src.lax.control_flow import (
  cond_p as cond_p,
  cumlogsumexp_p as cumlogsumexp_p,
  cummax_p as cummax_p,
  cummin_p as cummin_p,
  cumprod_p as cumprod_p,
  cumsum_p as cumsum_p,
  linear_solve_p as linear_solve_p,
  scan_p as scan_p,
  while_p as while_p,
)

from jax._src.lax.fft import (
  fft_p as fft_p,
)

from jax._src.lax.parallel import (
  all_gather_p as all_gather_p,
  all_to_all_p as all_to_all_p,
  axis_index_p as axis_index_p,
  pmax_p as pmax_p,
  pmin_p as pmin_p,
  ppermute_p as ppermute_p,
  psum_p as psum_p,
  ragged_all_to_all_p as ragged_all_to_all_p,
)

from jax._src.lax.ann import (
  approx_top_k_p as approx_top_k_p
)

from jax._src.lax.linalg import (
  cholesky_p as cholesky_p,
  eig_p as eig_p,
  eigh_p as eigh_p,
  hessenberg_p as hessenberg_p,
  lu_p as lu_p,
  householder_product_p as householder_product_p,
  qr_p as qr_p,
  svd_p as svd_p,
  triangular_solve_p as triangular_solve_p,
  tridiagonal_p as tridiagonal_p,
  tridiagonal_solve_p as tridiagonal_solve_p,
  schur_p as schur_p,
)

from jax._src.pjit import sharding_constraint_p as sharding_constraint_p

from jax._src.prng import (
  random_bits_p as random_bits_p,
  random_fold_in_p as random_fold_in_p,
  random_seed_p as random_seed_p,
  random_split_p as random_split_p,
  threefry2x32_p as threefry2x32_p,
)

from jax._src.random import random_gamma_p as random_gamma_p
