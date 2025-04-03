# Copyright 2018 The JAX Authors.
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

from jax.scipy.stats import bernoulli as bernoulli
from jax.scipy.stats import beta as beta
from jax.scipy.stats import binom as binom
from jax.scipy.stats import cauchy as cauchy
from jax.scipy.stats import dirichlet as dirichlet
from jax.scipy.stats import expon as expon
from jax.scipy.stats import gamma as gamma
from jax.scipy.stats import geom as geom
from jax.scipy.stats import laplace as laplace
from jax.scipy.stats import logistic as logistic
from jax.scipy.stats import multinomial as multinomial
from jax.scipy.stats import multivariate_normal as multivariate_normal
from jax.scipy.stats import nbinom as nbinom
from jax.scipy.stats import norm as norm
from jax.scipy.stats import pareto as pareto
from jax.scipy.stats import poisson as poisson
from jax.scipy.stats import t as t
from jax.scipy.stats import uniform as uniform
from jax.scipy.stats import chi2 as chi2
from jax.scipy.stats import betabinom as betabinom
from jax.scipy.stats import gennorm as gennorm
from jax.scipy.stats import truncnorm as truncnorm
from jax._src.scipy.stats.kde import gaussian_kde as gaussian_kde
from jax._src.scipy.stats._core import mode as mode, rankdata as rankdata, sem as sem
from jax.scipy.stats import vonmises as vonmises
from jax.scipy.stats import wrapcauchy as wrapcauchy
