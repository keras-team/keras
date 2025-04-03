// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct xnn_experiment_config {
  int dummy;  // C requires that a struct or union has at least one member
};

struct xnn_experiment_config* xnn_get_experiment_config();

void xnn_experiment_enable_adaptive_avx_optimization();


#ifdef __cplusplus
}  // extern "C"
#endif
