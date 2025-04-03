// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GEMMLOWP_META_SINGLE_THREAD_TRANSFORM_H_
#define GEMMLOWP_META_SINGLE_THREAD_TRANSFORM_H_

#include <iostream>
#include "base.h"

namespace gemmlowp {
namespace meta {

template <typename Params, int kernel_size>
void Transform1D(const Params& params);

namespace internal {

class Transform1DExecutor {
 public:
  template <typename P, int kernel_size, int leftovers>
  static void ExecuteDispatch1D(const P& params) {
    Transform1DKernel<typename P::InType, typename P::OutType,
                      typename P::Kernel, kernel_size,
                      leftovers>::Transform(params.input, params.kernel,
                                            params.output);
  }
};

template <typename E, typename P, int kernel_size, int variable_leftovers>
struct Dispatch1D {
  static void Execute(const P& params, int leftovers) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "Dispatch(1): " << kernel_size << ":" << variable_leftovers
              << std::endl
              << std::flush;
#endif
#endif
    if (leftovers == variable_leftovers) {
      E::template ExecuteDispatch1D<P, kernel_size, variable_leftovers>(params);
    } else {
      Dispatch1D<E, P, kernel_size, variable_leftovers - 1>::Execute(params,
                                                                     leftovers);
    }
  }
};

template <typename E, typename P, int kernel_size>
struct Dispatch1D<E, P, kernel_size, 0> {
  static void Execute(const P& params, int leftovers) {
#ifdef DEBUG
#ifdef DEBUG_METAGEMM_VERBOSE
    std::cout << "Dispatch(1): " << kernel_size << ": 0" << std::endl
              << std::flush;
#endif
#endif
    if (leftovers == 0) {
      E::template ExecuteDispatch1D<P, kernel_size, 0>(params);
    } else {
      std::cerr << "FATAL: dispatch1D failed: ran out of cases." << std::endl
                << std::flush;
      std::exit(1);
    }
  }
};

}  // namespace internal

template <typename Params, int kernel_size>
inline void Transform1D(const Params& params) {
  internal::Dispatch1D<internal::Transform1DExecutor, Params, kernel_size,
                       kernel_size - 1>::Execute(params, params.kernel.count %
                                                             kernel_size);
}

}  // namespace meta
}  // namespace gemmlowp

#endif  // GEMMLOWP_META_SINGLE_THREAD_TRANSFORM_H_
