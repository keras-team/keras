// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
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

#ifndef GEMMLOWP_META_OPERATIONS_COMMON_H_
#define GEMMLOWP_META_OPERATIONS_COMMON_H_

class Quantized8BitOperation {
 public:
  Quantized8BitOperation(std::int32_t lhs_offset, std::int32_t rhs_offset,
                         std::int32_t sum_offset, std::int32_t multiplier,
                         std::int32_t shift)
      : lhs_offset(lhs_offset),
        rhs_offset(rhs_offset),
        sum_offset(sum_offset),
        multiplier(multiplier),
        shift(shift) {}

 protected:
  std::int32_t lhs_offset;
  std::int32_t rhs_offset;
  std::int32_t sum_offset;
  std::int32_t multiplier;
  std::int32_t shift;
};

class FloatOperation {
 public:
  FloatOperation(std::int32_t lhs_offset, std::int32_t rhs_offset,
                 float result_offset)
      : lhs_offset(lhs_offset),
        rhs_offset(rhs_offset),
        result_offset(result_offset) {}

 protected:
  std::int32_t lhs_offset;
  std::int32_t rhs_offset;
  float result_offset;
};

class Int32Operation {
 public:
  Int32Operation(std::int32_t lhs_offset, std::int32_t rhs_offset)
      : lhs_offset(lhs_offset), rhs_offset(rhs_offset) {}

 protected:
  std::int32_t lhs_offset;
  std::int32_t rhs_offset;
};

#endif  // GEMMLOWP_META_OPERATIONS_COMMON_H_
