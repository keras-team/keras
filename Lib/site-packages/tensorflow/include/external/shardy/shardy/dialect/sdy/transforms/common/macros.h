/* Copyright 2024 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_MACROS_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_MACROS_H_

// Macro to assign value from std::optional<T> or return std::nullopt.
#define SDY_ASSIGN_OR_RETURN_IF_NULLOPT(lhs, expr)                           \
  SDY_ASSIGN_OR_RETURN_IF_NULLOPT_IMPL(CONCAT_(_expr_result, __LINE__), lhs, \
                                       expr)

// =================================================================
// == Implementation details, do not rely on anything below here. ==
// =================================================================

#define CONCAT_INNER_(x, y) x##y
#define CONCAT_(x, y) CONCAT_INNER_(x, y)

#define SDY_ASSIGN_OR_RETURN_IF_NULLOPT_IMPL(result, lhs, expr) \
  auto result = expr;                                           \
  if (!result.has_value()) {                                    \
    return std::nullopt;                                        \
  }                                                             \
  lhs = std::move(result).value();

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_MACROS_H_
