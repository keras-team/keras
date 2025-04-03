/*
    Copyright (c) 2005-2021 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB__pipeline_filters_deduction_H
#define __TBB__pipeline_filters_deduction_H

#include "_config.h"
#include <utility>
#include <type_traits>

namespace tbb {
namespace detail {
namespace d1 {

template <typename Input, typename Output>
struct declare_fitler_types {
    using input_type = typename std::remove_const<typename std::remove_reference<Input>::type>::type;
    using output_type = typename std::remove_const<typename std::remove_reference<Output>::type>::type;
};

template <typename T> struct body_types;

template <typename T, typename Input, typename Output>
struct body_types<Output(T::*)(Input) const> : declare_fitler_types<Input, Output> {};

template <typename T, typename Input, typename Output>
struct body_types<Output(T::*)(Input)> : declare_fitler_types<Input, Output> {};

} // namespace d1
} // namespace detail
} // namespace tbb

#endif // __TBB__pipeline_filters_deduction_H
