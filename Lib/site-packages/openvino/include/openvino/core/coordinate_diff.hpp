// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/rtti.hpp"

namespace ov {
/// \brief A difference (signed) of tensor element coordinates.
class CoordinateDiff : public std::vector<std::ptrdiff_t> {
public:
    OPENVINO_API CoordinateDiff(const std::initializer_list<std::ptrdiff_t>& diffs);

    OPENVINO_API CoordinateDiff(const std::vector<std::ptrdiff_t>& diffs);

    OPENVINO_API CoordinateDiff(const CoordinateDiff& diffs);

    OPENVINO_API explicit CoordinateDiff(size_t n, std::ptrdiff_t initial_value = 0);

    template <class InputIterator>
    CoordinateDiff(InputIterator first, InputIterator last) : std::vector<std::ptrdiff_t>(first, last) {}

    OPENVINO_API ~CoordinateDiff();

    OPENVINO_API CoordinateDiff();

    OPENVINO_API CoordinateDiff& operator=(const CoordinateDiff& v);

    OPENVINO_API CoordinateDiff& operator=(CoordinateDiff&& v) noexcept;
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const CoordinateDiff& coordinate_diff);

template <>
class OPENVINO_API AttributeAdapter<CoordinateDiff>
    : public IndirectVectorValueAccessor<CoordinateDiff, std::vector<int64_t>>

{
public:
    AttributeAdapter(CoordinateDiff& value)
        : IndirectVectorValueAccessor<CoordinateDiff, std::vector<int64_t>>(value) {}

    OPENVINO_RTTI("AttributeAdapter<CoordinateDiff>");
};

}  // namespace ov
