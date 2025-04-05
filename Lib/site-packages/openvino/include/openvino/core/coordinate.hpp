// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
/// \brief Coordinates for a tensor element
class Coordinate : public std::vector<size_t> {
public:
    OPENVINO_API Coordinate();
    OPENVINO_API Coordinate(const std::initializer_list<size_t>& axes);

    OPENVINO_API Coordinate(const Shape& shape);

    OPENVINO_API Coordinate(const std::vector<size_t>& axes);

    OPENVINO_API Coordinate(const Coordinate& axes);

    OPENVINO_API Coordinate(size_t n, size_t initial_value = 0);

    OPENVINO_API ~Coordinate();

    template <class InputIterator>
    Coordinate(InputIterator first, InputIterator last) : std::vector<size_t>(first, last) {}

    OPENVINO_API Coordinate& operator=(const Coordinate& v);

    OPENVINO_API Coordinate& operator=(Coordinate&& v) noexcept;
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const Coordinate& coordinate);

template <>
class OPENVINO_API AttributeAdapter<Coordinate> : public IndirectVectorValueAccessor<Coordinate, std::vector<int64_t>> {
public:
    AttributeAdapter(Coordinate& value) : IndirectVectorValueAccessor<Coordinate, std::vector<int64_t>>(value) {}

    OPENVINO_RTTI("AttributeAdapter<Coordinate>");
};
}  // namespace ov
