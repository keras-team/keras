// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>
#include <set>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/rtti.hpp"

namespace ov {
/// \brief A set of axes.
class AxisSet : public std::set<size_t> {
public:
    OPENVINO_API AxisSet();

    OPENVINO_API AxisSet(const std::initializer_list<size_t>& axes);

    OPENVINO_API AxisSet(const std::set<size_t>& axes);

    OPENVINO_API AxisSet(const std::vector<size_t>& axes);

    OPENVINO_API AxisSet(const AxisSet& axes);

    OPENVINO_API AxisSet& operator=(const AxisSet& v);

    OPENVINO_API AxisSet& operator=(AxisSet&& v) noexcept;

    OPENVINO_API std::vector<int64_t> to_vector() const;
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const AxisSet& axis_set);

template <>
class OPENVINO_API AttributeAdapter<ov::AxisSet> : public ValueAccessor<std::vector<int64_t>> {
public:
    AttributeAdapter(ov::AxisSet& value) : m_ref(value) {}

    const std::vector<int64_t>& get() override;
    void set(const std::vector<int64_t>& value) override;
    operator ov::AxisSet&() {
        return m_ref;
    }
    OPENVINO_RTTI("AttributeAdapter<AxisSet>");

protected:
    ov::AxisSet& m_ref;
    std::vector<int64_t> m_buffer;
    bool m_buffer_valid{false};
};

}  // namespace ov
