// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type.hpp"

namespace ov {
namespace op {
/// \brief Modes for the `Pad` operator.
enum class PadMode { CONSTANT = 0, EDGE, REFLECT, SYMMETRIC };

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const PadMode& type);

/// \brief Padding Type used for `Convolution` and `Pooling`
///
/// Follows ONNX padding type definitions
/// EXPLICIT   - Pad dimensions are explicity specified
/// SAME_LOWER - Pad dimensions computed to match input shape
///              Ceil(num_dims/2) at the beginning and
///              Floor(num_dims/2) at the end
/// SAME_UPPER - Pad dimensions computed to match input shape
///              Floor(num_dims/2) at the beginning and
///              Ceil(num_dims/2) at the end
/// VALID      - No padding
/// AUTO       - Deprecated. User should not use it in the future
/// NOTSET     - Deprecated. User should not use it in the future

enum class PadType {
    EXPLICIT = 0,
    SAME_LOWER,
    SAME_UPPER,
    VALID,
    AUTO = SAME_UPPER,
    NOTSET = EXPLICIT,
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const PadType& type);

/// \brief Rounding Type used for `Pooling` operators.
enum class RoundingType {
    FLOOR = 0,
    CEIL = 1,
    CEIL_TORCH = 2,
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const RoundingType& type);

/// \brief Specifies the algorithm to use for implicit broadcasting of a tensor
///        to align with another tensor
///
/// NONE  - No implicit broadcasting of tensor
/// NUMPY - Numpy-style implicit broadcasting
///         (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
///         Right-align dimensions of the two tensors, with missing dimensions
///         treated as size 1 dimensions. After alignment, for each dimension,
///         their sizes should either match or one of them should be of size 1.
///         Size 1 dimension will be implicitly broadcast to match the other
///         size.
///
///         E.g.,
///              A: Shape(2, 1, 6)
///              B: Shape(   3, 1)
///         Result: Shape(2, 3, 6)
///
///              A: Shape(2, 1, 6)
///              B: Shape(   3, 1)
///         Result: Shape(2, 3, 6)
/// PDPD  - PaddlePaddle-style implicit broadcasting
///         (https://github.com/PaddlePaddle/Paddle/blob/release/1.5/paddle/
///                  fluid/operators/elementwise/elementwise_op.h#L126)
///         Broadcast B to match the shape of A, where axis is the start
///         dimension index to align B with A. If axis is -1 (default), i
///         axis = rank(A) - rank(B). The trailing dimensions of size 1 for B
///         will be ignored.
///
///         E.g.,
///              A: Shape(2, 3, 4, 5)
///              B: Shape(   3, 4   ) with axis =1
///         Result: Shape(2, 3, 4, 5)
///
///              A: Shape(2, 3, 4, 5)
///              B: Shape(   3, 1   ) with axis = 1
///         Result: Shape(2, 3, 4, 5)
///
enum class AutoBroadcastType {
    NONE = 0,
    EXPLICIT = NONE,
    NUMPY,
    PDPD,
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const AutoBroadcastType& type);
/// \brief BroadcastType specifies rules used for mapping of input tensor axes to output
/// shape axes.
///
/// \note  Broadcasting rules are different for Broadcast op and for element-wise ops.
///        AutoBroadcastType::NUMPY is equivalent of BroadcastType::BIDIRECTIONAL
///        according to spec.
///
/// EXPLICIT      - Mapping of the input data shape to output shape
///                 based on axes_mapping input.
/// NUMPY         - Numpy broadcasting rules, aligned with ONNX Broadcasting.
///                 (https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md)
/// PDPD          - PaddlePaddle-style implicit broadcasting.
///                 For more informaction see AutoBroadcastType documentation.
/// BIDIRECTIONAL - The broadcast rule is similar to
///                 numpy.array(input) * numpy.ones(target_shape).
///                 Dimensions are right alignment.
enum class BroadcastType { NONE, EXPLICIT = NONE, NUMPY, PDPD, BIDIRECTIONAL };

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const BroadcastType& type);

/// \brief Specifies how eps is combined with L2 value
enum class EpsMode {
    // Add bias to norm
    ADD,
    // Calculate max of norm and bias
    MAX
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const EpsMode& type);

enum class TopKSortType {
    // Returned values are not sorte
    NONE,
    // Sort result based on element indices
    SORT_INDICES,
    // Sort result based on element values
    SORT_VALUES,
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const TopKSortType& type);

enum class TopKMode {
    MAX,
    MIN,
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const TopKMode& type);

enum class PhiloxAlignment { TENSORFLOW, PYTORCH, MOCK };

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const PhiloxAlignment& alignment);

/// \brief Implicit broadcast specification
struct OPENVINO_API AutoBroadcastSpec {
    AutoBroadcastSpec() : m_type(AutoBroadcastType::NONE), m_axis(0) {}
    AutoBroadcastSpec(AutoBroadcastType type) {
        m_type = type;
        m_axis = (m_type == AutoBroadcastType::PDPD) ? -1 : 0;
    }
    AutoBroadcastSpec(const char* type) : AutoBroadcastSpec(type_from_string(type)) {}
    AutoBroadcastSpec(AutoBroadcastType type, int64_t axis) : m_type(type), m_axis(axis) {}

    AutoBroadcastType m_type;  // Implicit broadcasting algorithm
    int64_t m_axis;            // Axis to start alignment on

    bool operator==(const AutoBroadcastSpec& a) const {
        return a.m_type == m_type && a.m_axis == m_axis;
    }

    bool operator!=(const AutoBroadcastSpec& a) const {
        return !(*this == a);
    }

private:
    AutoBroadcastType type_from_string(const std::string& type) const;
};

/// \brief Implicit broadcast specification
struct OPENVINO_API BroadcastModeSpec {
    BroadcastModeSpec() : m_type(BroadcastType::NUMPY), m_axis(0) {}
    BroadcastModeSpec(BroadcastType type) {
        m_type = type;
        m_axis = (m_type == BroadcastType::PDPD) ? -1 : 0;
    }
    BroadcastModeSpec(const char* type) : BroadcastModeSpec(as_enum<BroadcastType>(type)) {}
    BroadcastModeSpec(BroadcastType type, int64_t axis) : m_type(type), m_axis(axis) {}

    BroadcastType m_type;  // Implicit broadcasting algorithm
    int64_t m_axis;        // Axis to start alignment on

    bool operator==(const BroadcastModeSpec& a) const {
        return a.m_type == m_type && a.m_axis == m_axis;
    }
};

///
/// \brief      This class defines possible recurrent sequence directions.
///
enum class RecurrentSequenceDirection { FORWARD, REVERSE, BIDIRECTIONAL };

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const RecurrentSequenceDirection& direction);
}  // namespace op

template <>
class OPENVINO_API AttributeAdapter<op::PadMode> : public EnumAttributeAdapterBase<op::PadMode> {
public:
    AttributeAdapter(op::PadMode& value) : EnumAttributeAdapterBase<op::PadMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<PadMode>");
};

template <>
class OPENVINO_API AttributeAdapter<op::PadType> : public EnumAttributeAdapterBase<op::PadType> {
public:
    AttributeAdapter(op::PadType& value) : EnumAttributeAdapterBase<op::PadType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<PadType>");
};

template <>
class OPENVINO_API AttributeAdapter<op::RoundingType> : public EnumAttributeAdapterBase<op::RoundingType> {
public:
    AttributeAdapter(op::RoundingType& value) : EnumAttributeAdapterBase<op::RoundingType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<RoundingType>");
};

template <>
class OPENVINO_API AttributeAdapter<op::AutoBroadcastType> : public EnumAttributeAdapterBase<op::AutoBroadcastType> {
public:
    AttributeAdapter(op::AutoBroadcastType& value) : EnumAttributeAdapterBase<op::AutoBroadcastType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<AutoBroadcastType>");
};

template <>
class OPENVINO_API AttributeAdapter<op::BroadcastType> : public EnumAttributeAdapterBase<op::BroadcastType> {
public:
    AttributeAdapter(op::BroadcastType& value) : EnumAttributeAdapterBase<op::BroadcastType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<BroadcastType>");
};

template <>
class OPENVINO_API AttributeAdapter<op::EpsMode> : public EnumAttributeAdapterBase<op::EpsMode> {
public:
    AttributeAdapter(op::EpsMode& value) : EnumAttributeAdapterBase<op::EpsMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<EpsMode>");
};

template <>
class OPENVINO_API AttributeAdapter<op::TopKSortType> : public EnumAttributeAdapterBase<op::TopKSortType> {
public:
    AttributeAdapter(op::TopKSortType& value) : EnumAttributeAdapterBase<op::TopKSortType>(value) {}

    OPENVINO_RTTI("AttributeAdapter<TopKSortType>");
};

template <>
class OPENVINO_API AttributeAdapter<op::TopKMode> : public EnumAttributeAdapterBase<op::TopKMode> {
public:
    AttributeAdapter(op::TopKMode& value) : EnumAttributeAdapterBase<op::TopKMode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<TopKMode>");
};

template <>
class OPENVINO_API AttributeAdapter<op::PhiloxAlignment> : public EnumAttributeAdapterBase<op::PhiloxAlignment> {
public:
    AttributeAdapter(op::PhiloxAlignment& value) : EnumAttributeAdapterBase<op::PhiloxAlignment>(value) {}

    OPENVINO_RTTI("AttributeAdapter<PhiloxAlignment>");
};

template <>
class OPENVINO_API AttributeAdapter<op::AutoBroadcastSpec> : public VisitorAdapter {
public:
    AttributeAdapter(op::AutoBroadcastSpec& value) : m_ref(value) {}
    bool visit_attributes(AttributeVisitor& visitor) override;

    OPENVINO_RTTI("AttributeAdapter<AutoBroadcastSpec>");

protected:
    op::AutoBroadcastSpec& m_ref;
};

template <>
class OPENVINO_API AttributeAdapter<op::BroadcastModeSpec> : public VisitorAdapter {
public:
    AttributeAdapter(op::BroadcastModeSpec& value) : m_ref(value) {}
    bool visit_attributes(AttributeVisitor& visitor) override;

    OPENVINO_RTTI("AttributeAdapter<BroadcastModeSpec>");

protected:
    op::BroadcastModeSpec& m_ref;
};

template <>
class OPENVINO_API AttributeAdapter<op::RecurrentSequenceDirection>
    : public EnumAttributeAdapterBase<op::RecurrentSequenceDirection> {
public:
    AttributeAdapter(op::RecurrentSequenceDirection& value)
        : EnumAttributeAdapterBase<op::RecurrentSequenceDirection>(value) {}

    OPENVINO_RTTI("AttributeAdapter<RecurrentSequenceDirection>");
};
}  // namespace ov
