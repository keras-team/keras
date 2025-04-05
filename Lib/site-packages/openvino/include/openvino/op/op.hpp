// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/node.hpp"
#include "openvino/core/rtti.hpp"

#define _OPENVINO_RTTI_OP_WITH_TYPE(TYPE_NAME) _OPENVINO_RTTI_OP_WITH_TYPE_VERSION(TYPE_NAME, "extension")

#define _OPENVINO_RTTI_OP_WITH_TYPE_VERSION(TYPE_NAME, VERSION_NAME) \
    _OPENVINO_RTTI_WITH_TYPE_VERSION_PARENT(TYPE_NAME, VERSION_NAME, ::ov::op::Op)

#define OPENVINO_OP(...)                                                                                  \
    _OPENVINO_RTTI_EXPAND(_OPENVINO_RTTI_DEFINITION_SELECTOR_3(__VA_ARGS__,                               \
                                                               _OPENVINO_RTTI_WITH_TYPE_VERSION_PARENT,   \
                                                               _OPENVINO_RTTI_OP_WITH_TYPE_VERSION,       \
                                                               _OPENVINO_RTTI_OP_WITH_TYPE)(__VA_ARGS__)) \
    /* Add accessibility for Op to the method: evaluate from the Base class                               \
     Usually C++ allows to use virtual methods of Base class from Derived class but if they have          \
     the same name and not all of them are overrided in Derived class, the only overrided methods         \
     will be available from Derived class. We need to explicitly cast Derived to Base class to            \
     have an access to remaining methods or use this using. */                                            \
    using ov::op::Op::evaluate;                                                                           \
    using ov::op::Op::evaluate_lower;                                                                     \
    using ov::op::Op::evaluate_upper;

namespace ov {
namespace op {
/// \brief Root of all actual ops
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Op : public Node {
protected:
    Op() : Node() {}
    Op(const OutputVector& arguments);

public:
    _OPENVINO_HIDDEN_METHOD static const ::ov::Node::type_info_t& get_type_info_static() {
        static ::ov::Node::type_info_t info{"Op", "util"};
        info.hash();
        return info;
    }
    const ::ov::Node::type_info_t& get_type_info() const override {
        return get_type_info_static();
    }
};
}  // namespace op
}  // namespace ov
