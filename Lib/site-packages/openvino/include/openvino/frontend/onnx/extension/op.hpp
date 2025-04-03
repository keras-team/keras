// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/onnx/extension/conversion.hpp"

namespace ov {
namespace frontend {
namespace onnx {

template <typename OVOpType = void>
class OpExtension : public ConversionExtension {
public:
    OpExtension(const std::map<std::string, std::string>& attr_names_map = {},
                const std::map<std::string, ov::Any>& attr_values_map = {})
        : OpExtension(OVOpType::get_type_info_static().name, "", attr_names_map, attr_values_map) {}

    OpExtension(const std::string& fw_type_name,
                const std::map<std::string, std::string>& attr_names_map = {},
                const std::map<std::string, ov::Any>& attr_values_map = {})
        : OpExtension(fw_type_name, "", attr_names_map, attr_values_map) {}

    OpExtension(const std::string& fw_type_name,
                const std::string& fw_domain,
                const std::map<std::string, std::string>& attr_names_map = {},
                const std::map<std::string, ov::Any>& attr_values_map = {})
        : ConversionExtension(fw_type_name,
                              fw_domain,
                              OpConversionFunction(
                                  []() {
                                      return std::make_shared<OVOpType>();
                                  },
                                  attr_names_map,
                                  attr_values_map)) {}
};

template <>
class OpExtension<void> : public ConversionExtension {
public:
    OpExtension() = delete;

    explicit OpExtension(const std::string& fw_ov_type_name,
                         const std::map<std::string, std::string>& attr_names_map = {},
                         const std::map<std::string, ov::Any>& attr_values_map = {})
        : OpExtension(fw_ov_type_name, fw_ov_type_name, attr_names_map, attr_values_map) {}

    OpExtension(const std::string& ov_type_name,
                const std::string& fw_type_name,
                const std::map<std::string, std::string>& attr_names_map = {},
                const std::map<std::string, ov::Any>& attr_values_map = {})
        : OpExtension(ov_type_name, fw_type_name, "", attr_names_map, attr_values_map) {}

    OpExtension(const std::string& ov_type_name,
                const std::string& fw_type_name,
                const std::string& fw_domain_name,
                const std::map<std::string, std::string>& attr_names_map = {},
                const std::map<std::string, ov::Any>& attr_values_map = {})
        : ConversionExtension(fw_type_name,
                              fw_domain_name,
                              OpConversionFunction(
                                  [ov_type_name]() {
                                      return create_ov_node_by_name(ov_type_name);
                                  },
                                  attr_names_map,
                                  attr_values_map)) {}
};

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
