// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/pass/pass.hpp"

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
#    include <filesystem>
#endif

namespace ov {
namespace pass {

/**
 * @brief Serialize transformation converts ov::Model into IR files
 * @attention
 * - dynamic shapes are not supported
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API Serialize : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("Serialize");

    enum class Version : uint8_t {
        UNSPECIFIED = 0,  // Use the latest or function version
        IR_V10 = 10,      // v10 IR
        IR_V11 = 11       // v11 IR
    };
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    Serialize(std::ostream& xmlFile, std::ostream& binFile, Version version = Version::UNSPECIFIED);

    Serialize(const std::string& xmlPath, const std::string& binPath, Version version = Version::UNSPECIFIED);

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
    Serialize(const std::filesystem::path& xmlPath,
              const std::filesystem::path& binPath,
              Version version = Version::UNSPECIFIED)
        : Serialize(xmlPath.string(), binPath.string(), version) {}
#endif

private:
    std::ostream* m_xmlFile;
    std::ostream* m_binFile;
    const std::string m_xmlPath;
    const std::string m_binPath;
    const Version m_version;
    const std::map<std::string, ov::OpSet> m_custom_opsets;
};

/**
 * @brief StreamSerialize transformation converts ov::Model into single binary stream
 * @attention
 * - dynamic shapes are not supported
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API StreamSerialize : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("StreamSerialize");

    struct DataHeader {
        size_t custom_data_offset;
        size_t custom_data_size;
        size_t consts_offset;
        size_t consts_size;
        size_t model_offset;
        size_t model_size;
    };

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    StreamSerialize(std::ostream& stream,
                    const std::function<void(std::ostream&)>& custom_data_serializer = {},
                    const std::function<std::string(const std::string&)>& cache_encrypt = {},
                    Serialize::Version version = Serialize::Version::UNSPECIFIED);

private:
    std::ostream& m_stream;
    std::function<void(std::ostream&)> m_custom_data_serializer;
    std::function<std::string(const std::string&)> m_cache_encrypt;
    const Serialize::Version m_version;
};
}  // namespace pass
}  // namespace ov
