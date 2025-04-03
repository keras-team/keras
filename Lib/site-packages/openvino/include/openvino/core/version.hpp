// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <map>
#include <ostream>

#include "openvino/core/core_visibility.hpp"

/**
 * @def OPENVINO_VERSION_MAJOR
 * @brief Defines OpenVINO major version
 *
 * @def OPENVINO_VERSION_MINOR
 * @brief Defines OpenVINO minor version
 *
 * @def OPENVINO_VERSION_PATCH
 * @brief Defines OpenVINO patch version
 */

#define OPENVINO_VERSION_MAJOR 2025
#define OPENVINO_VERSION_MINOR 0
#define OPENVINO_VERSION_PATCH 0

namespace ov {

/**
 * @struct Version
 * @brief  Represents version information that describes plugins and the OpemVINO library
 */
#pragma pack(push, 1)
struct Version {
    /**
     * @brief A null terminated string with build number
     */
    const char* buildNumber;

    /**
     * @brief A null terminated description string
     */
    const char* description;
};
#pragma pack(pop)

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const Version& version);

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const std::map<std::string, Version>& versions);

/**
 * @brief Gets the current OpenVINO version
 * @return The current OpenVINO version
 */
OPENVINO_API_C(const Version) get_openvino_version() noexcept;

}  // namespace ov
