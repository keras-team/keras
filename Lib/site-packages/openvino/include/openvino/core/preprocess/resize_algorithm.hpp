// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace preprocess {

/// \brief An enum containing all supported resize(interpolation) algorithms available in preprocessing
enum class ResizeAlgorithm {
    RESIZE_LINEAR,           //!< Linear interpolation matching the TensorFlow behavior
    RESIZE_CUBIC,            //!< Cubic interpolation
    RESIZE_NEAREST,          //!< Nearest interpolation
    RESIZE_BILINEAR_PILLOW,  //!< Bilinear interpolation matching the Pillow behavior
    RESIZE_BICUBIC_PILLOW    //!< Bicubic interpolation matching the Pillow behavior
};

}  // namespace preprocess
}  // namespace ov
