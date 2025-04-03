// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace preprocess {

/// \brief Color format enumeration for conversion
enum class ColorFormat {
    UNDEFINED,          //!< Undefined color format
    NV12_SINGLE_PLANE,  //!< Image in NV12 format represented as separate tensors for Y and UV planes.
    NV12_TWO_PLANES,    //!< Image in NV12 format represented as separate tensors for Y and UV planes.
    I420_SINGLE_PLANE,  //!< Image in I420 (YUV) format as single tensor
    I420_THREE_PLANES,  //!< Image in I420 format represented as separate tensors for Y, U and V planes.
    RGB,                //!< Image in RGB interleaved format (3 channels)
    BGR,                //!< Image in BGR interleaved format (3 channels)
    GRAY,               //!< Image in GRAY format (1 channel)
    RGBX,               //!< Image in RGBX interleaved format (4 channels)
    BGRX                //!< Image in BGRX interleaved format (4 channels)
};

}  // namespace preprocess
}  // namespace ov
