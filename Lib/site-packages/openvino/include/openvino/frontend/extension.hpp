// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Includes all extensions.
//

#pragma once

#ifdef IN_OV_CORE_LIBRARY
#    error("extension.hpp is for external use only")
#endif

#include "extension/conversion.hpp"
#include "extension/decoder_transformation.hpp"
#include "extension/op.hpp"
#include "extension/progress_reporter.hpp"
#include "extension/telemetry.hpp"
