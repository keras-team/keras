//
// Copyright 2019 gRPC authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_BACKEND_METRIC_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_BACKEND_METRIC_H

#include <grpc/support/port_platform.h>

#include <grpc/slice.h>

#include "src/core/ext/filters/client_channel/lb_policy.h"
#include "src/core/lib/gprpp/arena.h"

namespace grpc_core {

// Parses the serialized load report and allocates a BackendMetricData
// object on the arena.
const LoadBalancingPolicy::BackendMetricData* ParseBackendMetricData(
    const grpc_slice& serialized_load_report, Arena* arena);

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_BACKEND_METRIC_H */
