//
// Copyright 2017 gRPC authors.
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

#ifndef GRPC_CORE_EXT_FILTERS_WORKAROUNDS_WORKAROUND_UTILS_H
#define GRPC_CORE_EXT_FILTERS_WORKAROUNDS_WORKAROUND_UTILS_H

#include <grpc/support/port_platform.h>

#include <grpc/support/workaround_list.h>

#include "src/core/lib/transport/metadata.h"

#define GRPC_WORKAROUND_PRIORITY_HIGH 10001
#define GRPC_WORKAROUND_PROIRITY_LOW 9999

typedef struct grpc_workaround_user_agent_md {
  bool workaround_active[GRPC_MAX_WORKAROUND_ID];
} grpc_workaround_user_agent_md;

grpc_workaround_user_agent_md* grpc_parse_user_agent(grpc_mdelem md);

typedef bool (*user_agent_parser)(grpc_mdelem);

void grpc_register_workaround(uint32_t id, user_agent_parser parser);

#endif /* GRPC_CORE_EXT_FILTERS_WORKAROUNDS_WORKAROUND_UTILS_H */
