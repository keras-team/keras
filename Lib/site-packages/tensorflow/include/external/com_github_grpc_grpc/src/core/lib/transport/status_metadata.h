/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GRPC_CORE_LIB_TRANSPORT_STATUS_METADATA_H
#define GRPC_CORE_LIB_TRANSPORT_STATUS_METADATA_H

#include <grpc/support/port_platform.h>

#include <grpc/status.h>

#include "src/core/lib/transport/metadata.h"
#include "src/core/lib/transport/static_metadata.h"

grpc_status_code grpc_get_status_code_from_metadata(grpc_mdelem md);

/** Get a grpc_mdelem of grpc-status: X where X is the numeric value of
    status_code.

    The returned elem is owned by the caller. */
grpc_mdelem grpc_get_reffed_status_elem_slowpath(int status_code);
inline grpc_mdelem grpc_get_reffed_status_elem(int status_code) {
  switch (status_code) {
    case 0:
      return GRPC_MDELEM_GRPC_STATUS_0;
    case 1:
      return GRPC_MDELEM_GRPC_STATUS_1;
    case 2:
      return GRPC_MDELEM_GRPC_STATUS_2;
  }
  return grpc_get_reffed_status_elem_slowpath(status_code);
}

#endif /* GRPC_CORE_LIB_TRANSPORT_STATUS_METADATA_H */
