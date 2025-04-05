/*
 *
 * Copyright 2015 gRPC authors.
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

#ifndef GRPC_CORE_LIB_IOMGR_TCP_CLIENT_H
#define GRPC_CORE_LIB_IOMGR_TCP_CLIENT_H

#include <grpc/support/port_platform.h>

#include <grpc/impl/codegen/grpc_types.h>
#include <grpc/support/time.h>
#include "src/core/lib/iomgr/endpoint.h"
#include "src/core/lib/iomgr/pollset_set.h"
#include "src/core/lib/iomgr/resolve_address.h"

typedef struct grpc_tcp_client_vtable {
  void (*connect)(grpc_closure* on_connect, grpc_endpoint** endpoint,
                  grpc_pollset_set* interested_parties,
                  const grpc_channel_args* channel_args,
                  const grpc_resolved_address* addr, grpc_millis deadline);
} grpc_tcp_client_vtable;

/* Asynchronously connect to an address (specified as (addr, len)), and call
   cb with arg and the completed connection when done (or call cb with arg and
   NULL on failure).
   interested_parties points to a set of pollsets that would be interested
   in this connection being established (in order to continue their work) */
void grpc_tcp_client_connect(grpc_closure* on_connect, grpc_endpoint** endpoint,
                             grpc_pollset_set* interested_parties,
                             const grpc_channel_args* channel_args,
                             const grpc_resolved_address* addr,
                             grpc_millis deadline);

void grpc_tcp_client_global_init();

void grpc_set_tcp_client_impl(grpc_tcp_client_vtable* impl);

#endif /* GRPC_CORE_LIB_IOMGR_TCP_CLIENT_H */
