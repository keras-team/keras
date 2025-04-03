/*
 *
 * Copyright 2018 gRPC authors.
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

#ifndef GRPC_CORE_LIB_IOMGR_RESOLVE_ADDRESS_CUSTOM_H
#define GRPC_CORE_LIB_IOMGR_RESOLVE_ADDRESS_CUSTOM_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/iomgr/port.h"

#include "src/core/lib/iomgr/resolve_address.h"
#include "src/core/lib/iomgr/sockaddr.h"

typedef struct grpc_custom_resolver grpc_custom_resolver;

typedef struct grpc_custom_resolver_vtable {
  grpc_error* (*resolve)(char* host, char* port, grpc_resolved_addresses** res);
  void (*resolve_async)(grpc_custom_resolver* resolver, char* host, char* port);
} grpc_custom_resolver_vtable;

void grpc_custom_resolve_callback(grpc_custom_resolver* resolver,
                                  grpc_resolved_addresses* result,
                                  grpc_error* error);

/* Internal APIs */
void grpc_custom_resolver_init(grpc_custom_resolver_vtable* impl);

#endif /* GRPC_CORE_LIB_IOMGR_RESOLVE_ADDRESS_CUSTOM_H */
