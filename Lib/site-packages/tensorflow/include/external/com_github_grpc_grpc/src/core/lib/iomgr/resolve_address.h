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

#ifndef GRPC_CORE_LIB_IOMGR_RESOLVE_ADDRESS_H
#define GRPC_CORE_LIB_IOMGR_RESOLVE_ADDRESS_H

#include <grpc/support/port_platform.h>

#include <stddef.h>

#include "src/core/lib/iomgr/port.h"

#ifdef GRPC_UV
#include <uv.h>
#endif

#ifdef GRPC_WINSOCK_SOCKET
#include <ws2tcpip.h>
#endif

#if defined(GRPC_POSIX_SOCKET) || defined(GRPC_CFSTREAM)
#include <sys/socket.h>
#endif

#include "src/core/lib/iomgr/pollset_set.h"

#define GRPC_MAX_SOCKADDR_SIZE 128

typedef struct {
  char addr[GRPC_MAX_SOCKADDR_SIZE];
  socklen_t len;
} grpc_resolved_address;

typedef struct {
  size_t naddrs;
  grpc_resolved_address* addrs;
} grpc_resolved_addresses;

typedef struct grpc_address_resolver_vtable {
  void (*resolve_address)(const char* addr, const char* default_port,
                          grpc_pollset_set* interested_parties,
                          grpc_closure* on_done,
                          grpc_resolved_addresses** addresses);
  grpc_error* (*blocking_resolve_address)(const char* name,
                                          const char* default_port,
                                          grpc_resolved_addresses** addresses);
} grpc_address_resolver_vtable;

void grpc_set_resolver_impl(grpc_address_resolver_vtable* vtable);

/* Asynchronously resolve addr. Use default_port if a port isn't designated
   in addr, otherwise use the port in addr. */
/* TODO(apolcyn): add a timeout here */
void grpc_resolve_address(const char* addr, const char* default_port,
                          grpc_pollset_set* interested_parties,
                          grpc_closure* on_done,
                          grpc_resolved_addresses** addresses);

/* Destroy resolved addresses */
void grpc_resolved_addresses_destroy(grpc_resolved_addresses* addresses);

/* Resolve addr in a blocking fashion. On success,
   result must be freed with grpc_resolved_addresses_destroy. */
grpc_error* grpc_blocking_resolve_address(const char* name,
                                          const char* default_port,
                                          grpc_resolved_addresses** addresses);

#endif /* GRPC_CORE_LIB_IOMGR_RESOLVE_ADDRESS_H */
