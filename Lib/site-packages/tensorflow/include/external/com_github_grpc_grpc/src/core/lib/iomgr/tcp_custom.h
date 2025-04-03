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

#ifndef GRPC_CORE_LIB_IOMGR_TCP_CUSTOM_H
#define GRPC_CORE_LIB_IOMGR_TCP_CUSTOM_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/iomgr/endpoint.h"
#include "src/core/lib/iomgr/sockaddr.h"

typedef struct grpc_tcp_listener grpc_tcp_listener;
typedef struct grpc_custom_tcp_connect grpc_custom_tcp_connect;

typedef struct grpc_custom_socket {
  // Implementation defined
  void* impl;
  grpc_endpoint* endpoint;
  grpc_tcp_listener* listener;
  grpc_custom_tcp_connect* connector;
  int refs;
} grpc_custom_socket;

typedef void (*grpc_custom_connect_callback)(grpc_custom_socket* socket,
                                             grpc_error* error);
typedef void (*grpc_custom_write_callback)(grpc_custom_socket* socket,
                                           grpc_error* error);
typedef void (*grpc_custom_read_callback)(grpc_custom_socket* socket,
                                          size_t nread, grpc_error* error);
typedef void (*grpc_custom_accept_callback)(grpc_custom_socket* socket,
                                            grpc_custom_socket* client,
                                            grpc_error* error);
typedef void (*grpc_custom_close_callback)(grpc_custom_socket* socket);

typedef struct grpc_socket_vtable {
  grpc_error* (*init)(grpc_custom_socket* socket, int domain);
  void (*connect)(grpc_custom_socket* socket, const grpc_sockaddr* addr,
                  size_t len, grpc_custom_connect_callback cb);
  void (*destroy)(grpc_custom_socket* socket);
  void (*shutdown)(grpc_custom_socket* socket);
  void (*close)(grpc_custom_socket* socket, grpc_custom_close_callback cb);
  void (*write)(grpc_custom_socket* socket, grpc_slice_buffer* slices,
                grpc_custom_write_callback cb);
  void (*read)(grpc_custom_socket* socket, char* buffer, size_t length,
               grpc_custom_read_callback cb);
  grpc_error* (*getpeername)(grpc_custom_socket* socket,
                             const grpc_sockaddr* addr, int* len);
  grpc_error* (*getsockname)(grpc_custom_socket* socket,
                             const grpc_sockaddr* addr, int* len);
  grpc_error* (*bind)(grpc_custom_socket* socket, const grpc_sockaddr* addr,
                      size_t len, int flags);
  grpc_error* (*listen)(grpc_custom_socket* socket);
  void (*accept)(grpc_custom_socket* socket, grpc_custom_socket* client,
                 grpc_custom_accept_callback cb);
} grpc_socket_vtable;

/* Internal APIs */
void grpc_custom_endpoint_init(grpc_socket_vtable* impl);

void grpc_custom_close_server_callback(grpc_tcp_listener* listener);

grpc_endpoint* custom_tcp_endpoint_create(grpc_custom_socket* socket,
                                          grpc_resource_quota* resource_quota,
                                          char* peer_string);

#endif /* GRPC_CORE_LIB_IOMGR_TCP_CUSTOM_H */
