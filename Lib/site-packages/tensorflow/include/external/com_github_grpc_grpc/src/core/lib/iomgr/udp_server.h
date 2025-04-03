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

#ifndef GRPC_CORE_LIB_IOMGR_UDP_SERVER_H
#define GRPC_CORE_LIB_IOMGR_UDP_SERVER_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/iomgr/endpoint.h"
#include "src/core/lib/iomgr/ev_posix.h"
#include "src/core/lib/iomgr/resolve_address.h"

/* Forward decl of struct grpc_server */
/* This is not typedef'ed to avoid a typedef-redefinition error */
struct grpc_server;

/* Forward decl of grpc_udp_server */
typedef struct grpc_udp_server grpc_udp_server;

/* An interface associated with a socket. udp server delivers I/O event on that
 * socket to the subclass of this interface which is created through
 * GrpcUdpHandlerFactory.
 * Its implementation should do the real IO work, e.g. read packet and write. */
class GrpcUdpHandler {
 public:
  GrpcUdpHandler(grpc_fd* /* emfd */, void* /* user_data */) {}
  virtual ~GrpcUdpHandler() {}

  // Interfaces to be implemented by subclasses to do the actual setup/tear down
  // or I/O.

  // Called when data is available to read from the socket. Returns true if
  // there is more data to read after this call.
  virtual bool Read() = 0;
  // Called when socket becomes write unblocked. The given closure should be
  // scheduled when the socket becomes blocked next time.
  virtual void OnCanWrite(void* user_data,
                          grpc_closure* notify_on_write_closure) = 0;
  // Called before the gRPC FD is orphaned. Notify udp server to continue
  // orphaning fd by scheduling the given closure, afterwards the associated fd
  // will be closed.
  virtual void OnFdAboutToOrphan(grpc_closure* orphan_fd_closure,
                                 void* user_data) = 0;
};

class GrpcUdpHandlerFactory {
 public:
  virtual ~GrpcUdpHandlerFactory() {}
  /* Called when start to listen on a socket.
   * Return an instance of the implementation of GrpcUdpHandler interface which
   * will process I/O events for this socket from now on. */
  virtual GrpcUdpHandler* CreateUdpHandler(grpc_fd* emfd, void* user_data) = 0;
  virtual void DestroyUdpHandler(GrpcUdpHandler* handler) = 0;
};

/* Create a server, initially not bound to any ports */
grpc_udp_server* grpc_udp_server_create(const grpc_channel_args* args);

/* Start listening to bound ports. user_data is passed to callbacks. */
void grpc_udp_server_start(grpc_udp_server* udp_server, grpc_pollset** pollsets,
                           size_t pollset_count, void* user_data);

int grpc_udp_server_get_fd(grpc_udp_server* s, unsigned port_index);

/* Add a port to the server, returning port number on success, or negative
   on failure.

   Create |num_listeners| sockets for given address to listen on using
   SO_REUSEPORT if supported.

   The :: and 0.0.0.0 wildcard addresses are treated identically, accepting
   both IPv4 and IPv6 connections, but :: is the preferred style. This usually
   creates |num_listeners| sockets, but possibly 2 * |num_listeners| on systems
   which support IPv6, but not dualstack sockets. */

/* TODO(ctiller): deprecate this, and make grpc_udp_server_add_ports to handle
                  all of the multiple socket port matching logic in one place */
int grpc_udp_server_add_port(grpc_udp_server* s,
                             const grpc_resolved_address* addr,
                             int rcv_buf_size, int snd_buf_size,
                             GrpcUdpHandlerFactory* handler_factory,
                             size_t num_listeners);

void grpc_udp_server_destroy(grpc_udp_server* server, grpc_closure* on_done);

#endif /* GRPC_CORE_LIB_IOMGR_UDP_SERVER_H */
