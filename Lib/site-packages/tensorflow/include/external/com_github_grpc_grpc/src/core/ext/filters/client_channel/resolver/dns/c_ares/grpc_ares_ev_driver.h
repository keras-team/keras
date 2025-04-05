/*
 *
 * Copyright 2016 gRPC authors.
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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RESOLVER_DNS_C_ARES_GRPC_ARES_EV_DRIVER_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RESOLVER_DNS_C_ARES_GRPC_ARES_EV_DRIVER_H

#include <grpc/support/port_platform.h>

#include <ares.h>
#include "src/core/ext/filters/client_channel/resolver/dns/c_ares/grpc_ares_wrapper.h"
#include "src/core/lib/iomgr/pollset_set.h"

typedef struct grpc_ares_ev_driver grpc_ares_ev_driver;

/* Start \a ev_driver. It will keep working until all IO on its ares_channel is
   done, or grpc_ares_ev_driver_destroy() is called. It may notify the callbacks
   bound to its ares_channel when necessary. */
void grpc_ares_ev_driver_start_locked(grpc_ares_ev_driver* ev_driver);

/* Returns the ares_channel owned by \a ev_driver. To bind a c-ares query to
   \a ev_driver, use the ares_channel owned by \a ev_driver as the arg of the
   query. */
ares_channel* grpc_ares_ev_driver_get_channel_locked(
    grpc_ares_ev_driver* ev_driver);

/* Creates a new grpc_ares_ev_driver. Returns GRPC_ERROR_NONE if \a ev_driver is
   created successfully. */
grpc_error* grpc_ares_ev_driver_create_locked(grpc_ares_ev_driver** ev_driver,
                                              grpc_pollset_set* pollset_set,
                                              int query_timeout_ms,
                                              grpc_core::Combiner* combiner,
                                              grpc_ares_request* request);

/* Called back when all DNS lookups have completed. */
void grpc_ares_ev_driver_on_queries_complete_locked(
    grpc_ares_ev_driver* ev_driver);

/* Shutdown all the grpc_fds used by \a ev_driver */
void grpc_ares_ev_driver_shutdown_locked(grpc_ares_ev_driver* ev_driver);

/* Exposed in this header for C-core tests only */
extern void (*grpc_ares_test_only_inject_config)(ares_channel channel);

namespace grpc_core {

/* A wrapped fd that integrates with the grpc iomgr of the current platform.
 * A GrpcPolledFd knows how to create grpc platform-specific iomgr endpoints
 * from "ares_socket_t" sockets, and then sign up for readability/writeability
 * with that poller, and do shutdown and destruction. */
class GrpcPolledFd {
 public:
  virtual ~GrpcPolledFd() {}
  /* Called when c-ares library is interested and there's no pending callback */
  virtual void RegisterForOnReadableLocked(grpc_closure* read_closure) = 0;
  /* Called when c-ares library is interested and there's no pending callback */
  virtual void RegisterForOnWriteableLocked(grpc_closure* write_closure) = 0;
  /* Indicates if there is data left even after just being read from */
  virtual bool IsFdStillReadableLocked() = 0;
  /* Called once and only once. Must cause cancellation of any pending
   * read/write callbacks. */
  virtual void ShutdownLocked(grpc_error* error) = 0;
  /* Get the underlying ares_socket_t that this was created from */
  virtual ares_socket_t GetWrappedAresSocketLocked() = 0;
  /* A unique name, for logging */
  virtual const char* GetName() = 0;
};

/* A GrpcPolledFdFactory is 1-to-1 with and owned by the
 * ares event driver. It knows how to create GrpcPolledFd's
 * for the current platform, and the ares driver uses it for all of
 * its fd's. */
class GrpcPolledFdFactory {
 public:
  virtual ~GrpcPolledFdFactory() {}
  /* Creates a new wrapped fd for the current platform */
  virtual GrpcPolledFd* NewGrpcPolledFdLocked(
      ares_socket_t as, grpc_pollset_set* driver_pollset_set,
      Combiner* combiner) = 0;
  /* Optionally configures the ares channel after creation */
  virtual void ConfigureAresChannelLocked(ares_channel channel) = 0;
};

std::unique_ptr<GrpcPolledFdFactory> NewGrpcPolledFdFactory(Combiner* combiner);

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RESOLVER_DNS_C_ARES_GRPC_ARES_EV_DRIVER_H \
        */
