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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RESOLVER_DNS_C_ARES_GRPC_ARES_WRAPPER_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RESOLVER_DNS_C_ARES_GRPC_ARES_WRAPPER_H

#include <grpc/support/port_platform.h>

#include "src/core/ext/filters/client_channel/server_address.h"
#include "src/core/lib/iomgr/iomgr.h"
#include "src/core/lib/iomgr/polling_entity.h"
#include "src/core/lib/iomgr/resolve_address.h"

#define GRPC_DNS_ARES_DEFAULT_QUERY_TIMEOUT_MS 120000

extern grpc_core::TraceFlag grpc_trace_cares_address_sorting;

extern grpc_core::TraceFlag grpc_trace_cares_resolver;

#define GRPC_CARES_TRACE_LOG(format, ...)                           \
  do {                                                              \
    if (GRPC_TRACE_FLAG_ENABLED(grpc_trace_cares_resolver)) {       \
      gpr_log(GPR_DEBUG, "(c-ares resolver) " format, __VA_ARGS__); \
    }                                                               \
  } while (0)

typedef struct grpc_ares_request grpc_ares_request;

/* Asynchronously resolve \a name. Use \a default_port if a port isn't
   designated in \a name, otherwise use the port in \a name. grpc_ares_init()
   must be called at least once before this function. \a on_done may be
   called directly in this function without being scheduled with \a exec_ctx,
   so it must not try to acquire locks that are being held by the caller. */
extern void (*grpc_resolve_address_ares)(const char* name,
                                         const char* default_port,
                                         grpc_pollset_set* interested_parties,
                                         grpc_closure* on_done,
                                         grpc_resolved_addresses** addresses);

/* Asynchronously resolve \a name. It will try to resolve grpclb SRV records in
  addition to the normal address records. For normal address records, it uses
  \a default_port if a port isn't designated in \a name, otherwise it uses the
  port in \a name. grpc_ares_init() must be called at least once before this
  function. \a on_done may be called directly in this function without being
  scheduled with \a exec_ctx, so it must not try to acquire locks that are
  being held by the caller. The returned grpc_ares_request object is owned
  by the caller and it is safe to free after on_done is called back. */
extern grpc_ares_request* (*grpc_dns_lookup_ares_locked)(
    const char* dns_server, const char* name, const char* default_port,
    grpc_pollset_set* interested_parties, grpc_closure* on_done,
    std::unique_ptr<grpc_core::ServerAddressList>* addresses, bool check_grpclb,
    char** service_config_json, int query_timeout_ms,
    grpc_core::Combiner* combiner);

/* Cancel the pending grpc_ares_request \a request */
extern void (*grpc_cancel_ares_request_locked)(grpc_ares_request* request);

/* Initialize gRPC ares wrapper. Must be called at least once before
   grpc_resolve_address_ares(). */
grpc_error* grpc_ares_init(void);

/* Uninitialized gRPC ares wrapper. If there was more than one previous call to
   grpc_ares_init(), this function uninitializes the gRPC ares wrapper only if
   it has been called the same number of times as grpc_ares_init(). */
void grpc_ares_cleanup(void);

/** Schedules the desired callback for request completion
 * and destroys the grpc_ares_request */
void grpc_ares_complete_request_locked(grpc_ares_request* request);

/* Indicates whether or not AAAA queries should be attempted. */
/* E.g., return false if ipv6 is known to not be available. */
bool grpc_ares_query_ipv6();

/* Sorts destinations in lb_addrs according to RFC 6724. */
void grpc_cares_wrapper_address_sorting_sort(
    grpc_core::ServerAddressList* addresses);

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_RESOLVER_DNS_C_ARES_GRPC_ARES_WRAPPER_H \
        */
