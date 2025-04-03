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

#ifndef GRPC_CORE_LIB_IOMGR_IOMGR_CUSTOM_H
#define GRPC_CORE_LIB_IOMGR_IOMGR_CUSTOM_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/iomgr/pollset_custom.h"
#include "src/core/lib/iomgr/resolve_address_custom.h"
#include "src/core/lib/iomgr/tcp_custom.h"
#include "src/core/lib/iomgr/timer_custom.h"

#include <grpc/support/thd_id.h>

/* The thread ID of the thread on which grpc was initialized. Used to verify
 * that all calls into the custom iomgr are made on that same thread */
extern gpr_thd_id g_init_thread;

#ifdef GRPC_CUSTOM_IOMGR_THREAD_CHECK
#define GRPC_CUSTOM_IOMGR_ASSERT_SAME_THREAD() \
  GPR_ASSERT(gpr_thd_currentid() == g_init_thread)
#else
#define GRPC_CUSTOM_IOMGR_ASSERT_SAME_THREAD()
#endif /* GRPC_CUSTOM_IOMGR_THREAD_CHECK */

extern bool g_custom_iomgr_enabled;

void grpc_custom_iomgr_init(grpc_socket_vtable* socket,
                            grpc_custom_resolver_vtable* resolver,
                            grpc_custom_timer_vtable* timer,
                            grpc_custom_poller_vtable* poller);

#endif /* GRPC_CORE_LIB_IOMGR_IOMGR_CUSTOM_H */
