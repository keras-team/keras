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

#ifndef GRPC_CORE_LIB_GPR_TLS_PTHREAD_H
#define GRPC_CORE_LIB_GPR_TLS_PTHREAD_H

#include <grpc/support/port_platform.h>

#include <grpc/support/log.h> /* for GPR_ASSERT */
#include <pthread.h>

/** Thread local storage based on pthread library calls.
   #include tls.h to use this - and see that file for documentation */

struct gpr_pthread_thread_local {
  pthread_key_t key;
};

/** Use GPR_TLS_DECL to declare tls static variables outside a class */
#define GPR_TLS_DECL(name) static struct gpr_pthread_thread_local name = {0}

/** Use GPR_TLS_CLASS_DECL to declare tls static variable members of a class.
 *  GPR_TLS_CLASS_DEF needs to be called to define this member. */
#define GPR_TLS_CLASS_DECL(name) static struct gpr_pthread_thread_local name

/** Use GPR_TLS_CLASS_DEF to declare tls static variable members of a class.
 *  GPR_TLS_CLASS_DEF needs to be called to define this member. */
#define GPR_TLS_CLASS_DEF(name) struct gpr_pthread_thread_local name = {0}

#define gpr_tls_init(tls) GPR_ASSERT(0 == pthread_key_create(&(tls)->key, NULL))
#define gpr_tls_destroy(tls) pthread_key_delete((tls)->key)
#define gpr_tls_get(tls) ((intptr_t)pthread_getspecific((tls)->key))
#ifdef __cplusplus
extern "C" {
#endif
intptr_t gpr_tls_set(struct gpr_pthread_thread_local* tls, intptr_t value);
#ifdef __cplusplus
}
#endif

#endif /* GRPC_CORE_LIB_GPR_TLS_PTHREAD_H */
