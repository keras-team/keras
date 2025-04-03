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

#ifndef GRPC_CORE_LIB_GPR_TLS_H
#define GRPC_CORE_LIB_GPR_TLS_H

#include <grpc/support/port_platform.h>

/** Thread local storage.

   A minimal wrapper that should be implementable across many compilers,
   and implementable efficiently across most modern compilers.

   Thread locals have type intptr_t.

   Declaring a thread local variable 'foo':
     GPR_TLS_DECL(foo);
   Thread locals always have static scope.

   Declaring a thread local class variable 'foo':
     GPR_TLS_CLASS_DECL(foo);

   Defining the thread local class variable:
     GPR_TLS_CLASS_DEF(foo);

   Initializing a thread local (must be done at library initialization
   time):
     gpr_tls_init(&foo);

   Destroying a thread local:
     gpr_tls_destroy(&foo);

   Setting a thread local (returns new_value):
     gpr_tls_set(&foo, new_value);

   Accessing a thread local:
     current_value = gpr_tls_get(&foo);

   ALL functions here may be implemented as macros. */

#ifdef GPR_GCC_TLS
#include "src/core/lib/gpr/tls_gcc.h"
#endif

#ifdef GPR_MSVC_TLS
#include "src/core/lib/gpr/tls_msvc.h"
#endif

#ifdef GPR_PTHREAD_TLS
#include "src/core/lib/gpr/tls_pthread.h"
#endif

#endif /* GRPC_CORE_LIB_GPR_TLS_H */
