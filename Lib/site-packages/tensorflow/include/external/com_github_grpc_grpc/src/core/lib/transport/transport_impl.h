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

#ifndef GRPC_CORE_LIB_TRANSPORT_TRANSPORT_IMPL_H
#define GRPC_CORE_LIB_TRANSPORT_TRANSPORT_IMPL_H

#include <grpc/support/port_platform.h>

#include "src/core/lib/transport/transport.h"

typedef struct grpc_transport_vtable {
  /* Memory required for a single stream element - this is allocated by upper
     layers and initialized by the transport */
  size_t sizeof_stream; /* = sizeof(transport stream) */

  /* name of this transport implementation */
  const char* name;

  /* implementation of grpc_transport_init_stream */
  int (*init_stream)(grpc_transport* self, grpc_stream* stream,
                     grpc_stream_refcount* refcount, const void* server_data,
                     grpc_core::Arena* arena);

  /* implementation of grpc_transport_set_pollset */
  void (*set_pollset)(grpc_transport* self, grpc_stream* stream,
                      grpc_pollset* pollset);

  /* implementation of grpc_transport_set_pollset */
  void (*set_pollset_set)(grpc_transport* self, grpc_stream* stream,
                          grpc_pollset_set* pollset_set);

  /* implementation of grpc_transport_perform_stream_op */
  void (*perform_stream_op)(grpc_transport* self, grpc_stream* stream,
                            grpc_transport_stream_op_batch* op);

  /* implementation of grpc_transport_perform_op */
  void (*perform_op)(grpc_transport* self, grpc_transport_op* op);

  /* implementation of grpc_transport_destroy_stream */
  void (*destroy_stream)(grpc_transport* self, grpc_stream* stream,
                         grpc_closure* then_schedule_closure);

  /* implementation of grpc_transport_destroy */
  void (*destroy)(grpc_transport* self);

  /* implementation of grpc_transport_get_endpoint */
  grpc_endpoint* (*get_endpoint)(grpc_transport* self);
} grpc_transport_vtable;

/* an instance of a grpc transport */
struct grpc_transport {
  /* pointer to a vtable defining operations on this transport */
  const grpc_transport_vtable* vtable;
};

#endif /* GRPC_CORE_LIB_TRANSPORT_TRANSPORT_IMPL_H */
