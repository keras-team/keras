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

#ifndef GRPC_GRPC_POSIX_H
#define GRPC_GRPC_POSIX_H

#include <grpc/support/port_platform.h>

#include <grpc/impl/codegen/grpc_types.h>

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \mainpage GRPC Core POSIX
 *
 * The GRPC Core POSIX library provides some POSIX-specific low-level
 * functionality on top of GRPC Core.
 */

/** Create a client channel to 'target' using file descriptor 'fd'. The 'target'
    argument will be used to indicate the name for this channel. See the comment
    for grpc_insecure_channel_create for description of 'args' argument. */
GRPCAPI grpc_channel* grpc_insecure_channel_create_from_fd(
    const char* target, int fd, const grpc_channel_args* args);

/** Add the connected communication channel based on file descriptor 'fd' to the
    'server'. The 'fd' must be an open file descriptor corresponding to a
    connected socket. Events from the file descriptor may come on any of the
    server completion queues (i.e completion queues registered via the
    grpc_server_register_completion_queue API).

    The 'reserved' pointer MUST be NULL.
    */
GRPCAPI void grpc_server_add_insecure_channel_from_fd(grpc_server* server,
                                                      void* reserved, int fd);

#ifdef __cplusplus
}
#endif

#endif /* GRPC_GRPC_POSIX_H */
