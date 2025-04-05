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

#ifndef GRPC_CORE_LIB_IOMGR_LOAD_FILE_H
#define GRPC_CORE_LIB_IOMGR_LOAD_FILE_H

#include <grpc/support/port_platform.h>

#include <stdio.h>

#include <grpc/slice.h>

#include "src/core/lib/iomgr/error.h"

/* Loads the content of a file into a slice. add_null_terminator will add
   a NULL terminator if non-zero. */
grpc_error* grpc_load_file(const char* filename, int add_null_terminator,
                           grpc_slice* slice);

#endif /* GRPC_CORE_LIB_IOMGR_LOAD_FILE_H */
