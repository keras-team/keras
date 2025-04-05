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

#ifndef GRPC_CORE_LIB_GPR_MURMUR_HASH_H
#define GRPC_CORE_LIB_GPR_MURMUR_HASH_H

#include <grpc/support/port_platform.h>

#include <stddef.h>

/* compute the hash of key (length len) */
uint32_t gpr_murmur_hash3(const void* key, size_t len, uint32_t seed);

#endif /* GRPC_CORE_LIB_GPR_MURMUR_HASH_H */
