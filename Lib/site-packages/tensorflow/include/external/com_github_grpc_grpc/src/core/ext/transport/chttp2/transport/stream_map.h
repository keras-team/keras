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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_STREAM_MAP_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_STREAM_MAP_H

#include <grpc/support/port_platform.h>

#include <stddef.h>

/* Data structure to map a uint32_t to a data object (represented by a void*)

   Represented as a sorted array of keys, and a corresponding array of values.
   Lookups are performed with binary search.
   Adds are restricted to strictly higher keys than previously seen (this is
   guaranteed by http2). */
typedef struct {
  uint32_t* keys;
  void** values;
  size_t count;
  size_t free;
  size_t capacity;
} grpc_chttp2_stream_map;

void grpc_chttp2_stream_map_init(grpc_chttp2_stream_map* map,
                                 size_t initial_capacity);
void grpc_chttp2_stream_map_destroy(grpc_chttp2_stream_map* map);

/* Add a new key: given http2 semantics, new keys must always be greater than
   existing keys - this is asserted */
void grpc_chttp2_stream_map_add(grpc_chttp2_stream_map* map, uint32_t key,
                                void* value);

/* Delete an existing key - returns the previous value of the key if it existed,
   or NULL otherwise */
void* grpc_chttp2_stream_map_delete(grpc_chttp2_stream_map* map, uint32_t key);

/* Return an existing key, or NULL if it does not exist */
void* grpc_chttp2_stream_map_find(grpc_chttp2_stream_map* map, uint32_t key);

/* Return a random entry */
void* grpc_chttp2_stream_map_rand(grpc_chttp2_stream_map* map);

/* How many (populated) entries are in the stream map? */
size_t grpc_chttp2_stream_map_size(grpc_chttp2_stream_map* map);

/* Callback on each stream */
void grpc_chttp2_stream_map_for_each(grpc_chttp2_stream_map* map,
                                     void (*f)(void* user_data, uint32_t key,
                                               void* value),
                                     void* user_data);

#endif /* GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_STREAM_MAP_H */
