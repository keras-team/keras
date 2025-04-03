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

#ifndef GRPC_CORE_LIB_URI_URI_PARSER_H
#define GRPC_CORE_LIB_URI_URI_PARSER_H

#include <grpc/support/port_platform.h>

#include <stddef.h>

typedef struct {
  char* scheme;
  char* authority;
  char* path;
  char* query;
  /** Query substrings separated by '&' */
  char** query_parts;
  /** Number of elements in \a query_parts and \a query_parts_values */
  size_t num_query_parts;
  /** Split each query part by '='. NULL if not present. */
  char** query_parts_values;
  char* fragment;
} grpc_uri;

/** parse a uri, return NULL on failure */
grpc_uri* grpc_uri_parse(const char* uri_text, bool suppress_errors);

/** return the part of a query string after the '=' in "?key=xxx&...", or NULL
 * if key is not present */
const char* grpc_uri_get_query_arg(const grpc_uri* uri, const char* key);

/** destroy a uri */
void grpc_uri_destroy(grpc_uri* uri);

#endif /* GRPC_CORE_LIB_URI_URI_PARSER_H */
