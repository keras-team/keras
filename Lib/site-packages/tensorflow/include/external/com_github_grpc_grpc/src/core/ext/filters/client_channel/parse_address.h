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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_PARSE_ADDRESS_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_PARSE_ADDRESS_H

#include <grpc/support/port_platform.h>

#include <stddef.h>

#include "src/core/lib/iomgr/resolve_address.h"
#include "src/core/lib/uri/uri_parser.h"

/** Populate \a resolved_addr from \a uri, whose path is expected to contain a
 * unix socket path. Returns true upon success. */
bool grpc_parse_unix(const grpc_uri* uri, grpc_resolved_address* resolved_addr);

/** Populate \a resolved_addr from \a uri, whose path is expected to contain an
 * IPv4 host:port pair. Returns true upon success. */
bool grpc_parse_ipv4(const grpc_uri* uri, grpc_resolved_address* resolved_addr);

/** Populate \a resolved_addr from \a uri, whose path is expected to contain an
 * IPv6 host:port pair. Returns true upon success. */
bool grpc_parse_ipv6(const grpc_uri* uri, grpc_resolved_address* resolved_addr);

/** Populate \a resolved_addr from \a uri. Returns true upon success. */
bool grpc_parse_uri(const grpc_uri* uri, grpc_resolved_address* resolved_addr);

/** Parse bare IPv4 or IPv6 "IP:port" strings. */
bool grpc_parse_ipv4_hostport(const char* hostport, grpc_resolved_address* addr,
                              bool log_errors);
bool grpc_parse_ipv6_hostport(const char* hostport, grpc_resolved_address* addr,
                              bool log_errors);

/* Converts named or numeric port to a uint16 suitable for use in a sockaddr. */
uint16_t grpc_strhtons(const char* port);

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_PARSE_ADDRESS_H */
