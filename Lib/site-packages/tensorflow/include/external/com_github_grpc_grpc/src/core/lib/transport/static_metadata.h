/*
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
 */

/*
 * WARNING: Auto-generated code.
 *
 * To make changes to this file, change
 * tools/codegen/core/gen_static_metadata.py, and then re-run it.
 *
 * See metadata.h for an explanation of the interface here, and metadata.cc for
 * an explanation of what's going on.
 */

#ifndef GRPC_CORE_LIB_TRANSPORT_STATIC_METADATA_H
#define GRPC_CORE_LIB_TRANSPORT_STATIC_METADATA_H

#include <grpc/support/port_platform.h>

#include <cstdint>

#include "src/core/lib/transport/metadata.h"

static_assert(
    std::is_trivially_destructible<grpc_core::StaticMetadataSlice>::value,
    "grpc_core::StaticMetadataSlice must be trivially destructible.");
#define GRPC_STATIC_MDSTR_COUNT 108

void grpc_init_static_metadata_ctx(void);
void grpc_destroy_static_metadata_ctx(void);
namespace grpc_core {
#ifndef NDEBUG
constexpr uint64_t kGrpcStaticMetadataInitCanary = 0xCAFEF00DC0FFEE11L;
uint64_t StaticMetadataInitCanary();
#endif
extern const StaticMetadataSlice* g_static_metadata_slice_table;
}
inline const grpc_core::StaticMetadataSlice* grpc_static_slice_table() {
  GPR_DEBUG_ASSERT(grpc_core::StaticMetadataInitCanary() ==
                   grpc_core::kGrpcStaticMetadataInitCanary);
  GPR_DEBUG_ASSERT(grpc_core::g_static_metadata_slice_table != nullptr);
  return grpc_core::g_static_metadata_slice_table;
}

/* ":path" */
#define GRPC_MDSTR_PATH (grpc_static_slice_table()[0])
/* ":method" */
#define GRPC_MDSTR_METHOD (grpc_static_slice_table()[1])
/* ":status" */
#define GRPC_MDSTR_STATUS (grpc_static_slice_table()[2])
/* ":authority" */
#define GRPC_MDSTR_AUTHORITY (grpc_static_slice_table()[3])
/* ":scheme" */
#define GRPC_MDSTR_SCHEME (grpc_static_slice_table()[4])
/* "te" */
#define GRPC_MDSTR_TE (grpc_static_slice_table()[5])
/* "grpc-message" */
#define GRPC_MDSTR_GRPC_MESSAGE (grpc_static_slice_table()[6])
/* "grpc-status" */
#define GRPC_MDSTR_GRPC_STATUS (grpc_static_slice_table()[7])
/* "grpc-payload-bin" */
#define GRPC_MDSTR_GRPC_PAYLOAD_BIN (grpc_static_slice_table()[8])
/* "grpc-encoding" */
#define GRPC_MDSTR_GRPC_ENCODING (grpc_static_slice_table()[9])
/* "grpc-accept-encoding" */
#define GRPC_MDSTR_GRPC_ACCEPT_ENCODING (grpc_static_slice_table()[10])
/* "grpc-server-stats-bin" */
#define GRPC_MDSTR_GRPC_SERVER_STATS_BIN (grpc_static_slice_table()[11])
/* "grpc-tags-bin" */
#define GRPC_MDSTR_GRPC_TAGS_BIN (grpc_static_slice_table()[12])
/* "grpc-trace-bin" */
#define GRPC_MDSTR_GRPC_TRACE_BIN (grpc_static_slice_table()[13])
/* "content-type" */
#define GRPC_MDSTR_CONTENT_TYPE (grpc_static_slice_table()[14])
/* "content-encoding" */
#define GRPC_MDSTR_CONTENT_ENCODING (grpc_static_slice_table()[15])
/* "accept-encoding" */
#define GRPC_MDSTR_ACCEPT_ENCODING (grpc_static_slice_table()[16])
/* "grpc-internal-encoding-request" */
#define GRPC_MDSTR_GRPC_INTERNAL_ENCODING_REQUEST \
  (grpc_static_slice_table()[17])
/* "grpc-internal-stream-encoding-request" */
#define GRPC_MDSTR_GRPC_INTERNAL_STREAM_ENCODING_REQUEST \
  (grpc_static_slice_table()[18])
/* "user-agent" */
#define GRPC_MDSTR_USER_AGENT (grpc_static_slice_table()[19])
/* "host" */
#define GRPC_MDSTR_HOST (grpc_static_slice_table()[20])
/* "grpc-previous-rpc-attempts" */
#define GRPC_MDSTR_GRPC_PREVIOUS_RPC_ATTEMPTS (grpc_static_slice_table()[21])
/* "grpc-retry-pushback-ms" */
#define GRPC_MDSTR_GRPC_RETRY_PUSHBACK_MS (grpc_static_slice_table()[22])
/* "x-endpoint-load-metrics-bin" */
#define GRPC_MDSTR_X_ENDPOINT_LOAD_METRICS_BIN (grpc_static_slice_table()[23])
/* "grpc-timeout" */
#define GRPC_MDSTR_GRPC_TIMEOUT (grpc_static_slice_table()[24])
/* "1" */
#define GRPC_MDSTR_1 (grpc_static_slice_table()[25])
/* "2" */
#define GRPC_MDSTR_2 (grpc_static_slice_table()[26])
/* "3" */
#define GRPC_MDSTR_3 (grpc_static_slice_table()[27])
/* "4" */
#define GRPC_MDSTR_4 (grpc_static_slice_table()[28])
/* "" */
#define GRPC_MDSTR_EMPTY (grpc_static_slice_table()[29])
/* "grpc.wait_for_ready" */
#define GRPC_MDSTR_GRPC_DOT_WAIT_FOR_READY (grpc_static_slice_table()[30])
/* "grpc.timeout" */
#define GRPC_MDSTR_GRPC_DOT_TIMEOUT (grpc_static_slice_table()[31])
/* "grpc.max_request_message_bytes" */
#define GRPC_MDSTR_GRPC_DOT_MAX_REQUEST_MESSAGE_BYTES \
  (grpc_static_slice_table()[32])
/* "grpc.max_response_message_bytes" */
#define GRPC_MDSTR_GRPC_DOT_MAX_RESPONSE_MESSAGE_BYTES \
  (grpc_static_slice_table()[33])
/* "/grpc.lb.v1.LoadBalancer/BalanceLoad" */
#define GRPC_MDSTR_SLASH_GRPC_DOT_LB_DOT_V1_DOT_LOADBALANCER_SLASH_BALANCELOAD \
  (grpc_static_slice_table()[34])
/* "/envoy.service.load_stats.v2.LoadReportingService/StreamLoadStats" */
#define GRPC_MDSTR_SLASH_ENVOY_DOT_SERVICE_DOT_LOAD_STATS_DOT_V2_DOT_LOADREPORTINGSERVICE_SLASH_STREAMLOADSTATS \
  (grpc_static_slice_table()[35])
/* "/grpc.health.v1.Health/Watch" */
#define GRPC_MDSTR_SLASH_GRPC_DOT_HEALTH_DOT_V1_DOT_HEALTH_SLASH_WATCH \
  (grpc_static_slice_table()[36])
/* "/envoy.service.discovery.v2.AggregatedDiscoveryService/StreamAggregatedResources"
 */
#define GRPC_MDSTR_SLASH_ENVOY_DOT_SERVICE_DOT_DISCOVERY_DOT_V2_DOT_AGGREGATEDDISCOVERYSERVICE_SLASH_STREAMAGGREGATEDRESOURCES \
  (grpc_static_slice_table()[37])
/* "deflate" */
#define GRPC_MDSTR_DEFLATE (grpc_static_slice_table()[38])
/* "gzip" */
#define GRPC_MDSTR_GZIP (grpc_static_slice_table()[39])
/* "stream/gzip" */
#define GRPC_MDSTR_STREAM_SLASH_GZIP (grpc_static_slice_table()[40])
/* "GET" */
#define GRPC_MDSTR_GET (grpc_static_slice_table()[41])
/* "POST" */
#define GRPC_MDSTR_POST (grpc_static_slice_table()[42])
/* "/" */
#define GRPC_MDSTR_SLASH (grpc_static_slice_table()[43])
/* "/index.html" */
#define GRPC_MDSTR_SLASH_INDEX_DOT_HTML (grpc_static_slice_table()[44])
/* "http" */
#define GRPC_MDSTR_HTTP (grpc_static_slice_table()[45])
/* "https" */
#define GRPC_MDSTR_HTTPS (grpc_static_slice_table()[46])
/* "200" */
#define GRPC_MDSTR_200 (grpc_static_slice_table()[47])
/* "204" */
#define GRPC_MDSTR_204 (grpc_static_slice_table()[48])
/* "206" */
#define GRPC_MDSTR_206 (grpc_static_slice_table()[49])
/* "304" */
#define GRPC_MDSTR_304 (grpc_static_slice_table()[50])
/* "400" */
#define GRPC_MDSTR_400 (grpc_static_slice_table()[51])
/* "404" */
#define GRPC_MDSTR_404 (grpc_static_slice_table()[52])
/* "500" */
#define GRPC_MDSTR_500 (grpc_static_slice_table()[53])
/* "accept-charset" */
#define GRPC_MDSTR_ACCEPT_CHARSET (grpc_static_slice_table()[54])
/* "gzip, deflate" */
#define GRPC_MDSTR_GZIP_COMMA_DEFLATE (grpc_static_slice_table()[55])
/* "accept-language" */
#define GRPC_MDSTR_ACCEPT_LANGUAGE (grpc_static_slice_table()[56])
/* "accept-ranges" */
#define GRPC_MDSTR_ACCEPT_RANGES (grpc_static_slice_table()[57])
/* "accept" */
#define GRPC_MDSTR_ACCEPT (grpc_static_slice_table()[58])
/* "access-control-allow-origin" */
#define GRPC_MDSTR_ACCESS_CONTROL_ALLOW_ORIGIN (grpc_static_slice_table()[59])
/* "age" */
#define GRPC_MDSTR_AGE (grpc_static_slice_table()[60])
/* "allow" */
#define GRPC_MDSTR_ALLOW (grpc_static_slice_table()[61])
/* "authorization" */
#define GRPC_MDSTR_AUTHORIZATION (grpc_static_slice_table()[62])
/* "cache-control" */
#define GRPC_MDSTR_CACHE_CONTROL (grpc_static_slice_table()[63])
/* "content-disposition" */
#define GRPC_MDSTR_CONTENT_DISPOSITION (grpc_static_slice_table()[64])
/* "content-language" */
#define GRPC_MDSTR_CONTENT_LANGUAGE (grpc_static_slice_table()[65])
/* "content-length" */
#define GRPC_MDSTR_CONTENT_LENGTH (grpc_static_slice_table()[66])
/* "content-location" */
#define GRPC_MDSTR_CONTENT_LOCATION (grpc_static_slice_table()[67])
/* "content-range" */
#define GRPC_MDSTR_CONTENT_RANGE (grpc_static_slice_table()[68])
/* "cookie" */
#define GRPC_MDSTR_COOKIE (grpc_static_slice_table()[69])
/* "date" */
#define GRPC_MDSTR_DATE (grpc_static_slice_table()[70])
/* "etag" */
#define GRPC_MDSTR_ETAG (grpc_static_slice_table()[71])
/* "expect" */
#define GRPC_MDSTR_EXPECT (grpc_static_slice_table()[72])
/* "expires" */
#define GRPC_MDSTR_EXPIRES (grpc_static_slice_table()[73])
/* "from" */
#define GRPC_MDSTR_FROM (grpc_static_slice_table()[74])
/* "if-match" */
#define GRPC_MDSTR_IF_MATCH (grpc_static_slice_table()[75])
/* "if-modified-since" */
#define GRPC_MDSTR_IF_MODIFIED_SINCE (grpc_static_slice_table()[76])
/* "if-none-match" */
#define GRPC_MDSTR_IF_NONE_MATCH (grpc_static_slice_table()[77])
/* "if-range" */
#define GRPC_MDSTR_IF_RANGE (grpc_static_slice_table()[78])
/* "if-unmodified-since" */
#define GRPC_MDSTR_IF_UNMODIFIED_SINCE (grpc_static_slice_table()[79])
/* "last-modified" */
#define GRPC_MDSTR_LAST_MODIFIED (grpc_static_slice_table()[80])
/* "link" */
#define GRPC_MDSTR_LINK (grpc_static_slice_table()[81])
/* "location" */
#define GRPC_MDSTR_LOCATION (grpc_static_slice_table()[82])
/* "max-forwards" */
#define GRPC_MDSTR_MAX_FORWARDS (grpc_static_slice_table()[83])
/* "proxy-authenticate" */
#define GRPC_MDSTR_PROXY_AUTHENTICATE (grpc_static_slice_table()[84])
/* "proxy-authorization" */
#define GRPC_MDSTR_PROXY_AUTHORIZATION (grpc_static_slice_table()[85])
/* "range" */
#define GRPC_MDSTR_RANGE (grpc_static_slice_table()[86])
/* "referer" */
#define GRPC_MDSTR_REFERER (grpc_static_slice_table()[87])
/* "refresh" */
#define GRPC_MDSTR_REFRESH (grpc_static_slice_table()[88])
/* "retry-after" */
#define GRPC_MDSTR_RETRY_AFTER (grpc_static_slice_table()[89])
/* "server" */
#define GRPC_MDSTR_SERVER (grpc_static_slice_table()[90])
/* "set-cookie" */
#define GRPC_MDSTR_SET_COOKIE (grpc_static_slice_table()[91])
/* "strict-transport-security" */
#define GRPC_MDSTR_STRICT_TRANSPORT_SECURITY (grpc_static_slice_table()[92])
/* "transfer-encoding" */
#define GRPC_MDSTR_TRANSFER_ENCODING (grpc_static_slice_table()[93])
/* "vary" */
#define GRPC_MDSTR_VARY (grpc_static_slice_table()[94])
/* "via" */
#define GRPC_MDSTR_VIA (grpc_static_slice_table()[95])
/* "www-authenticate" */
#define GRPC_MDSTR_WWW_AUTHENTICATE (grpc_static_slice_table()[96])
/* "0" */
#define GRPC_MDSTR_0 (grpc_static_slice_table()[97])
/* "identity" */
#define GRPC_MDSTR_IDENTITY (grpc_static_slice_table()[98])
/* "trailers" */
#define GRPC_MDSTR_TRAILERS (grpc_static_slice_table()[99])
/* "application/grpc" */
#define GRPC_MDSTR_APPLICATION_SLASH_GRPC (grpc_static_slice_table()[100])
/* "grpc" */
#define GRPC_MDSTR_GRPC (grpc_static_slice_table()[101])
/* "PUT" */
#define GRPC_MDSTR_PUT (grpc_static_slice_table()[102])
/* "lb-cost-bin" */
#define GRPC_MDSTR_LB_COST_BIN (grpc_static_slice_table()[103])
/* "identity,deflate" */
#define GRPC_MDSTR_IDENTITY_COMMA_DEFLATE (grpc_static_slice_table()[104])
/* "identity,gzip" */
#define GRPC_MDSTR_IDENTITY_COMMA_GZIP (grpc_static_slice_table()[105])
/* "deflate,gzip" */
#define GRPC_MDSTR_DEFLATE_COMMA_GZIP (grpc_static_slice_table()[106])
/* "identity,deflate,gzip" */
#define GRPC_MDSTR_IDENTITY_COMMA_DEFLATE_COMMA_GZIP \
  (grpc_static_slice_table()[107])

namespace grpc_core {
struct StaticSliceRefcount;
extern StaticSliceRefcount* g_static_metadata_slice_refcounts;
}  // namespace grpc_core
inline grpc_core::StaticSliceRefcount* grpc_static_metadata_refcounts() {
  GPR_DEBUG_ASSERT(grpc_core::StaticMetadataInitCanary() ==
                   grpc_core::kGrpcStaticMetadataInitCanary);
  GPR_DEBUG_ASSERT(grpc_core::g_static_metadata_slice_refcounts != nullptr);
  return grpc_core::g_static_metadata_slice_refcounts;
}

#define GRPC_IS_STATIC_METADATA_STRING(slice) \
  ((slice).refcount != NULL &&                \
   (slice).refcount->GetType() == grpc_slice_refcount::Type::STATIC)

#define GRPC_STATIC_METADATA_INDEX(static_slice)                              \
  (reinterpret_cast<grpc_core::StaticSliceRefcount*>((static_slice).refcount) \
       ->index)

#define GRPC_STATIC_MDELEM_COUNT 85

namespace grpc_core {
extern StaticMetadata* g_static_mdelem_table;
extern grpc_mdelem* g_static_mdelem_manifested;
}  // namespace grpc_core
inline grpc_core::StaticMetadata* grpc_static_mdelem_table() {
  GPR_DEBUG_ASSERT(grpc_core::StaticMetadataInitCanary() ==
                   grpc_core::kGrpcStaticMetadataInitCanary);
  GPR_DEBUG_ASSERT(grpc_core::g_static_mdelem_table != nullptr);
  return grpc_core::g_static_mdelem_table;
}
inline grpc_mdelem* grpc_static_mdelem_manifested() {
  GPR_DEBUG_ASSERT(grpc_core::StaticMetadataInitCanary() ==
                   grpc_core::kGrpcStaticMetadataInitCanary);
  GPR_DEBUG_ASSERT(grpc_core::g_static_mdelem_manifested != nullptr);
  return grpc_core::g_static_mdelem_manifested;
}

extern uintptr_t grpc_static_mdelem_user_data[GRPC_STATIC_MDELEM_COUNT];
/* ":authority": "" */
#define GRPC_MDELEM_AUTHORITY_EMPTY (grpc_static_mdelem_manifested()[0])
/* ":method": "GET" */
#define GRPC_MDELEM_METHOD_GET (grpc_static_mdelem_manifested()[1])
/* ":method": "POST" */
#define GRPC_MDELEM_METHOD_POST (grpc_static_mdelem_manifested()[2])
/* ":path": "/" */
#define GRPC_MDELEM_PATH_SLASH (grpc_static_mdelem_manifested()[3])
/* ":path": "/index.html" */
#define GRPC_MDELEM_PATH_SLASH_INDEX_DOT_HTML \
  (grpc_static_mdelem_manifested()[4])
/* ":scheme": "http" */
#define GRPC_MDELEM_SCHEME_HTTP (grpc_static_mdelem_manifested()[5])
/* ":scheme": "https" */
#define GRPC_MDELEM_SCHEME_HTTPS (grpc_static_mdelem_manifested()[6])
/* ":status": "200" */
#define GRPC_MDELEM_STATUS_200 (grpc_static_mdelem_manifested()[7])
/* ":status": "204" */
#define GRPC_MDELEM_STATUS_204 (grpc_static_mdelem_manifested()[8])
/* ":status": "206" */
#define GRPC_MDELEM_STATUS_206 (grpc_static_mdelem_manifested()[9])
/* ":status": "304" */
#define GRPC_MDELEM_STATUS_304 (grpc_static_mdelem_manifested()[10])
/* ":status": "400" */
#define GRPC_MDELEM_STATUS_400 (grpc_static_mdelem_manifested()[11])
/* ":status": "404" */
#define GRPC_MDELEM_STATUS_404 (grpc_static_mdelem_manifested()[12])
/* ":status": "500" */
#define GRPC_MDELEM_STATUS_500 (grpc_static_mdelem_manifested()[13])
/* "accept-charset": "" */
#define GRPC_MDELEM_ACCEPT_CHARSET_EMPTY (grpc_static_mdelem_manifested()[14])
/* "accept-encoding": "gzip, deflate" */
#define GRPC_MDELEM_ACCEPT_ENCODING_GZIP_COMMA_DEFLATE \
  (grpc_static_mdelem_manifested()[15])
/* "accept-language": "" */
#define GRPC_MDELEM_ACCEPT_LANGUAGE_EMPTY (grpc_static_mdelem_manifested()[16])
/* "accept-ranges": "" */
#define GRPC_MDELEM_ACCEPT_RANGES_EMPTY (grpc_static_mdelem_manifested()[17])
/* "accept": "" */
#define GRPC_MDELEM_ACCEPT_EMPTY (grpc_static_mdelem_manifested()[18])
/* "access-control-allow-origin": "" */
#define GRPC_MDELEM_ACCESS_CONTROL_ALLOW_ORIGIN_EMPTY \
  (grpc_static_mdelem_manifested()[19])
/* "age": "" */
#define GRPC_MDELEM_AGE_EMPTY (grpc_static_mdelem_manifested()[20])
/* "allow": "" */
#define GRPC_MDELEM_ALLOW_EMPTY (grpc_static_mdelem_manifested()[21])
/* "authorization": "" */
#define GRPC_MDELEM_AUTHORIZATION_EMPTY (grpc_static_mdelem_manifested()[22])
/* "cache-control": "" */
#define GRPC_MDELEM_CACHE_CONTROL_EMPTY (grpc_static_mdelem_manifested()[23])
/* "content-disposition": "" */
#define GRPC_MDELEM_CONTENT_DISPOSITION_EMPTY \
  (grpc_static_mdelem_manifested()[24])
/* "content-encoding": "" */
#define GRPC_MDELEM_CONTENT_ENCODING_EMPTY (grpc_static_mdelem_manifested()[25])
/* "content-language": "" */
#define GRPC_MDELEM_CONTENT_LANGUAGE_EMPTY (grpc_static_mdelem_manifested()[26])
/* "content-length": "" */
#define GRPC_MDELEM_CONTENT_LENGTH_EMPTY (grpc_static_mdelem_manifested()[27])
/* "content-location": "" */
#define GRPC_MDELEM_CONTENT_LOCATION_EMPTY (grpc_static_mdelem_manifested()[28])
/* "content-range": "" */
#define GRPC_MDELEM_CONTENT_RANGE_EMPTY (grpc_static_mdelem_manifested()[29])
/* "content-type": "" */
#define GRPC_MDELEM_CONTENT_TYPE_EMPTY (grpc_static_mdelem_manifested()[30])
/* "cookie": "" */
#define GRPC_MDELEM_COOKIE_EMPTY (grpc_static_mdelem_manifested()[31])
/* "date": "" */
#define GRPC_MDELEM_DATE_EMPTY (grpc_static_mdelem_manifested()[32])
/* "etag": "" */
#define GRPC_MDELEM_ETAG_EMPTY (grpc_static_mdelem_manifested()[33])
/* "expect": "" */
#define GRPC_MDELEM_EXPECT_EMPTY (grpc_static_mdelem_manifested()[34])
/* "expires": "" */
#define GRPC_MDELEM_EXPIRES_EMPTY (grpc_static_mdelem_manifested()[35])
/* "from": "" */
#define GRPC_MDELEM_FROM_EMPTY (grpc_static_mdelem_manifested()[36])
/* "host": "" */
#define GRPC_MDELEM_HOST_EMPTY (grpc_static_mdelem_manifested()[37])
/* "if-match": "" */
#define GRPC_MDELEM_IF_MATCH_EMPTY (grpc_static_mdelem_manifested()[38])
/* "if-modified-since": "" */
#define GRPC_MDELEM_IF_MODIFIED_SINCE_EMPTY \
  (grpc_static_mdelem_manifested()[39])
/* "if-none-match": "" */
#define GRPC_MDELEM_IF_NONE_MATCH_EMPTY (grpc_static_mdelem_manifested()[40])
/* "if-range": "" */
#define GRPC_MDELEM_IF_RANGE_EMPTY (grpc_static_mdelem_manifested()[41])
/* "if-unmodified-since": "" */
#define GRPC_MDELEM_IF_UNMODIFIED_SINCE_EMPTY \
  (grpc_static_mdelem_manifested()[42])
/* "last-modified": "" */
#define GRPC_MDELEM_LAST_MODIFIED_EMPTY (grpc_static_mdelem_manifested()[43])
/* "link": "" */
#define GRPC_MDELEM_LINK_EMPTY (grpc_static_mdelem_manifested()[44])
/* "location": "" */
#define GRPC_MDELEM_LOCATION_EMPTY (grpc_static_mdelem_manifested()[45])
/* "max-forwards": "" */
#define GRPC_MDELEM_MAX_FORWARDS_EMPTY (grpc_static_mdelem_manifested()[46])
/* "proxy-authenticate": "" */
#define GRPC_MDELEM_PROXY_AUTHENTICATE_EMPTY \
  (grpc_static_mdelem_manifested()[47])
/* "proxy-authorization": "" */
#define GRPC_MDELEM_PROXY_AUTHORIZATION_EMPTY \
  (grpc_static_mdelem_manifested()[48])
/* "range": "" */
#define GRPC_MDELEM_RANGE_EMPTY (grpc_static_mdelem_manifested()[49])
/* "referer": "" */
#define GRPC_MDELEM_REFERER_EMPTY (grpc_static_mdelem_manifested()[50])
/* "refresh": "" */
#define GRPC_MDELEM_REFRESH_EMPTY (grpc_static_mdelem_manifested()[51])
/* "retry-after": "" */
#define GRPC_MDELEM_RETRY_AFTER_EMPTY (grpc_static_mdelem_manifested()[52])
/* "server": "" */
#define GRPC_MDELEM_SERVER_EMPTY (grpc_static_mdelem_manifested()[53])
/* "set-cookie": "" */
#define GRPC_MDELEM_SET_COOKIE_EMPTY (grpc_static_mdelem_manifested()[54])
/* "strict-transport-security": "" */
#define GRPC_MDELEM_STRICT_TRANSPORT_SECURITY_EMPTY \
  (grpc_static_mdelem_manifested()[55])
/* "transfer-encoding": "" */
#define GRPC_MDELEM_TRANSFER_ENCODING_EMPTY \
  (grpc_static_mdelem_manifested()[56])
/* "user-agent": "" */
#define GRPC_MDELEM_USER_AGENT_EMPTY (grpc_static_mdelem_manifested()[57])
/* "vary": "" */
#define GRPC_MDELEM_VARY_EMPTY (grpc_static_mdelem_manifested()[58])
/* "via": "" */
#define GRPC_MDELEM_VIA_EMPTY (grpc_static_mdelem_manifested()[59])
/* "www-authenticate": "" */
#define GRPC_MDELEM_WWW_AUTHENTICATE_EMPTY (grpc_static_mdelem_manifested()[60])
/* "grpc-status": "0" */
#define GRPC_MDELEM_GRPC_STATUS_0 (grpc_static_mdelem_manifested()[61])
/* "grpc-status": "1" */
#define GRPC_MDELEM_GRPC_STATUS_1 (grpc_static_mdelem_manifested()[62])
/* "grpc-status": "2" */
#define GRPC_MDELEM_GRPC_STATUS_2 (grpc_static_mdelem_manifested()[63])
/* "grpc-encoding": "identity" */
#define GRPC_MDELEM_GRPC_ENCODING_IDENTITY (grpc_static_mdelem_manifested()[64])
/* "grpc-encoding": "gzip" */
#define GRPC_MDELEM_GRPC_ENCODING_GZIP (grpc_static_mdelem_manifested()[65])
/* "grpc-encoding": "deflate" */
#define GRPC_MDELEM_GRPC_ENCODING_DEFLATE (grpc_static_mdelem_manifested()[66])
/* "te": "trailers" */
#define GRPC_MDELEM_TE_TRAILERS (grpc_static_mdelem_manifested()[67])
/* "content-type": "application/grpc" */
#define GRPC_MDELEM_CONTENT_TYPE_APPLICATION_SLASH_GRPC \
  (grpc_static_mdelem_manifested()[68])
/* ":scheme": "grpc" */
#define GRPC_MDELEM_SCHEME_GRPC (grpc_static_mdelem_manifested()[69])
/* ":method": "PUT" */
#define GRPC_MDELEM_METHOD_PUT (grpc_static_mdelem_manifested()[70])
/* "accept-encoding": "" */
#define GRPC_MDELEM_ACCEPT_ENCODING_EMPTY (grpc_static_mdelem_manifested()[71])
/* "content-encoding": "identity" */
#define GRPC_MDELEM_CONTENT_ENCODING_IDENTITY \
  (grpc_static_mdelem_manifested()[72])
/* "content-encoding": "gzip" */
#define GRPC_MDELEM_CONTENT_ENCODING_GZIP (grpc_static_mdelem_manifested()[73])
/* "lb-cost-bin": "" */
#define GRPC_MDELEM_LB_COST_BIN_EMPTY (grpc_static_mdelem_manifested()[74])
/* "grpc-accept-encoding": "identity" */
#define GRPC_MDELEM_GRPC_ACCEPT_ENCODING_IDENTITY \
  (grpc_static_mdelem_manifested()[75])
/* "grpc-accept-encoding": "deflate" */
#define GRPC_MDELEM_GRPC_ACCEPT_ENCODING_DEFLATE \
  (grpc_static_mdelem_manifested()[76])
/* "grpc-accept-encoding": "identity,deflate" */
#define GRPC_MDELEM_GRPC_ACCEPT_ENCODING_IDENTITY_COMMA_DEFLATE \
  (grpc_static_mdelem_manifested()[77])
/* "grpc-accept-encoding": "gzip" */
#define GRPC_MDELEM_GRPC_ACCEPT_ENCODING_GZIP \
  (grpc_static_mdelem_manifested()[78])
/* "grpc-accept-encoding": "identity,gzip" */
#define GRPC_MDELEM_GRPC_ACCEPT_ENCODING_IDENTITY_COMMA_GZIP \
  (grpc_static_mdelem_manifested()[79])
/* "grpc-accept-encoding": "deflate,gzip" */
#define GRPC_MDELEM_GRPC_ACCEPT_ENCODING_DEFLATE_COMMA_GZIP \
  (grpc_static_mdelem_manifested()[80])
/* "grpc-accept-encoding": "identity,deflate,gzip" */
#define GRPC_MDELEM_GRPC_ACCEPT_ENCODING_IDENTITY_COMMA_DEFLATE_COMMA_GZIP \
  (grpc_static_mdelem_manifested()[81])
/* "accept-encoding": "identity" */
#define GRPC_MDELEM_ACCEPT_ENCODING_IDENTITY \
  (grpc_static_mdelem_manifested()[82])
/* "accept-encoding": "gzip" */
#define GRPC_MDELEM_ACCEPT_ENCODING_GZIP (grpc_static_mdelem_manifested()[83])
/* "accept-encoding": "identity,gzip" */
#define GRPC_MDELEM_ACCEPT_ENCODING_IDENTITY_COMMA_GZIP \
  (grpc_static_mdelem_manifested()[84])

grpc_mdelem grpc_static_mdelem_for_static_strings(intptr_t a, intptr_t b);
typedef enum {
  GRPC_BATCH_PATH,
  GRPC_BATCH_METHOD,
  GRPC_BATCH_STATUS,
  GRPC_BATCH_AUTHORITY,
  GRPC_BATCH_SCHEME,
  GRPC_BATCH_TE,
  GRPC_BATCH_GRPC_MESSAGE,
  GRPC_BATCH_GRPC_STATUS,
  GRPC_BATCH_GRPC_PAYLOAD_BIN,
  GRPC_BATCH_GRPC_ENCODING,
  GRPC_BATCH_GRPC_ACCEPT_ENCODING,
  GRPC_BATCH_GRPC_SERVER_STATS_BIN,
  GRPC_BATCH_GRPC_TAGS_BIN,
  GRPC_BATCH_GRPC_TRACE_BIN,
  GRPC_BATCH_CONTENT_TYPE,
  GRPC_BATCH_CONTENT_ENCODING,
  GRPC_BATCH_ACCEPT_ENCODING,
  GRPC_BATCH_GRPC_INTERNAL_ENCODING_REQUEST,
  GRPC_BATCH_GRPC_INTERNAL_STREAM_ENCODING_REQUEST,
  GRPC_BATCH_USER_AGENT,
  GRPC_BATCH_HOST,
  GRPC_BATCH_GRPC_PREVIOUS_RPC_ATTEMPTS,
  GRPC_BATCH_GRPC_RETRY_PUSHBACK_MS,
  GRPC_BATCH_X_ENDPOINT_LOAD_METRICS_BIN,
  GRPC_BATCH_CALLOUTS_COUNT
} grpc_metadata_batch_callouts_index;

typedef union {
  struct grpc_linked_mdelem* array[GRPC_BATCH_CALLOUTS_COUNT];
  struct {
    struct grpc_linked_mdelem* path;
    struct grpc_linked_mdelem* method;
    struct grpc_linked_mdelem* status;
    struct grpc_linked_mdelem* authority;
    struct grpc_linked_mdelem* scheme;
    struct grpc_linked_mdelem* te;
    struct grpc_linked_mdelem* grpc_message;
    struct grpc_linked_mdelem* grpc_status;
    struct grpc_linked_mdelem* grpc_payload_bin;
    struct grpc_linked_mdelem* grpc_encoding;
    struct grpc_linked_mdelem* grpc_accept_encoding;
    struct grpc_linked_mdelem* grpc_server_stats_bin;
    struct grpc_linked_mdelem* grpc_tags_bin;
    struct grpc_linked_mdelem* grpc_trace_bin;
    struct grpc_linked_mdelem* content_type;
    struct grpc_linked_mdelem* content_encoding;
    struct grpc_linked_mdelem* accept_encoding;
    struct grpc_linked_mdelem* grpc_internal_encoding_request;
    struct grpc_linked_mdelem* grpc_internal_stream_encoding_request;
    struct grpc_linked_mdelem* user_agent;
    struct grpc_linked_mdelem* host;
    struct grpc_linked_mdelem* grpc_previous_rpc_attempts;
    struct grpc_linked_mdelem* grpc_retry_pushback_ms;
    struct grpc_linked_mdelem* x_endpoint_load_metrics_bin;
  } named;
} grpc_metadata_batch_callouts;

#define GRPC_BATCH_INDEX_OF(slice)                                             \
  (GRPC_IS_STATIC_METADATA_STRING((slice)) &&                                  \
           reinterpret_cast<grpc_core::StaticSliceRefcount*>((slice).refcount) \
                   ->index <= static_cast<uint32_t>(GRPC_BATCH_CALLOUTS_COUNT) \
       ? static_cast<grpc_metadata_batch_callouts_index>(                      \
             reinterpret_cast<grpc_core::StaticSliceRefcount*>(                \
                 (slice).refcount)                                             \
                 ->index)                                                      \
       : GRPC_BATCH_CALLOUTS_COUNT)

extern const uint8_t grpc_static_accept_encoding_metadata[8];
#define GRPC_MDELEM_ACCEPT_ENCODING_FOR_ALGORITHMS(algs)                \
  (GRPC_MAKE_MDELEM(&grpc_static_mdelem_table()                         \
                         [grpc_static_accept_encoding_metadata[(algs)]] \
                             .data(),                                   \
                    GRPC_MDELEM_STORAGE_STATIC))

extern const uint8_t grpc_static_accept_stream_encoding_metadata[4];
#define GRPC_MDELEM_ACCEPT_STREAM_ENCODING_FOR_ALGORITHMS(algs)                \
  (GRPC_MAKE_MDELEM(&grpc_static_mdelem_table()                                \
                         [grpc_static_accept_stream_encoding_metadata[(algs)]] \
                             .data(),                                          \
                    GRPC_MDELEM_STORAGE_STATIC))
#endif /* GRPC_CORE_LIB_TRANSPORT_STATIC_METADATA_H */
