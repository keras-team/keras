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

#ifndef GRPC_IMPL_CODEGEN_GRPC_TYPES_H
#define GRPC_IMPL_CODEGEN_GRPC_TYPES_H

#include <grpc/impl/codegen/port_platform.h>

#include <grpc/impl/codegen/compression_types.h>
#include <grpc/impl/codegen/gpr_types.h>
#include <grpc/impl/codegen/slice.h>
#include <grpc/impl/codegen/status.h>

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  GRPC_BB_RAW
  /** Future types may include GRPC_BB_PROTOBUF, etc. */
} grpc_byte_buffer_type;

typedef struct grpc_byte_buffer {
  void* reserved;
  grpc_byte_buffer_type type;
  union grpc_byte_buffer_data {
    struct /* internal */ {
      void* reserved[8];
    } reserved;
    struct grpc_compressed_buffer {
      grpc_compression_algorithm compression;
      grpc_slice_buffer slice_buffer;
    } raw;
  } data;
} grpc_byte_buffer;

/** Completion Queues enable notification of the completion of
 * asynchronous actions. */
typedef struct grpc_completion_queue grpc_completion_queue;

/** An alarm associated with a completion queue. */
typedef struct grpc_alarm grpc_alarm;

/** The Channel interface allows creation of Call objects. */
typedef struct grpc_channel grpc_channel;

/** A server listens to some port and responds to request calls */
typedef struct grpc_server grpc_server;

/** A Call represents an RPC. When created, it is in a configuration state
    allowing properties to be set until it is invoked. After invoke, the Call
    can have messages written to it and read from it. */
typedef struct grpc_call grpc_call;

/** The Socket Mutator interface allows changes on socket options */
typedef struct grpc_socket_mutator grpc_socket_mutator;

/** The Socket Factory interface creates and binds sockets */
typedef struct grpc_socket_factory grpc_socket_factory;

/** Type specifier for grpc_arg */
typedef enum {
  GRPC_ARG_STRING,
  GRPC_ARG_INTEGER,
  GRPC_ARG_POINTER
} grpc_arg_type;

typedef struct grpc_arg_pointer_vtable {
  void* (*copy)(void* p);
  void (*destroy)(void* p);
  int (*cmp)(void* p, void* q);
} grpc_arg_pointer_vtable;

/** A single argument... each argument has a key and a value

    A note on naming keys:
      Keys are namespaced into groups, usually grouped by library, and are
      keys for module XYZ are named XYZ.key1, XYZ.key2, etc. Module names must
      be restricted to the regex [A-Za-z][_A-Za-z0-9]{,15}.
      Key names must be restricted to the regex [A-Za-z][_A-Za-z0-9]{,47}.

    GRPC core library keys are prefixed by grpc.

    Library authors are strongly encouraged to \#define symbolic constants for
    their keys so that it's possible to change them in the future. */
typedef struct {
  grpc_arg_type type;
  char* key;
  union grpc_arg_value {
    char* string;
    int integer;
    struct grpc_arg_pointer {
      void* p;
      const grpc_arg_pointer_vtable* vtable;
    } pointer;
  } value;
} grpc_arg;

/** An array of arguments that can be passed around.

    Used to set optional channel-level configuration.
    These configuration options are modelled as key-value pairs as defined
    by grpc_arg; keys are strings to allow easy backwards-compatible extension
    by arbitrary parties. All evaluation is performed at channel creation
    time (i.e. the keys and values in this structure need only live through the
    creation invocation).

    However, if one of the args has grpc_arg_type==GRPC_ARG_POINTER, then the
    grpc_arg_pointer_vtable must live until the channel args are done being
    used by core (i.e. when the object for use with which they were passed
    is destroyed).

    See the description of the \ref grpc_arg_keys "available args" for more
    details. */
typedef struct {
  size_t num_args;
  grpc_arg* args;
} grpc_channel_args;

/** \defgroup grpc_arg_keys
 * Channel argument keys.
 * \{
 */
/** If non-zero, enable census for tracing and stats collection. */
#define GRPC_ARG_ENABLE_CENSUS "grpc.census"
/** If non-zero, enable load reporting. */
#define GRPC_ARG_ENABLE_LOAD_REPORTING "grpc.loadreporting"
/** Request that optional features default to off (regardless of what they
    usually default to) - to enable tight control over what gets enabled */
#define GRPC_ARG_MINIMAL_STACK "grpc.minimal_stack"
/** Maximum number of concurrent incoming streams to allow on a http2
    connection. Int valued. */
#define GRPC_ARG_MAX_CONCURRENT_STREAMS "grpc.max_concurrent_streams"
/** Maximum message length that the channel can receive. Int valued, bytes.
    -1 means unlimited. */
#define GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH "grpc.max_receive_message_length"
/** \deprecated For backward compatibility.
 * Use GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH instead. */
#define GRPC_ARG_MAX_MESSAGE_LENGTH GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH
/** Maximum message length that the channel can send. Int valued, bytes.
    -1 means unlimited. */
#define GRPC_ARG_MAX_SEND_MESSAGE_LENGTH "grpc.max_send_message_length"
/** Maximum time that a channel may have no outstanding rpcs, after which the
 * server will close the connection. Int valued, milliseconds. INT_MAX means
 * unlimited. */
#define GRPC_ARG_MAX_CONNECTION_IDLE_MS "grpc.max_connection_idle_ms"
/** Maximum time that a channel may exist. Int valued, milliseconds.
 * INT_MAX means unlimited. */
#define GRPC_ARG_MAX_CONNECTION_AGE_MS "grpc.max_connection_age_ms"
/** Grace period after the channel reaches its max age. Int valued,
   milliseconds. INT_MAX means unlimited. */
#define GRPC_ARG_MAX_CONNECTION_AGE_GRACE_MS "grpc.max_connection_age_grace_ms"
/** Timeout after the last RPC finishes on the client channel at which the
 * channel goes back into IDLE state. Int valued, milliseconds. INT_MAX means
 * unlimited. The default value is 30 minutes and the min value is 1 second. */
#define GRPC_ARG_CLIENT_IDLE_TIMEOUT_MS "grpc.client_idle_timeout_ms"
/** Enable/disable support for per-message compression. Defaults to 1, unless
    GRPC_ARG_MINIMAL_STACK is enabled, in which case it defaults to 0. */
#define GRPC_ARG_ENABLE_PER_MESSAGE_COMPRESSION "grpc.per_message_compression"
/** Enable/disable support for deadline checking. Defaults to 1, unless
    GRPC_ARG_MINIMAL_STACK is enabled, in which case it defaults to 0 */
#define GRPC_ARG_ENABLE_DEADLINE_CHECKS "grpc.enable_deadline_checking"
/** Initial stream ID for http2 transports. Int valued. */
#define GRPC_ARG_HTTP2_INITIAL_SEQUENCE_NUMBER \
  "grpc.http2.initial_sequence_number"
/** Amount to read ahead on individual streams. Defaults to 64kb, larger
    values can help throughput on high-latency connections.
    NOTE: at some point we'd like to auto-tune this, and this parameter
    will become a no-op. Int valued, bytes. */
#define GRPC_ARG_HTTP2_STREAM_LOOKAHEAD_BYTES "grpc.http2.lookahead_bytes"
/** How much memory to use for hpack decoding. Int valued, bytes. */
#define GRPC_ARG_HTTP2_HPACK_TABLE_SIZE_DECODER \
  "grpc.http2.hpack_table_size.decoder"
/** How much memory to use for hpack encoding. Int valued, bytes. */
#define GRPC_ARG_HTTP2_HPACK_TABLE_SIZE_ENCODER \
  "grpc.http2.hpack_table_size.encoder"
/** How big a frame are we willing to receive via HTTP2.
    Min 16384, max 16777215. Larger values give lower CPU usage for large
    messages, but more head of line blocking for small messages. */
#define GRPC_ARG_HTTP2_MAX_FRAME_SIZE "grpc.http2.max_frame_size"
/** Should BDP probing be performed? */
#define GRPC_ARG_HTTP2_BDP_PROBE "grpc.http2.bdp_probe"
/** Minimum time between sending successive ping frames without receiving any
    data frame, Int valued, milliseconds. */
#define GRPC_ARG_HTTP2_MIN_SENT_PING_INTERVAL_WITHOUT_DATA_MS \
  "grpc.http2.min_time_between_pings_ms"
/** Minimum allowed time between a server receiving successive ping frames
   without sending any data frame. Int valued, milliseconds */
#define GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS \
  "grpc.http2.min_ping_interval_without_data_ms"
/** Channel arg to override the http2 :scheme header */
#define GRPC_ARG_HTTP2_SCHEME "grpc.http2_scheme"
/** How many pings can we send before needing to send a data frame or header
    frame? (0 indicates that an infinite number of pings can be sent without
    sending a data frame or header frame) */
#define GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA \
  "grpc.http2.max_pings_without_data"
/** How many misbehaving pings the server can bear before sending goaway and
    closing the transport? (0 indicates that the server can bear an infinite
    number of misbehaving pings) */
#define GRPC_ARG_HTTP2_MAX_PING_STRIKES "grpc.http2.max_ping_strikes"
/** How much data are we willing to queue up per stream if
    GRPC_WRITE_BUFFER_HINT is set? This is an upper bound */
#define GRPC_ARG_HTTP2_WRITE_BUFFER_SIZE "grpc.http2.write_buffer_size"
/** Should we allow receipt of true-binary data on http2 connections?
    Defaults to on (1) */
#define GRPC_ARG_HTTP2_ENABLE_TRUE_BINARY "grpc.http2.true_binary"
/** After a duration of this time the client/server pings its peer to see if the
    transport is still alive. Int valued, milliseconds. */
#define GRPC_ARG_KEEPALIVE_TIME_MS "grpc.keepalive_time_ms"
/** After waiting for a duration of this time, if the keepalive ping sender does
    not receive the ping ack, it will close the transport. Int valued,
    milliseconds. */
#define GRPC_ARG_KEEPALIVE_TIMEOUT_MS "grpc.keepalive_timeout_ms"
/** Is it permissible to send keepalive pings without any outstanding streams.
    Int valued, 0(false)/1(true). */
#define GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS \
  "grpc.keepalive_permit_without_calls"
/** Default authority to pass if none specified on call construction. A string.
 * */
#define GRPC_ARG_DEFAULT_AUTHORITY "grpc.default_authority"
/** Primary user agent: goes at the start of the user-agent metadata
    sent on each request. A string. */
#define GRPC_ARG_PRIMARY_USER_AGENT_STRING "grpc.primary_user_agent"
/** Secondary user agent: goes at the end of the user-agent metadata
    sent on each request. A string. */
#define GRPC_ARG_SECONDARY_USER_AGENT_STRING "grpc.secondary_user_agent"
/** The minimum time between subsequent connection attempts, in ms */
#define GRPC_ARG_MIN_RECONNECT_BACKOFF_MS "grpc.min_reconnect_backoff_ms"
/** The maximum time between subsequent connection attempts, in ms */
#define GRPC_ARG_MAX_RECONNECT_BACKOFF_MS "grpc.max_reconnect_backoff_ms"
/** The time between the first and second connection attempts, in ms */
#define GRPC_ARG_INITIAL_RECONNECT_BACKOFF_MS \
  "grpc.initial_reconnect_backoff_ms"
/** Minimum amount of time between DNS resolutions, in ms */
#define GRPC_ARG_DNS_MIN_TIME_BETWEEN_RESOLUTIONS_MS \
  "grpc.dns_min_time_between_resolutions_ms"
/** The timeout used on servers for finishing handshaking on an incoming
    connection.  Defaults to 120 seconds. */
#define GRPC_ARG_SERVER_HANDSHAKE_TIMEOUT_MS "grpc.server_handshake_timeout_ms"
/** This *should* be used for testing only.
    The caller of the secure_channel_create functions may override the target
    name used for SSL host name checking using this channel argument which is of
    type \a GRPC_ARG_STRING. If this argument is not specified, the name used
    for SSL host name checking will be the target parameter (assuming that the
    secure channel is an SSL channel). If this parameter is specified and the
    underlying is not an SSL channel, it will just be ignored. */
#define GRPC_SSL_TARGET_NAME_OVERRIDE_ARG "grpc.ssl_target_name_override"
/** If non-zero, a pointer to a session cache (a pointer of type
    grpc_ssl_session_cache*). (use grpc_ssl_session_cache_arg_vtable() to fetch
    an appropriate pointer arg vtable) */
#define GRPC_SSL_SESSION_CACHE_ARG "grpc.ssl_session_cache"
/** If non-zero, it will determine the maximum frame size used by TSI's frame
 *  protector.
 *
 *  NOTE: Be aware that using a large "max_frame_size" is memory inefficient
 *        for non-zerocopy protectors. Also, increasing this value above 1MiB
 *        can break old binaries that don't support larger than 1MiB frame
 *        size. */
#define GRPC_ARG_TSI_MAX_FRAME_SIZE "grpc.tsi.max_frame_size"
/** Maximum metadata size, in bytes. Note this limit applies to the max sum of
    all metadata key-value entries in a batch of headers. */
#define GRPC_ARG_MAX_METADATA_SIZE "grpc.max_metadata_size"
/** If non-zero, allow the use of SO_REUSEPORT if it's available (default 1) */
#define GRPC_ARG_ALLOW_REUSEPORT "grpc.so_reuseport"
/** If non-zero, a pointer to a buffer pool (a pointer of type
 * grpc_resource_quota*). (use grpc_resource_quota_arg_vtable() to fetch an
 * appropriate pointer arg vtable) */
#define GRPC_ARG_RESOURCE_QUOTA "grpc.resource_quota"
/** If non-zero, expand wildcard addresses to a list of local addresses. */
#define GRPC_ARG_EXPAND_WILDCARD_ADDRS "grpc.expand_wildcard_addrs"
/** Service config data in JSON form.
    This value will be ignored if the name resolver returns a service config. */
#define GRPC_ARG_SERVICE_CONFIG "grpc.service_config"
/** Disable looking up the service config via the name resolver. */
#define GRPC_ARG_SERVICE_CONFIG_DISABLE_RESOLUTION \
  "grpc.service_config_disable_resolution"
/** LB policy name. */
#define GRPC_ARG_LB_POLICY_NAME "grpc.lb_policy_name"
/** The grpc_socket_mutator instance that set the socket options. A pointer. */
#define GRPC_ARG_SOCKET_MUTATOR "grpc.socket_mutator"
/** The grpc_socket_factory instance to create and bind sockets. A pointer. */
#define GRPC_ARG_SOCKET_FACTORY "grpc.socket_factory"
/** The maximum amount of memory used by trace events per channel trace node.
 * Once the maximum is reached, subsequent events will evict the oldest events
 * from the buffer. The unit for this knob is bytes. Setting it to zero causes
 * channel tracing to be disabled. */
#define GRPC_ARG_MAX_CHANNEL_TRACE_EVENT_MEMORY_PER_NODE \
  "grpc.max_channel_trace_event_memory_per_node"
/** If non-zero, gRPC library will track stats and information at at per channel
 * level. Disabling channelz naturally disables channel tracing. The default
 * is for channelz to be enabled. */
#define GRPC_ARG_ENABLE_CHANNELZ "grpc.enable_channelz"
/** If non-zero, Cronet transport will coalesce packets to fewer frames
 * when possible. */
#define GRPC_ARG_USE_CRONET_PACKET_COALESCING \
  "grpc.use_cronet_packet_coalescing"
/** Channel arg (integer) setting how large a slice to try and read from the
   wire each time recvmsg (or equivalent) is called **/
#define GRPC_ARG_TCP_READ_CHUNK_SIZE "grpc.experimental.tcp_read_chunk_size"
/** Note this is not a "channel arg" key. This is the default slice size to use
 * when trying to read from the wire if the GRPC_ARG_TCP_READ_CHUNK_SIZE
 * channel arg is unspecified. */
#define GRPC_TCP_DEFAULT_READ_SLICE_SIZE 8192
#define GRPC_ARG_TCP_MIN_READ_CHUNK_SIZE \
  "grpc.experimental.tcp_min_read_chunk_size"
#define GRPC_ARG_TCP_MAX_READ_CHUNK_SIZE \
  "grpc.experimental.tcp_max_read_chunk_size"
/* TCP TX Zerocopy enable state: zero is disabled, non-zero is enabled. By
   default, it is disabled. */
#define GRPC_ARG_TCP_TX_ZEROCOPY_ENABLED \
  "grpc.experimental.tcp_tx_zerocopy_enabled"
/* TCP TX Zerocopy send threshold: only zerocopy if >= this many bytes sent. By
   default, this is set to 16KB. */
#define GRPC_ARG_TCP_TX_ZEROCOPY_SEND_BYTES_THRESHOLD \
  "grpc.experimental.tcp_tx_zerocopy_send_bytes_threshold"
/* TCP TX Zerocopy max simultaneous sends: limit for maximum number of pending
   calls to tcp_write() using zerocopy. A tcp_write() is considered pending
   until the kernel performs the zerocopy-done callback for all sendmsg() calls
   issued by the tcp_write(). By default, this is set to 4. */
#define GRPC_ARG_TCP_TX_ZEROCOPY_MAX_SIMULT_SENDS \
  "grpc.experimental.tcp_tx_zerocopy_max_simultaneous_sends"
/* Timeout in milliseconds to use for calls to the grpclb load balancer.
   If 0 or unset, the balancer calls will have no deadline. */
#define GRPC_ARG_GRPCLB_CALL_TIMEOUT_MS "grpc.grpclb_call_timeout_ms"
/* Timeout in milliseconds to wait for the serverlist from the grpclb load
   balancer before using fallback backend addresses from the resolver.
   If 0, enter fallback mode immediately. Default value is 10000. */
#define GRPC_ARG_GRPCLB_FALLBACK_TIMEOUT_MS "grpc.grpclb_fallback_timeout_ms"
/* Timeout in milliseconds to wait for the serverlist from the xDS load
   balancer before using fallback backend addresses from the resolver.
   If 0, enter fallback mode immediately. Default value is 10000. */
#define GRPC_ARG_XDS_FALLBACK_TIMEOUT_MS "grpc.xds_fallback_timeout_ms"
/* Time in milliseconds to wait before a locality is deleted after it's removed
   from the received EDS update. If 0, delete the locality immediately. Default
   value is 15 minutes. */
#define GRPC_ARG_LOCALITY_RETENTION_INTERVAL_MS \
  "grpc.xds_locality_retention_interval_ms"
/* Timeout in milliseconds to wait for the localities of a specific priority to
   complete their initial connection attempt before xDS fails over to the next
   priority. Specifically, the connection attempt of a priority is considered
   completed when any locality of that priority is ready or all the localities
   of that priority fail to connect. If 0, failover happens immediately. Default
   value is 10 seconds. */
#define GRPC_ARG_XDS_FAILOVER_TIMEOUT_MS "grpc.xds_failover_timeout_ms"
/** If non-zero, grpc server's cronet compression workaround will be enabled */
#define GRPC_ARG_WORKAROUND_CRONET_COMPRESSION \
  "grpc.workaround.cronet_compression"
/** String defining the optimization target for a channel.
    Can be: "latency"    - attempt to minimize latency at the cost of throughput
            "blend"      - try to balance latency and throughput
            "throughput" - attempt to maximize throughput at the expense of
                           latency
    Defaults to "blend". In the current implementation "blend" is equivalent to
    "latency". */
#define GRPC_ARG_OPTIMIZATION_TARGET "grpc.optimization_target"
/** If set to zero, disables retry behavior. Otherwise, transparent retries
    are enabled for all RPCs, and configurable retries are enabled when they
    are configured via the service config. For details, see:
      https://github.com/grpc/proposal/blob/master/A6-client-retries.md
 */
#define GRPC_ARG_ENABLE_RETRIES "grpc.enable_retries"
/** Per-RPC retry buffer size, in bytes. Default is 256 KiB. */
#define GRPC_ARG_PER_RPC_RETRY_BUFFER_SIZE "grpc.per_rpc_retry_buffer_size"
/** Channel arg that carries the bridged objective c object for custom metrics
 * logging filter. */
#define GRPC_ARG_MOBILE_LOG_CONTEXT "grpc.mobile_log_context"
/** If non-zero, client authority filter is disabled for the channel */
#define GRPC_ARG_DISABLE_CLIENT_AUTHORITY_FILTER \
  "grpc.disable_client_authority_filter"
/** If set to zero, disables use of http proxies. Enabled by default. */
#define GRPC_ARG_ENABLE_HTTP_PROXY "grpc.enable_http_proxy"
/** If set to non zero, surfaces the user agent string to the server. User
    agent is surfaced by default. */
#define GRPC_ARG_SURFACE_USER_AGENT "grpc.surface_user_agent"
/** If set, inhibits health checking (which may be enabled via the
 *  service config.) */
#define GRPC_ARG_INHIBIT_HEALTH_CHECKING "grpc.inhibit_health_checking"
/** If set, the channel's resolver is allowed to query for SRV records.
 * For example, this is useful as a way to enable the "grpclb"
 * load balancing policy. Note that this only works with the "ares"
 * DNS resolver, and isn't supported by the "native" DNS resolver. */
#define GRPC_ARG_DNS_ENABLE_SRV_QUERIES "grpc.dns_enable_srv_queries"
/** If set, determines an upper bound on the number of milliseconds that the
 * c-ares based DNS resolver will wait on queries before cancelling them.
 * The default value is 120,000. Setting this to "0" will disable the
 * overall timeout entirely. Note that this doesn't include internal c-ares
 * timeouts/backoff/retry logic, and so the actual DNS resolution may time out
 * sooner than the value specified here. */
#define GRPC_ARG_DNS_ARES_QUERY_TIMEOUT_MS "grpc.dns_ares_query_timeout"
/** If set, uses a local subchannel pool within the channel. Otherwise, uses the
 * global subchannel pool. */
#define GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL "grpc.use_local_subchannel_pool"
/** gRPC Objective-C channel pooling domain string. */
#define GRPC_ARG_CHANNEL_POOL_DOMAIN "grpc.channel_pooling_domain"
/** gRPC Objective-C channel pooling id. */
#define GRPC_ARG_CHANNEL_ID "grpc.channel_id"
/** \} */

/** Result of a grpc call. If the caller satisfies the prerequisites of a
    particular operation, the grpc_call_error returned will be GRPC_CALL_OK.
    Receiving any other value listed here is an indication of a bug in the
    caller. */
typedef enum grpc_call_error {
  /** everything went ok */
  GRPC_CALL_OK = 0,
  /** something failed, we don't know what */
  GRPC_CALL_ERROR,
  /** this method is not available on the server */
  GRPC_CALL_ERROR_NOT_ON_SERVER,
  /** this method is not available on the client */
  GRPC_CALL_ERROR_NOT_ON_CLIENT,
  /** this method must be called before server_accept */
  GRPC_CALL_ERROR_ALREADY_ACCEPTED,
  /** this method must be called before invoke */
  GRPC_CALL_ERROR_ALREADY_INVOKED,
  /** this method must be called after invoke */
  GRPC_CALL_ERROR_NOT_INVOKED,
  /** this call is already finished
      (writes_done or write_status has already been called) */
  GRPC_CALL_ERROR_ALREADY_FINISHED,
  /** there is already an outstanding read/write operation on the call */
  GRPC_CALL_ERROR_TOO_MANY_OPERATIONS,
  /** the flags value was illegal for this call */
  GRPC_CALL_ERROR_INVALID_FLAGS,
  /** invalid metadata was passed to this call */
  GRPC_CALL_ERROR_INVALID_METADATA,
  /** invalid message was passed to this call */
  GRPC_CALL_ERROR_INVALID_MESSAGE,
  /** completion queue for notification has not been registered
   * with the server */
  GRPC_CALL_ERROR_NOT_SERVER_COMPLETION_QUEUE,
  /** this batch of operations leads to more operations than allowed */
  GRPC_CALL_ERROR_BATCH_TOO_BIG,
  /** payload type requested is not the type registered */
  GRPC_CALL_ERROR_PAYLOAD_TYPE_MISMATCH,
  /** completion queue has been shutdown */
  GRPC_CALL_ERROR_COMPLETION_QUEUE_SHUTDOWN
} grpc_call_error;

/** Default send/receive message size limits in bytes. -1 for unlimited. */
/** TODO(roth) Make this match the default receive limit after next release */
#define GRPC_DEFAULT_MAX_SEND_MESSAGE_LENGTH -1
#define GRPC_DEFAULT_MAX_RECV_MESSAGE_LENGTH (4 * 1024 * 1024)

/** Write Flags: */
/** Hint that the write may be buffered and need not go out on the wire
    immediately. GRPC is free to buffer the message until the next non-buffered
    write, or until writes_done, but it need not buffer completely or at all. */
#define GRPC_WRITE_BUFFER_HINT (0x00000001u)
/** Force compression to be disabled for a particular write
    (start_write/add_metadata). Illegal on invoke/accept. */
#define GRPC_WRITE_NO_COMPRESS (0x00000002u)
/** Force this message to be written to the socket before completing it */
#define GRPC_WRITE_THROUGH (0x00000004u)
/** Mask of all valid flags. */
#define GRPC_WRITE_USED_MASK \
  (GRPC_WRITE_BUFFER_HINT | GRPC_WRITE_NO_COMPRESS | GRPC_WRITE_THROUGH)

/** Initial metadata flags */
/** Signal that the call is idempotent */
#define GRPC_INITIAL_METADATA_IDEMPOTENT_REQUEST (0x00000010u)
/** Signal that the call should not return UNAVAILABLE before it has started */
#define GRPC_INITIAL_METADATA_WAIT_FOR_READY (0x00000020u)
/** Signal that the call is cacheable. GRPC is free to use GET verb */
#define GRPC_INITIAL_METADATA_CACHEABLE_REQUEST (0x00000040u)
/** Signal that GRPC_INITIAL_METADATA_WAIT_FOR_READY was explicitly set
    by the calling application. */
#define GRPC_INITIAL_METADATA_WAIT_FOR_READY_EXPLICITLY_SET (0x00000080u)
/** Signal that the initial metadata should be corked */
#define GRPC_INITIAL_METADATA_CORKED (0x00000100u)

/** Mask of all valid flags */
#define GRPC_INITIAL_METADATA_USED_MASK                  \
  (GRPC_INITIAL_METADATA_IDEMPOTENT_REQUEST |            \
   GRPC_INITIAL_METADATA_WAIT_FOR_READY |                \
   GRPC_INITIAL_METADATA_CACHEABLE_REQUEST |             \
   GRPC_INITIAL_METADATA_WAIT_FOR_READY_EXPLICITLY_SET | \
   GRPC_INITIAL_METADATA_CORKED | GRPC_WRITE_THROUGH)

/** A single metadata element */
typedef struct grpc_metadata {
  /** the key, value values are expected to line up with grpc_mdelem: if
     changing them, update metadata.h at the same time. */
  grpc_slice key;
  grpc_slice value;

  uint32_t flags;

  /** The following fields are reserved for grpc internal use.
      There is no need to initialize them, and they will be set to garbage
      during calls to grpc. */
  struct /* internal */ {
    void* obfuscated[4];
  } internal_data;
} grpc_metadata;

/** The type of completion (for grpc_event) */
typedef enum grpc_completion_type {
  /** Shutting down */
  GRPC_QUEUE_SHUTDOWN,
  /** No event before timeout */
  GRPC_QUEUE_TIMEOUT,
  /** Operation completion */
  GRPC_OP_COMPLETE
} grpc_completion_type;

/** The result of an operation.

    Returned by a completion queue when the operation started with tag. */
typedef struct grpc_event {
  /** The type of the completion. */
  grpc_completion_type type;
  /** If the grpc_completion_type is GRPC_OP_COMPLETE, this field indicates
      whether the operation was successful or not; 0 in case of failure and
      non-zero in case of success.
      If grpc_completion_type is GRPC_QUEUE_SHUTDOWN or GRPC_QUEUE_TIMEOUT, this
      field is guaranteed to be 0 */
  int success;
  /** The tag passed to grpc_call_start_batch etc to start this operation.
      *Only* GRPC_OP_COMPLETE has a tag. For all other grpc_completion_type
      values, tag is uninitialized. */
  void* tag;
} grpc_event;

typedef struct {
  size_t count;
  size_t capacity;
  grpc_metadata* metadata;
} grpc_metadata_array;

typedef struct {
  grpc_slice method;
  grpc_slice host;
  gpr_timespec deadline;
  uint32_t flags;
  void* reserved;
} grpc_call_details;

typedef enum {
  /** Send initial metadata: one and only one instance MUST be sent for each
      call, unless the call was cancelled - in which case this can be skipped.
      This op completes after all bytes of metadata have been accepted by
      outgoing flow control. */
  GRPC_OP_SEND_INITIAL_METADATA = 0,
  /** Send a message: 0 or more of these operations can occur for each call.
      This op completes after all bytes for the message have been accepted by
      outgoing flow control. */
  GRPC_OP_SEND_MESSAGE,
  /** Send a close from the client: one and only one instance MUST be sent from
      the client, unless the call was cancelled - in which case this can be
      skipped. This op completes after all bytes for the call
      (including the close) have passed outgoing flow control. */
  GRPC_OP_SEND_CLOSE_FROM_CLIENT,
  /** Send status from the server: one and only one instance MUST be sent from
      the server unless the call was cancelled - in which case this can be
      skipped. This op completes after all bytes for the call
      (including the status) have passed outgoing flow control. */
  GRPC_OP_SEND_STATUS_FROM_SERVER,
  /** Receive initial metadata: one and only one MUST be made on the client,
      must not be made on the server.
      This op completes after all initial metadata has been read from the
      peer. */
  GRPC_OP_RECV_INITIAL_METADATA,
  /** Receive a message: 0 or more of these operations can occur for each call.
      This op completes after all bytes of the received message have been
      read, or after a half-close has been received on this call. */
  GRPC_OP_RECV_MESSAGE,
  /** Receive status on the client: one and only one must be made on the client.
      This operation always succeeds, meaning ops paired with this operation
      will also appear to succeed, even though they may not have. In that case
      the status will indicate some failure.
      This op completes after all activity on the call has completed. */
  GRPC_OP_RECV_STATUS_ON_CLIENT,
  /** Receive close on the server: one and only one must be made on the
      server. This op completes after the close has been received by the
      server. This operation always succeeds, meaning ops paired with
      this operation will also appear to succeed, even though they may not
      have. */
  GRPC_OP_RECV_CLOSE_ON_SERVER
} grpc_op_type;

struct grpc_byte_buffer;

/** Operation data: one field for each op type (except SEND_CLOSE_FROM_CLIENT
   which has no arguments) */
typedef struct grpc_op {
  /** Operation type, as defined by grpc_op_type */
  grpc_op_type op;
  /** Write flags bitset for grpc_begin_messages */
  uint32_t flags;
  /** Reserved for future usage */
  void* reserved;
  union grpc_op_data {
    /** Reserved for future usage */
    struct /* internal */ {
      void* reserved[8];
    } reserved;
    struct grpc_op_send_initial_metadata {
      size_t count;
      grpc_metadata* metadata;
      /** If \a is_set, \a compression_level will be used for the call.
       * Otherwise, \a compression_level won't be considered */
      struct grpc_op_send_initial_metadata_maybe_compression_level {
        uint8_t is_set;
        grpc_compression_level level;
      } maybe_compression_level;
    } send_initial_metadata;
    struct grpc_op_send_message {
      /** This op takes ownership of the slices in send_message.  After
       * a call completes, the contents of send_message are not guaranteed
       * and likely empty.  The original owner should still call
       * grpc_byte_buffer_destroy() on this object however.
       */
      struct grpc_byte_buffer* send_message;
    } send_message;
    struct grpc_op_send_status_from_server {
      size_t trailing_metadata_count;
      grpc_metadata* trailing_metadata;
      grpc_status_code status;
      /** optional: set to NULL if no details need sending, non-NULL if they do
       * pointer will not be retained past the start_batch call
       */
      grpc_slice* status_details;
    } send_status_from_server;
    /** ownership of the array is with the caller, but ownership of the elements
        stays with the call object (ie key, value members are owned by the call
        object, recv_initial_metadata->array is owned by the caller).
        After the operation completes, call grpc_metadata_array_destroy on this
        value, or reuse it in a future op. */
    struct grpc_op_recv_initial_metadata {
      grpc_metadata_array* recv_initial_metadata;
    } recv_initial_metadata;
    /** ownership of the byte buffer is moved to the caller; the caller must
        call grpc_byte_buffer_destroy on this value, or reuse it in a future op.
        The returned byte buffer will be NULL if trailing metadata was
        received instead of a message.
       */
    struct grpc_op_recv_message {
      struct grpc_byte_buffer** recv_message;
    } recv_message;
    struct grpc_op_recv_status_on_client {
      /** ownership of the array is with the caller, but ownership of the
          elements stays with the call object (ie key, value members are owned
          by the call object, trailing_metadata->array is owned by the caller).
          After the operation completes, call grpc_metadata_array_destroy on
          this value, or reuse it in a future op. */
      grpc_metadata_array* trailing_metadata;
      grpc_status_code* status;
      grpc_slice* status_details;
      /** If this is not nullptr, it will be populated with the full fidelity
       * error string for debugging purposes. The application is responsible
       * for freeing the data by using gpr_free(). */
      const char** error_string;
    } recv_status_on_client;
    struct grpc_op_recv_close_on_server {
      /** out argument, set to 1 if the call failed in any way (seen as a
          cancellation on the server), or 0 if the call succeeded */
      int* cancelled;
    } recv_close_on_server;
  } data;
} grpc_op;

/** Information requested from the channel. */
typedef struct {
  /** If non-NULL, will be set to point to a string indicating the LB
   * policy name.  Caller takes ownership. */
  char** lb_policy_name;
  /** If non-NULL, will be set to point to a string containing the
   * service config used by the channel in JSON form. */
  char** service_config_json;
} grpc_channel_info;

typedef struct grpc_resource_quota grpc_resource_quota;

/** Completion queues internally MAY maintain a set of file descriptors in a
    structure called 'pollset'. This enum specifies if a completion queue has an
    associated pollset and any restrictions on the type of file descriptors that
    can be present in the pollset.

    I/O progress can only be made when grpc_completion_queue_next() or
    grpc_completion_queue_pluck() are called on the completion queue (unless the
    grpc_cq_polling_type is GRPC_CQ_NON_POLLING) and hence it is very important
    to actively call these APIs */
typedef enum {
  /** The completion queue will have an associated pollset and there is no
      restriction on the type of file descriptors the pollset may contain */
  GRPC_CQ_DEFAULT_POLLING,

  /** Similar to GRPC_CQ_DEFAULT_POLLING except that the completion queues will
      not contain any 'listening file descriptors' (i.e file descriptors used to
      listen to incoming channels) */
  GRPC_CQ_NON_LISTENING,

  /** The completion queue will not have an associated pollset. Note that
      grpc_completion_queue_next() or grpc_completion_queue_pluck() MUST still
      be called to pop events from the completion queue; it is not required to
      call them actively to make I/O progress */
  GRPC_CQ_NON_POLLING
} grpc_cq_polling_type;

/** Specifies the type of APIs to use to pop events from the completion queue */
typedef enum {
  /** Events are popped out by calling grpc_completion_queue_next() API ONLY */
  GRPC_CQ_NEXT,

  /** Events are popped out by calling grpc_completion_queue_pluck() API ONLY*/
  GRPC_CQ_PLUCK,

  /** EXPERIMENTAL: Events trigger a callback specified as the tag */
  GRPC_CQ_CALLBACK
} grpc_cq_completion_type;

/** EXPERIMENTAL: Specifies an interface class to be used as a tag
    for callback-based completion queues. This can be used directly,
    as the first element of a struct in C, or as a base class in C++.
    Its "run" value should be assigned to some non-member function, such as
    a static method. */
typedef struct grpc_experimental_completion_queue_functor {
  /** The run member specifies a function that will be called when this
      tag is extracted from the completion queue. Its arguments will be a
      pointer to this functor and a boolean that indicates whether the
      operation succeeded (non-zero) or failed (zero) */
  void (*functor_run)(struct grpc_experimental_completion_queue_functor*, int);

  /** The inlineable member specifies whether this functor can be run inline.
      This should only be used for trivial internally-defined functors. */
  int inlineable;

  /** The following fields are not API. They are meant for internal use. */
  int internal_success;
  struct grpc_experimental_completion_queue_functor* internal_next;
} grpc_experimental_completion_queue_functor;

/* The upgrade to version 2 is currently experimental. */

#define GRPC_CQ_CURRENT_VERSION 2
#define GRPC_CQ_VERSION_MINIMUM_FOR_CALLBACKABLE 2
typedef struct grpc_completion_queue_attributes {
  /** The version number of this structure. More fields might be added to this
     structure in future. */
  int version; /** Set to GRPC_CQ_CURRENT_VERSION */

  grpc_cq_completion_type cq_completion_type;

  grpc_cq_polling_type cq_polling_type;

  /* END OF VERSION 1 CQ ATTRIBUTES */

  /* EXPERIMENTAL: START OF VERSION 2 CQ ATTRIBUTES */
  /** When creating a callbackable CQ, pass in a functor to get invoked when
   * shutdown is complete */
  grpc_experimental_completion_queue_functor* cq_shutdown_cb;

  /* END OF VERSION 2 CQ ATTRIBUTES */
} grpc_completion_queue_attributes;

/** The completion queue factory structure is opaque to the callers of grpc */
typedef struct grpc_completion_queue_factory grpc_completion_queue_factory;

#ifdef __cplusplus
}
#endif

#endif /* GRPC_IMPL_CODEGEN_GRPC_TYPES_H */
