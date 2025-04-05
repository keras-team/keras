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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_INTERNAL_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_INTERNAL_H

#include <grpc/support/port_platform.h>

#include <assert.h>
#include <stdbool.h>

#include "src/core/ext/transport/chttp2/transport/flow_control.h"
#include "src/core/ext/transport/chttp2/transport/frame.h"
#include "src/core/ext/transport/chttp2/transport/frame_data.h"
#include "src/core/ext/transport/chttp2/transport/frame_goaway.h"
#include "src/core/ext/transport/chttp2/transport/frame_ping.h"
#include "src/core/ext/transport/chttp2/transport/frame_rst_stream.h"
#include "src/core/ext/transport/chttp2/transport/frame_settings.h"
#include "src/core/ext/transport/chttp2/transport/frame_window_update.h"
#include "src/core/ext/transport/chttp2/transport/hpack_encoder.h"
#include "src/core/ext/transport/chttp2/transport/hpack_parser.h"
#include "src/core/ext/transport/chttp2/transport/incoming_metadata.h"
#include "src/core/ext/transport/chttp2/transport/stream_map.h"
#include "src/core/lib/channel/channelz.h"
#include "src/core/lib/compression/stream_compression.h"
#include "src/core/lib/gprpp/manual_constructor.h"
#include "src/core/lib/iomgr/combiner.h"
#include "src/core/lib/iomgr/endpoint.h"
#include "src/core/lib/iomgr/timer.h"
#include "src/core/lib/transport/connectivity_state.h"
#include "src/core/lib/transport/transport_impl.h"

namespace grpc_core {
class ContextList;
}

/* streams are kept in various linked lists depending on what things need to
   happen to them... this enum labels each list */
typedef enum {
  GRPC_CHTTP2_LIST_WRITABLE,
  GRPC_CHTTP2_LIST_WRITING,
  GRPC_CHTTP2_LIST_STALLED_BY_TRANSPORT,
  GRPC_CHTTP2_LIST_STALLED_BY_STREAM,
  /** streams that are waiting to start because there are too many concurrent
      streams on the connection */
  GRPC_CHTTP2_LIST_WAITING_FOR_CONCURRENCY,
  STREAM_LIST_COUNT /* must be last */
} grpc_chttp2_stream_list_id;

typedef enum {
  GRPC_CHTTP2_WRITE_STATE_IDLE,
  GRPC_CHTTP2_WRITE_STATE_WRITING,
  GRPC_CHTTP2_WRITE_STATE_WRITING_WITH_MORE,
} grpc_chttp2_write_state;

typedef enum {
  GRPC_CHTTP2_OPTIMIZE_FOR_LATENCY,
  GRPC_CHTTP2_OPTIMIZE_FOR_THROUGHPUT,
} grpc_chttp2_optimization_target;

typedef enum {
  GRPC_CHTTP2_PCL_INITIATE = 0,
  GRPC_CHTTP2_PCL_NEXT,
  GRPC_CHTTP2_PCL_INFLIGHT,
  GRPC_CHTTP2_PCL_COUNT /* must be last */
} grpc_chttp2_ping_closure_list;

typedef enum {
  GRPC_CHTTP2_INITIATE_WRITE_INITIAL_WRITE,
  GRPC_CHTTP2_INITIATE_WRITE_START_NEW_STREAM,
  GRPC_CHTTP2_INITIATE_WRITE_SEND_MESSAGE,
  GRPC_CHTTP2_INITIATE_WRITE_SEND_INITIAL_METADATA,
  GRPC_CHTTP2_INITIATE_WRITE_SEND_TRAILING_METADATA,
  GRPC_CHTTP2_INITIATE_WRITE_RETRY_SEND_PING,
  GRPC_CHTTP2_INITIATE_WRITE_CONTINUE_PINGS,
  GRPC_CHTTP2_INITIATE_WRITE_GOAWAY_SENT,
  GRPC_CHTTP2_INITIATE_WRITE_RST_STREAM,
  GRPC_CHTTP2_INITIATE_WRITE_CLOSE_FROM_API,
  GRPC_CHTTP2_INITIATE_WRITE_STREAM_FLOW_CONTROL,
  GRPC_CHTTP2_INITIATE_WRITE_TRANSPORT_FLOW_CONTROL,
  GRPC_CHTTP2_INITIATE_WRITE_SEND_SETTINGS,
  GRPC_CHTTP2_INITIATE_WRITE_FLOW_CONTROL_UNSTALLED_BY_SETTING,
  GRPC_CHTTP2_INITIATE_WRITE_FLOW_CONTROL_UNSTALLED_BY_UPDATE,
  GRPC_CHTTP2_INITIATE_WRITE_APPLICATION_PING,
  GRPC_CHTTP2_INITIATE_WRITE_KEEPALIVE_PING,
  GRPC_CHTTP2_INITIATE_WRITE_TRANSPORT_FLOW_CONTROL_UNSTALLED,
  GRPC_CHTTP2_INITIATE_WRITE_PING_RESPONSE,
  GRPC_CHTTP2_INITIATE_WRITE_FORCE_RST_STREAM,
} grpc_chttp2_initiate_write_reason;

const char* grpc_chttp2_initiate_write_reason_string(
    grpc_chttp2_initiate_write_reason reason);

typedef struct {
  grpc_closure_list lists[GRPC_CHTTP2_PCL_COUNT] = {};
  uint64_t inflight_id = 0;
} grpc_chttp2_ping_queue;

typedef struct {
  int max_pings_without_data;
  int max_ping_strikes;
  grpc_millis min_sent_ping_interval_without_data;
  grpc_millis min_recv_ping_interval_without_data;
} grpc_chttp2_repeated_ping_policy;

typedef struct {
  grpc_millis last_ping_sent_time;
  int pings_before_data_required;
  grpc_timer delayed_ping_timer;
  bool is_delayed_ping_timer_set;
} grpc_chttp2_repeated_ping_state;

typedef struct {
  grpc_millis last_ping_recv_time;
  int ping_strikes;
} grpc_chttp2_server_ping_recv_state;

/* deframer state for the overall http2 stream of bytes */
typedef enum {
  /* prefix: one entry per http2 connection prefix byte */
  GRPC_DTS_CLIENT_PREFIX_0 = 0,
  GRPC_DTS_CLIENT_PREFIX_1,
  GRPC_DTS_CLIENT_PREFIX_2,
  GRPC_DTS_CLIENT_PREFIX_3,
  GRPC_DTS_CLIENT_PREFIX_4,
  GRPC_DTS_CLIENT_PREFIX_5,
  GRPC_DTS_CLIENT_PREFIX_6,
  GRPC_DTS_CLIENT_PREFIX_7,
  GRPC_DTS_CLIENT_PREFIX_8,
  GRPC_DTS_CLIENT_PREFIX_9,
  GRPC_DTS_CLIENT_PREFIX_10,
  GRPC_DTS_CLIENT_PREFIX_11,
  GRPC_DTS_CLIENT_PREFIX_12,
  GRPC_DTS_CLIENT_PREFIX_13,
  GRPC_DTS_CLIENT_PREFIX_14,
  GRPC_DTS_CLIENT_PREFIX_15,
  GRPC_DTS_CLIENT_PREFIX_16,
  GRPC_DTS_CLIENT_PREFIX_17,
  GRPC_DTS_CLIENT_PREFIX_18,
  GRPC_DTS_CLIENT_PREFIX_19,
  GRPC_DTS_CLIENT_PREFIX_20,
  GRPC_DTS_CLIENT_PREFIX_21,
  GRPC_DTS_CLIENT_PREFIX_22,
  GRPC_DTS_CLIENT_PREFIX_23,
  /* frame header byte 0... */
  /* must follow from the prefix states */
  GRPC_DTS_FH_0,
  GRPC_DTS_FH_1,
  GRPC_DTS_FH_2,
  GRPC_DTS_FH_3,
  GRPC_DTS_FH_4,
  GRPC_DTS_FH_5,
  GRPC_DTS_FH_6,
  GRPC_DTS_FH_7,
  /* ... frame header byte 8 */
  GRPC_DTS_FH_8,
  /* inside a http2 frame */
  GRPC_DTS_FRAME
} grpc_chttp2_deframe_transport_state;

typedef struct {
  grpc_chttp2_stream* head;
  grpc_chttp2_stream* tail;
} grpc_chttp2_stream_list;

typedef struct {
  grpc_chttp2_stream* next;
  grpc_chttp2_stream* prev;
} grpc_chttp2_stream_link;

/* We keep several sets of connection wide parameters */
typedef enum {
  /* The settings our peer has asked for (and we have acked) */
  GRPC_PEER_SETTINGS = 0,
  /* The settings we'd like to have */
  GRPC_LOCAL_SETTINGS,
  /* The settings we've published to our peer */
  GRPC_SENT_SETTINGS,
  /* The settings the peer has acked */
  GRPC_ACKED_SETTINGS,
  GRPC_NUM_SETTING_SETS
} grpc_chttp2_setting_set;

typedef enum {
  GRPC_CHTTP2_NO_GOAWAY_SEND,
  GRPC_CHTTP2_GOAWAY_SEND_SCHEDULED,
  GRPC_CHTTP2_GOAWAY_SENT,
} grpc_chttp2_sent_goaway_state;

typedef struct grpc_chttp2_write_cb {
  int64_t call_at_byte;
  grpc_closure* closure;
  struct grpc_chttp2_write_cb* next;
} grpc_chttp2_write_cb;

namespace grpc_core {

class Chttp2IncomingByteStream : public ByteStream {
 public:
  Chttp2IncomingByteStream(grpc_chttp2_transport* transport,
                           grpc_chttp2_stream* stream, uint32_t frame_size,
                           uint32_t flags);

  void Orphan() override;

  bool Next(size_t max_size_hint, grpc_closure* on_complete) override;
  grpc_error* Pull(grpc_slice* slice) override;
  void Shutdown(grpc_error* error) override;

  // TODO(roth): When I converted this class to C++, I wanted to make it
  // inherit from RefCounted or InternallyRefCounted instead of continuing
  // to use its own custom ref-counting code.  However, that would require
  // using multiple inheritance, which sucks in general.  And to make matters
  // worse, it causes problems with our New<> and Delete<> wrappers.
  // Specifically, unless RefCounted is first in the list of parent classes,
  // it will see a different value of the address of the object than the one
  // we actually allocated, in which case gpr_free() will be called on a
  // different address than the one we got from gpr_malloc(), thus causing a
  // crash.  Given the fragility of depending on that, as well as a desire to
  // avoid multiple inheritance in general, I've decided to leave this
  // alone for now.  We can revisit this once we're able to link against
  // libc++, at which point we can eliminate New<> and Delete<> and
  // switch to std::shared_ptr<>.
  void Ref() { refs_.Ref(); }
  void Unref() {
    if (GPR_UNLIKELY(refs_.Unref())) {
      delete this;
    }
  }

  void PublishError(grpc_error* error);

  grpc_error* Push(const grpc_slice& slice, grpc_slice* slice_out);

  grpc_error* Finished(grpc_error* error, bool reset_on_error);

  uint32_t remaining_bytes() const { return remaining_bytes_; }

 private:
  static void NextLocked(void* arg, grpc_error* error_ignored);
  static void OrphanLocked(void* arg, grpc_error* error_ignored);

  void MaybeCreateStreamDecompressionCtx();

  grpc_chttp2_transport* transport_;  // Immutable.
  grpc_chttp2_stream* stream_;        // Immutable.

  grpc_core::RefCount refs_;

  /* Accessed only by transport thread when stream->pending_byte_stream == false
   * Accessed only by application thread when stream->pending_byte_stream ==
   * true */
  uint32_t remaining_bytes_;

  /* Accessed only by transport thread when stream->pending_byte_stream == false
   * Accessed only by application thread when stream->pending_byte_stream ==
   * true */
  struct {
    grpc_closure closure;
    size_t max_size_hint;
    grpc_closure* on_complete;
  } next_action_;
  grpc_closure destroy_action_;
};

}  // namespace grpc_core

typedef enum {
  GRPC_CHTTP2_KEEPALIVE_STATE_WAITING,
  GRPC_CHTTP2_KEEPALIVE_STATE_PINGING,
  GRPC_CHTTP2_KEEPALIVE_STATE_DYING,
  GRPC_CHTTP2_KEEPALIVE_STATE_DISABLED,
} grpc_chttp2_keepalive_state;

struct grpc_chttp2_transport {
  grpc_chttp2_transport(const grpc_channel_args* channel_args,
                        grpc_endpoint* ep, bool is_client,
                        grpc_resource_user* resource_user);
  ~grpc_chttp2_transport();

  grpc_transport base; /* must be first */
  grpc_core::RefCount refs;
  grpc_endpoint* ep;
  char* peer_string;

  grpc_resource_user* resource_user;

  grpc_core::Combiner* combiner;

  grpc_closure* notify_on_receive_settings = nullptr;

  /** write execution state of the transport */
  grpc_chttp2_write_state write_state = GRPC_CHTTP2_WRITE_STATE_IDLE;

  /** is the transport destroying itself? */
  uint8_t destroying = false;
  /** has the upper layer closed the transport? */
  grpc_error* closed_with_error = GRPC_ERROR_NONE;

  /** is there a read request to the endpoint outstanding? */
  uint8_t endpoint_reading = 1;

  /** various lists of streams */
  grpc_chttp2_stream_list lists[STREAM_LIST_COUNT] = {};

  /** maps stream id to grpc_chttp2_stream objects */
  grpc_chttp2_stream_map stream_map;

  grpc_closure write_action_begin_locked;
  grpc_closure write_action;
  grpc_closure write_action_end_locked;

  grpc_closure read_action_locked;

  /** incoming read bytes */
  grpc_slice_buffer read_buffer;

  /** address to place a newly accepted stream - set and unset by
      grpc_chttp2_parsing_accept_stream; used by init_stream to
      publish the accepted server stream */
  grpc_chttp2_stream** accepting_stream = nullptr;

  /* accept stream callback */
  void (*accept_stream_cb)(void* user_data, grpc_transport* transport,
                           const void* server_data);
  void* accept_stream_cb_user_data;

  /** connectivity tracking */
  grpc_core::ConnectivityStateTracker state_tracker;

  /** data to write now */
  grpc_slice_buffer outbuf;
  /** hpack encoding */
  grpc_chttp2_hpack_compressor hpack_compressor;
  /** is this a client? */
  bool is_client;

  /** data to write next write */
  grpc_slice_buffer qbuf;

  /** how much data are we willing to buffer when the WRITE_BUFFER_HINT is set?
   */
  uint32_t write_buffer_size = grpc_core::chttp2::kDefaultWindow;

  /** Set to a grpc_error object if a goaway frame is received. By default, set
   * to GRPC_ERROR_NONE */
  grpc_error* goaway_error = GRPC_ERROR_NONE;

  grpc_chttp2_sent_goaway_state sent_goaway_state = GRPC_CHTTP2_NO_GOAWAY_SEND;

  /** are the local settings dirty and need to be sent? */
  bool dirtied_local_settings = true;
  /** have local settings been sent? */
  bool sent_local_settings = false;
  /** bitmask of setting indexes to send out
      Hack: it's common for implementations to assume 65536 bytes initial send
      window -- this should by rights be 0 */
  uint32_t force_send_settings = 1 << GRPC_CHTTP2_SETTINGS_INITIAL_WINDOW_SIZE;
  /** settings values */
  uint32_t settings[GRPC_NUM_SETTING_SETS][GRPC_CHTTP2_NUM_SETTINGS];

  /** what is the next stream id to be allocated by this peer?
      copied to next_stream_id in parsing when parsing commences */
  uint32_t next_stream_id = 0;

  /** last new stream id */
  uint32_t last_new_stream_id = 0;

  /** ping queues for various ping insertion points */
  grpc_chttp2_ping_queue ping_queue = grpc_chttp2_ping_queue();
  grpc_chttp2_repeated_ping_policy ping_policy;
  grpc_chttp2_repeated_ping_state ping_state;
  uint64_t ping_ctr = 0; /* unique id for pings */
  grpc_closure retry_initiate_ping_locked;

  /** ping acks */
  size_t ping_ack_count = 0;
  size_t ping_ack_capacity = 0;
  uint64_t* ping_acks = nullptr;
  grpc_chttp2_server_ping_recv_state ping_recv_state;

  /** parser for headers */
  grpc_chttp2_hpack_parser hpack_parser;
  /** simple one shot parsers */
  union {
    grpc_chttp2_window_update_parser window_update;
    grpc_chttp2_settings_parser settings;
    grpc_chttp2_ping_parser ping;
    grpc_chttp2_rst_stream_parser rst_stream;
  } simple;
  /** parser for goaway frames */
  grpc_chttp2_goaway_parser goaway_parser;

  grpc_core::PolymorphicManualConstructor<
      grpc_core::chttp2::TransportFlowControlBase,
      grpc_core::chttp2::TransportFlowControl,
      grpc_core::chttp2::TransportFlowControlDisabled>
      flow_control;
  /** initial window change. This is tracked as we parse settings frames from
   * the remote peer. If there is a positive delta, then we will make all
   * streams readable since they may have become unstalled */
  int64_t initial_window_update = 0;

  /* deframing */
  grpc_chttp2_deframe_transport_state deframe_state = GRPC_DTS_CLIENT_PREFIX_0;
  uint8_t incoming_frame_type = 0;
  uint8_t incoming_frame_flags = 0;
  uint8_t header_eof = 0;
  bool is_first_frame = true;
  uint32_t expect_continuation_stream_id = 0;
  uint32_t incoming_frame_size = 0;
  uint32_t incoming_stream_id = 0;

  /* active parser */
  void* parser_data = nullptr;
  grpc_chttp2_stream* incoming_stream = nullptr;
  grpc_error* (*parser)(void* parser_user_data, grpc_chttp2_transport* t,
                        grpc_chttp2_stream* s, const grpc_slice& slice,
                        int is_last);

  grpc_chttp2_write_cb* write_cb_pool = nullptr;

  /* bdp estimator */
  grpc_closure next_bdp_ping_timer_expired_locked;
  grpc_closure start_bdp_ping_locked;
  grpc_closure finish_bdp_ping_locked;

  /* if non-NULL, close the transport with this error when writes are finished
   */
  grpc_error* close_transport_on_writes_finished = GRPC_ERROR_NONE;

  /* a list of closures to run after writes are finished */
  grpc_closure_list run_after_write = GRPC_CLOSURE_LIST_INIT;

  /* buffer pool state */
  /** have we scheduled a benign cleanup? */
  bool benign_reclaimer_registered = false;
  /** have we scheduled a destructive cleanup? */
  bool destructive_reclaimer_registered = false;
  /** benign cleanup closure */
  grpc_closure benign_reclaimer_locked;
  /** destructive cleanup closure */
  grpc_closure destructive_reclaimer_locked;

  /* next bdp ping timer */
  bool have_next_bdp_ping_timer = false;
  /** If start_bdp_ping_locked has been called */
  bool bdp_ping_started = false;
  grpc_timer next_bdp_ping_timer;

  /* keep-alive ping support */
  /** Closure to initialize a keepalive ping */
  grpc_closure init_keepalive_ping_locked;
  /** Closure to run when the keepalive ping is sent */
  grpc_closure start_keepalive_ping_locked;
  /** Cousure to run when the keepalive ping ack is received */
  grpc_closure finish_keepalive_ping_locked;
  /** Closrue to run when the keepalive ping timeouts */
  grpc_closure keepalive_watchdog_fired_locked;
  /** timer to initiate ping events */
  grpc_timer keepalive_ping_timer;
  /** watchdog to kill the transport when waiting for the keepalive ping */
  grpc_timer keepalive_watchdog_timer;
  /** time duration in between pings */
  grpc_millis keepalive_time;
  /** grace period for a ping to complete before watchdog kicks in */
  grpc_millis keepalive_timeout;
  /** if keepalive pings are allowed when there's no outstanding streams */
  bool keepalive_permit_without_calls = false;
  /** If start_keepalive_ping_locked has been called */
  bool keepalive_ping_started = false;
  /** keep-alive state machine state */
  grpc_chttp2_keepalive_state keepalive_state;
  grpc_core::ContextList* cl = nullptr;
  grpc_core::RefCountedPtr<grpc_core::channelz::SocketNode> channelz_socket;
  uint32_t num_messages_in_next_write = 0;
  /** The number of pending induced frames (SETTINGS_ACK, PINGS_ACK and
   * RST_STREAM) in the outgoing buffer (t->qbuf). If this number goes beyond
   * DEFAULT_MAX_PENDING_INDUCED_FRAMES, we pause reading new frames. We would
   * only continue reading when we are able to write to the socket again,
   * thereby reducing the number of induced frames. */
  uint32_t num_pending_induced_frames = 0;
  bool reading_paused_on_pending_induced_frames = false;
};

typedef enum {
  GRPC_METADATA_NOT_PUBLISHED,
  GRPC_METADATA_SYNTHESIZED_FROM_FAKE,
  GRPC_METADATA_PUBLISHED_FROM_WIRE,
  GRPC_METADATA_PUBLISHED_AT_CLOSE
} grpc_published_metadata_method;

struct grpc_chttp2_stream {
  grpc_chttp2_stream(grpc_chttp2_transport* t, grpc_stream_refcount* refcount,
                     const void* server_data, grpc_core::Arena* arena);
  ~grpc_chttp2_stream();

  void* context;
  grpc_chttp2_transport* t;
  grpc_stream_refcount* refcount;
  // Reffer is a 0-len structure, simply reffing `t` and `refcount` in its ctor
  // before initializing the rest of the stream, to avoid cache misses. This
  // field MUST be right after `t` and `refcount`.
  struct Reffer {
    explicit Reffer(grpc_chttp2_stream* s);
  } reffer;

  grpc_closure destroy_stream;
  grpc_closure* destroy_stream_arg;

  grpc_chttp2_stream_link links[STREAM_LIST_COUNT];
  uint8_t included[STREAM_LIST_COUNT] = {};

  /** HTTP2 stream id for this stream, or zero if one has not been assigned */
  uint32_t id = 0;

  /** things the upper layers would like to send */
  grpc_metadata_batch* send_initial_metadata = nullptr;
  grpc_closure* send_initial_metadata_finished = nullptr;
  grpc_metadata_batch* send_trailing_metadata = nullptr;
  grpc_closure* send_trailing_metadata_finished = nullptr;

  grpc_core::OrphanablePtr<grpc_core::ByteStream> fetching_send_message;
  uint32_t fetched_send_message_length = 0;
  grpc_slice fetching_slice = grpc_empty_slice();
  int64_t next_message_end_offset;
  int64_t flow_controlled_bytes_written = 0;
  int64_t flow_controlled_bytes_flowed = 0;
  grpc_closure complete_fetch_locked;
  grpc_closure* fetching_send_message_finished = nullptr;

  grpc_metadata_batch* recv_initial_metadata;
  grpc_closure* recv_initial_metadata_ready = nullptr;
  bool* trailing_metadata_available = nullptr;
  grpc_core::OrphanablePtr<grpc_core::ByteStream>* recv_message;
  grpc_closure* recv_message_ready = nullptr;
  grpc_metadata_batch* recv_trailing_metadata;
  grpc_closure* recv_trailing_metadata_finished = nullptr;

  grpc_transport_stream_stats* collecting_stats = nullptr;
  grpc_transport_stream_stats stats = grpc_transport_stream_stats();

  /** Is this stream closed for writing. */
  bool write_closed = false;
  /** Is this stream reading half-closed. */
  bool read_closed = false;
  /** Are all published incoming byte streams closed. */
  bool all_incoming_byte_streams_finished = false;
  /** Has this stream seen an error.
      If true, then pending incoming frames can be thrown away. */
  bool seen_error = false;
  /** Are we buffering writes on this stream? If yes, we won't become writable
      until there's enough queued up in the flow_controlled_buffer */
  bool write_buffering = false;

  /* have we sent or received the EOS bit? */
  bool eos_received = false;
  bool eos_sent = false;

  /** the error that resulted in this stream being read-closed */
  grpc_error* read_closed_error = GRPC_ERROR_NONE;
  /** the error that resulted in this stream being write-closed */
  grpc_error* write_closed_error = GRPC_ERROR_NONE;

  grpc_published_metadata_method published_metadata[2] = {};
  bool final_metadata_requested = false;

  grpc_chttp2_incoming_metadata_buffer metadata_buffer[2];

  grpc_slice_buffer frame_storage; /* protected by t combiner */

  grpc_closure* on_next = nullptr;  /* protected by t combiner */
  bool pending_byte_stream = false; /* protected by t combiner */
  // cached length of buffer to be used by the transport thread in cases where
  // stream->pending_byte_stream == true. The value is saved before
  // application threads are allowed to modify
  // unprocessed_incoming_frames_buffer
  size_t unprocessed_incoming_frames_buffer_cached_length = 0;
  /* Accessed only by transport thread when stream->pending_byte_stream == false
   * Accessed only by application thread when stream->pending_byte_stream ==
   * true */
  grpc_slice_buffer unprocessed_incoming_frames_buffer;
  grpc_closure reset_byte_stream;
  grpc_error* byte_stream_error = GRPC_ERROR_NONE; /* protected by t combiner */
  bool received_last_frame = false;                /* protected by t combiner */

  grpc_millis deadline = GRPC_MILLIS_INF_FUTURE;

  /** saw some stream level error */
  grpc_error* forced_close_error = GRPC_ERROR_NONE;
  /** how many header frames have we received? */
  uint8_t header_frames_received = 0;
  /** parsing state for data frames */
  /* Accessed only by transport thread when stream->pending_byte_stream == false
   * Accessed only by application thread when stream->pending_byte_stream ==
   * true */
  grpc_chttp2_data_parser data_parser;
  /** number of bytes received - reset at end of parse thread execution */
  int64_t received_bytes = 0;

  bool sent_initial_metadata = false;
  bool sent_trailing_metadata = false;

  grpc_core::PolymorphicManualConstructor<
      grpc_core::chttp2::StreamFlowControlBase,
      grpc_core::chttp2::StreamFlowControl,
      grpc_core::chttp2::StreamFlowControlDisabled>
      flow_control;

  grpc_slice_buffer flow_controlled_buffer;

  grpc_chttp2_write_cb* on_flow_controlled_cbs = nullptr;
  grpc_chttp2_write_cb* on_write_finished_cbs = nullptr;
  grpc_chttp2_write_cb* finish_after_write = nullptr;
  size_t sending_bytes = 0;

  /* Stream compression method to be used. */
  grpc_stream_compression_method stream_compression_method =
      GRPC_STREAM_COMPRESSION_IDENTITY_COMPRESS;
  /* Stream decompression method to be used. */
  grpc_stream_compression_method stream_decompression_method =
      GRPC_STREAM_COMPRESSION_IDENTITY_DECOMPRESS;

  /** Whether bytes stored in unprocessed_incoming_byte_stream is decompressed
   */
  bool unprocessed_incoming_frames_decompressed = false;
  /** Whether the bytes needs to be traced using Fathom */
  bool traced = false;
  /** gRPC header bytes that are already decompressed */
  size_t decompressed_header_bytes = 0;
  /** Byte counter for number of bytes written */
  size_t byte_counter = 0;

  /** Amount of uncompressed bytes sent out when compressed_data_buffer is
   * emptied */
  size_t uncompressed_data_size;
  /** Stream compression compress context */
  grpc_stream_compression_context* stream_compression_ctx;
  /** Buffer storing data that is compressed but not sent */
  grpc_slice_buffer compressed_data_buffer;

  /** Stream compression decompress context */
  grpc_stream_compression_context* stream_decompression_ctx;
  /** Temporary buffer storing decompressed data.
   * Initialized, used, and destroyed only when stream uses (non-identity)
   * compression.
   */
  grpc_slice_buffer decompressed_data_buffer;
};

/** Transport writing call flow:
    grpc_chttp2_initiate_write() is called anywhere that we know bytes need to
    go out on the wire.
    If no other write has been started, a task is enqueued onto our workqueue.
    When that task executes, it obtains the global lock, and gathers the data
    to write.
    The global lock is dropped and we do the syscall to write.
    After writing, a follow-up check is made to see if another round of writing
    should be performed.

    The actual call chain is documented in the implementation of this function.
    */
void grpc_chttp2_initiate_write(grpc_chttp2_transport* t,
                                grpc_chttp2_initiate_write_reason reason);

typedef struct {
  /** are we writing? */
  bool writing;
  /** if writing: was it a complete flush (false) or a partial flush (true) */
  bool partial;
  /** did we queue any completions as part of beginning the write */
  bool early_results_scheduled;
} grpc_chttp2_begin_write_result;

grpc_chttp2_begin_write_result grpc_chttp2_begin_write(
    grpc_chttp2_transport* t);
void grpc_chttp2_end_write(grpc_chttp2_transport* t, grpc_error* error);

/** Process one slice of incoming data; return 1 if the connection is still
    viable after reading, or 0 if the connection should be torn down */
grpc_error* grpc_chttp2_perform_read(grpc_chttp2_transport* t,
                                     const grpc_slice& slice);

bool grpc_chttp2_list_add_writable_stream(grpc_chttp2_transport* t,
                                          grpc_chttp2_stream* s);
/** Get a writable stream
    returns non-zero if there was a stream available */
bool grpc_chttp2_list_pop_writable_stream(grpc_chttp2_transport* t,
                                          grpc_chttp2_stream** s);
bool grpc_chttp2_list_remove_writable_stream(grpc_chttp2_transport* t,
                                             grpc_chttp2_stream* s);

bool grpc_chttp2_list_add_writing_stream(grpc_chttp2_transport* t,
                                         grpc_chttp2_stream* s);
bool grpc_chttp2_list_have_writing_streams(grpc_chttp2_transport* t);
bool grpc_chttp2_list_pop_writing_stream(grpc_chttp2_transport* t,
                                         grpc_chttp2_stream** s);

void grpc_chttp2_list_add_written_stream(grpc_chttp2_transport* t,
                                         grpc_chttp2_stream* s);
bool grpc_chttp2_list_pop_written_stream(grpc_chttp2_transport* t,
                                         grpc_chttp2_stream** s);

void grpc_chttp2_list_add_waiting_for_concurrency(grpc_chttp2_transport* t,
                                                  grpc_chttp2_stream* s);
bool grpc_chttp2_list_pop_waiting_for_concurrency(grpc_chttp2_transport* t,
                                                  grpc_chttp2_stream** s);
void grpc_chttp2_list_remove_waiting_for_concurrency(grpc_chttp2_transport* t,
                                                     grpc_chttp2_stream* s);

void grpc_chttp2_list_add_stalled_by_transport(grpc_chttp2_transport* t,
                                               grpc_chttp2_stream* s);
bool grpc_chttp2_list_pop_stalled_by_transport(grpc_chttp2_transport* t,
                                               grpc_chttp2_stream** s);
void grpc_chttp2_list_remove_stalled_by_transport(grpc_chttp2_transport* t,
                                                  grpc_chttp2_stream* s);

void grpc_chttp2_list_add_stalled_by_stream(grpc_chttp2_transport* t,
                                            grpc_chttp2_stream* s);
bool grpc_chttp2_list_pop_stalled_by_stream(grpc_chttp2_transport* t,
                                            grpc_chttp2_stream** s);
bool grpc_chttp2_list_remove_stalled_by_stream(grpc_chttp2_transport* t,
                                               grpc_chttp2_stream* s);

/********* Flow Control ***************/

// Takes in a flow control action and performs all the needed operations.
void grpc_chttp2_act_on_flowctl_action(
    const grpc_core::chttp2::FlowControlAction& action,
    grpc_chttp2_transport* t, grpc_chttp2_stream* s);

/********* End of Flow Control ***************/

inline grpc_chttp2_stream* grpc_chttp2_parsing_lookup_stream(
    grpc_chttp2_transport* t, uint32_t id) {
  return static_cast<grpc_chttp2_stream*>(
      grpc_chttp2_stream_map_find(&t->stream_map, id));
}
grpc_chttp2_stream* grpc_chttp2_parsing_accept_stream(grpc_chttp2_transport* t,
                                                      uint32_t id);

void grpc_chttp2_add_incoming_goaway(grpc_chttp2_transport* t,
                                     uint32_t goaway_error,
                                     uint32_t last_stream_id,
                                     const grpc_slice& goaway_text);

void grpc_chttp2_parsing_become_skip_parser(grpc_chttp2_transport* t);

void grpc_chttp2_complete_closure_step(grpc_chttp2_transport* t,
                                       grpc_chttp2_stream* s,
                                       grpc_closure** pclosure,
                                       grpc_error* error, const char* desc);

#define GRPC_HEADER_SIZE_IN_BYTES 5
#define MAX_SIZE_T (~(size_t)0)

#define GRPC_CHTTP2_CLIENT_CONNECT_STRING "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
#define GRPC_CHTTP2_CLIENT_CONNECT_STRLEN \
  (sizeof(GRPC_CHTTP2_CLIENT_CONNECT_STRING) - 1)

// extern grpc_core::TraceFlag grpc_http_trace;
// extern grpc_core::TraceFlag grpc_flowctl_trace;

#define GRPC_CHTTP2_IF_TRACING(stmt)                \
  do {                                              \
    if (GRPC_TRACE_FLAG_ENABLED(grpc_http_trace)) { \
      (stmt);                                       \
    }                                               \
  } while (0)

void grpc_chttp2_fake_status(grpc_chttp2_transport* t,
                             grpc_chttp2_stream* stream, grpc_error* error);
void grpc_chttp2_mark_stream_closed(grpc_chttp2_transport* t,
                                    grpc_chttp2_stream* s, int close_reads,
                                    int close_writes, grpc_error* error);
void grpc_chttp2_start_writing(grpc_chttp2_transport* t);

#ifndef NDEBUG
#define GRPC_CHTTP2_STREAM_REF(stream, reason) \
  grpc_chttp2_stream_ref(stream, reason)
#define GRPC_CHTTP2_STREAM_UNREF(stream, reason) \
  grpc_chttp2_stream_unref(stream, reason)
void grpc_chttp2_stream_ref(grpc_chttp2_stream* s, const char* reason);
void grpc_chttp2_stream_unref(grpc_chttp2_stream* s, const char* reason);
#else
#define GRPC_CHTTP2_STREAM_REF(stream, reason) grpc_chttp2_stream_ref(stream)
#define GRPC_CHTTP2_STREAM_UNREF(stream, reason) \
  grpc_chttp2_stream_unref(stream)
void grpc_chttp2_stream_ref(grpc_chttp2_stream* s);
void grpc_chttp2_stream_unref(grpc_chttp2_stream* s);
#endif

#ifndef NDEBUG
#define GRPC_CHTTP2_REF_TRANSPORT(t, r) \
  grpc_chttp2_ref_transport(t, r, __FILE__, __LINE__)
#define GRPC_CHTTP2_UNREF_TRANSPORT(t, r) \
  grpc_chttp2_unref_transport(t, r, __FILE__, __LINE__)
inline void grpc_chttp2_unref_transport(grpc_chttp2_transport* t,
                                        const char* reason, const char* file,
                                        int line) {
  if (t->refs.Unref(grpc_core::DebugLocation(file, line), reason)) {
    delete t;
  }
}
inline void grpc_chttp2_ref_transport(grpc_chttp2_transport* t,
                                      const char* reason, const char* file,
                                      int line) {
  t->refs.Ref(grpc_core::DebugLocation(file, line), reason);
}
#else
#define GRPC_CHTTP2_REF_TRANSPORT(t, r) grpc_chttp2_ref_transport(t)
#define GRPC_CHTTP2_UNREF_TRANSPORT(t, r) grpc_chttp2_unref_transport(t)
inline void grpc_chttp2_unref_transport(grpc_chttp2_transport* t) {
  if (t->refs.Unref()) {
    delete t;
  }
}
inline void grpc_chttp2_ref_transport(grpc_chttp2_transport* t) {
  t->refs.Ref();
}
#endif

void grpc_chttp2_ack_ping(grpc_chttp2_transport* t, uint64_t id);

/** Add a new ping strike to ping_recv_state.ping_strikes. If
    ping_recv_state.ping_strikes > ping_policy.max_ping_strikes, it sends GOAWAY
    with error code ENHANCE_YOUR_CALM and additional debug data resembling
    "too_many_pings" followed by immediately closing the connection. */
void grpc_chttp2_add_ping_strike(grpc_chttp2_transport* t);

/** add a ref to the stream and add it to the writable list;
    ref will be dropped in writing.c */
void grpc_chttp2_mark_stream_writable(grpc_chttp2_transport* t,
                                      grpc_chttp2_stream* s);

void grpc_chttp2_cancel_stream(grpc_chttp2_transport* t, grpc_chttp2_stream* s,
                               grpc_error* due_to_error);

void grpc_chttp2_maybe_complete_recv_initial_metadata(grpc_chttp2_transport* t,
                                                      grpc_chttp2_stream* s);
void grpc_chttp2_maybe_complete_recv_message(grpc_chttp2_transport* t,
                                             grpc_chttp2_stream* s);
void grpc_chttp2_maybe_complete_recv_trailing_metadata(grpc_chttp2_transport* t,
                                                       grpc_chttp2_stream* s);

void grpc_chttp2_fail_pending_writes(grpc_chttp2_transport* t,
                                     grpc_chttp2_stream* s, grpc_error* error);

/** Set the default keepalive configurations, must only be called at
    initialization */
void grpc_chttp2_config_default_keepalive_args(grpc_channel_args* args,
                                               bool is_client);

void grpc_chttp2_retry_initiate_ping(void* tp, grpc_error* error);

#endif /* GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_INTERNAL_H */
