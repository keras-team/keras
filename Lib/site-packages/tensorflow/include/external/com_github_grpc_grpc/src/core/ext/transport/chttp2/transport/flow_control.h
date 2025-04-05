/*
 *
 * Copyright 2017 gRPC authors.
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

#ifndef GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_FLOW_CONTROL_H
#define GRPC_CORE_EXT_TRANSPORT_CHTTP2_TRANSPORT_FLOW_CONTROL_H

#include <grpc/support/port_platform.h>

#include <stdint.h>

#include "src/core/ext/transport/chttp2/transport/http2_settings.h"
#include "src/core/lib/gpr/useful.h"
#include "src/core/lib/gprpp/manual_constructor.h"
#include "src/core/lib/transport/bdp_estimator.h"
#include "src/core/lib/transport/pid_controller.h"

struct grpc_chttp2_transport;
struct grpc_chttp2_stream;

extern grpc_core::TraceFlag grpc_flowctl_trace;

namespace grpc {
namespace testing {
class TrickledCHTTP2;  // to make this a friend
}  // namespace testing
}  // namespace grpc

namespace grpc_core {
namespace chttp2 {

static constexpr uint32_t kDefaultWindow = 65535;
static constexpr int64_t kMaxWindow = static_cast<int64_t>((1u << 31) - 1);
// TODO(ncteisen): Tune this
static constexpr uint32_t kFrameSize = 1024 * 1024;

class TransportFlowControl;
class StreamFlowControl;

// Encapsulates a collections of actions the transport needs to take with
// regard to flow control. Each action comes with urgencies that tell the
// transport how quickly the action must take place.
class FlowControlAction {
 public:
  enum class Urgency : uint8_t {
    // Nothing to be done.
    NO_ACTION_NEEDED = 0,
    // Initiate a write to update the initial window immediately.
    UPDATE_IMMEDIATELY,
    // Push the flow control update into a send buffer, to be sent
    // out the next time a write is initiated.
    QUEUE_UPDATE,
  };

  Urgency send_stream_update() const { return send_stream_update_; }
  Urgency send_transport_update() const { return send_transport_update_; }
  Urgency send_initial_window_update() const {
    return send_initial_window_update_;
  }
  Urgency send_max_frame_size_update() const {
    return send_max_frame_size_update_;
  }
  uint32_t initial_window_size() const { return initial_window_size_; }
  uint32_t max_frame_size() const { return max_frame_size_; }

  FlowControlAction& set_send_stream_update(Urgency u) {
    send_stream_update_ = u;
    return *this;
  }
  FlowControlAction& set_send_transport_update(Urgency u) {
    send_transport_update_ = u;
    return *this;
  }
  FlowControlAction& set_send_initial_window_update(Urgency u,
                                                    uint32_t update) {
    send_initial_window_update_ = u;
    initial_window_size_ = update;
    return *this;
  }
  FlowControlAction& set_send_max_frame_size_update(Urgency u,
                                                    uint32_t update) {
    send_max_frame_size_update_ = u;
    max_frame_size_ = update;
    return *this;
  }

  static const char* UrgencyString(Urgency u);
  void Trace(grpc_chttp2_transport* t) const;

 private:
  Urgency send_stream_update_ = Urgency::NO_ACTION_NEEDED;
  Urgency send_transport_update_ = Urgency::NO_ACTION_NEEDED;
  Urgency send_initial_window_update_ = Urgency::NO_ACTION_NEEDED;
  Urgency send_max_frame_size_update_ = Urgency::NO_ACTION_NEEDED;
  uint32_t initial_window_size_ = 0;
  uint32_t max_frame_size_ = 0;
};

class FlowControlTrace {
 public:
  FlowControlTrace(const char* reason, TransportFlowControl* tfc,
                   StreamFlowControl* sfc) {
    if (enabled_) Init(reason, tfc, sfc);
  }

  ~FlowControlTrace() {
    if (enabled_) Finish();
  }

 private:
  void Init(const char* reason, TransportFlowControl* tfc,
            StreamFlowControl* sfc);
  void Finish();

  const bool enabled_ = GRPC_TRACE_FLAG_ENABLED(grpc_flowctl_trace);

  TransportFlowControl* tfc_;
  StreamFlowControl* sfc_;
  const char* reason_;
  int64_t remote_window_;
  int64_t target_window_;
  int64_t announced_window_;
  int64_t remote_window_delta_;
  int64_t local_window_delta_;
  int64_t announced_window_delta_;
};

// Fat interface with all methods a flow control implementation needs to
// support. gRPC C Core does not support pure virtual functions, so instead
// we abort in any methods which require implementation in the base class.
class TransportFlowControlBase {
 public:
  TransportFlowControlBase() {}
  virtual ~TransportFlowControlBase() {}

  // Is flow control enabled? This is needed in other codepaths like the checks
  // in parsing and in writing.
  virtual bool flow_control_enabled() const { abort(); }

  // Called to check if the transport needs to send a WINDOW_UPDATE frame
  virtual uint32_t MaybeSendUpdate(bool /* writing_anyway */) { abort(); }

  // Using the protected members, returns and Action to be taken by the
  // tranport.
  virtual FlowControlAction MakeAction() { abort(); }

  // Using the protected members, returns and Action to be taken by the
  // tranport. Also checks for updates to our BDP estimate and acts
  // accordingly.
  virtual FlowControlAction PeriodicUpdate() { abort(); }

  // Called to do bookkeeping when a stream owned by this transport sends
  // data on the wire
  virtual void StreamSentData(int64_t /* size */) { abort(); }

  // Called to do bookkeeping when a stream owned by this transport receives
  // data from the wire. Also does error checking for frame size.
  virtual grpc_error* RecvData(int64_t /* incoming_frame_size */) { abort(); }

  // Called to do bookkeeping when we receive a WINDOW_UPDATE frame.
  virtual void RecvUpdate(uint32_t /* size */) { abort(); }

  // Returns the BdpEstimator held by this object. Caller is responsible for
  // checking for nullptr. TODO(ncteisen): consider fully encapsulating all
  // bdp estimator actions inside TransportFlowControl
  virtual BdpEstimator* bdp_estimator() { return nullptr; }

  // Getters
  int64_t remote_window() const { return remote_window_; }
  virtual int64_t target_window() const { return target_initial_window_size_; }
  int64_t announced_window() const { return announced_window_; }

  // Used in certain benchmarks in which we don't want FlowControl to be a
  // factor
  virtual void TestOnlyForceHugeWindow() {}

 protected:
  friend class ::grpc::testing::TrickledCHTTP2;
  int64_t remote_window_ = kDefaultWindow;
  int64_t target_initial_window_size_ = kDefaultWindow;
  int64_t announced_window_ = kDefaultWindow;
};

// Implementation of flow control that does NOTHING. Always returns maximum
// values, never initiates writes, and assumes that the remote peer is doing
// the same. To be used to narrow down on flow control as the cause of negative
// performance.
class TransportFlowControlDisabled final : public TransportFlowControlBase {
 public:
  // Maxes out all values
  TransportFlowControlDisabled(grpc_chttp2_transport* t);

  bool flow_control_enabled() const override { return false; }

  // Never do anything.
  uint32_t MaybeSendUpdate(bool /* writing_anyway */) override { return 0; }
  FlowControlAction MakeAction() override { return FlowControlAction(); }
  FlowControlAction PeriodicUpdate() override { return FlowControlAction(); }
  void StreamSentData(int64_t /* size */) override {}
  grpc_error* RecvData(int64_t /* incoming_frame_size */) override {
    return GRPC_ERROR_NONE;
  }
  void RecvUpdate(uint32_t /* size */) override {}
};

// Implementation of flow control that abides to HTTP/2 spec and attempts
// to be as performant as possible.
class TransportFlowControl final : public TransportFlowControlBase {
 public:
  TransportFlowControl(const grpc_chttp2_transport* t, bool enable_bdp_probe);
  ~TransportFlowControl() {}

  bool flow_control_enabled() const override { return true; }

  bool bdp_probe() const { return enable_bdp_probe_; }

  // returns an announce if we should send a transport update to our peer,
  // else returns zero; writing_anyway indicates if a write would happen
  // regardless of the send - if it is false and this function returns non-zero,
  // this announce will cause a write to occur
  uint32_t MaybeSendUpdate(bool writing_anyway) override;

  // Reads the flow control data and returns and actionable struct that will
  // tell chttp2 exactly what it needs to do
  FlowControlAction MakeAction() override {
    return UpdateAction(FlowControlAction());
  }

  // Call periodically (at a low-ish rate, 100ms - 10s makes sense)
  // to perform more complex flow control calculations and return an action
  // to let chttp2 change its parameters
  FlowControlAction PeriodicUpdate() override;

  void StreamSentData(int64_t size) override { remote_window_ -= size; }

  grpc_error* ValidateRecvData(int64_t incoming_frame_size);
  void CommitRecvData(int64_t incoming_frame_size) {
    announced_window_ -= incoming_frame_size;
  }

  grpc_error* RecvData(int64_t incoming_frame_size) override {
    FlowControlTrace trace("  data recv", this, nullptr);
    grpc_error* error = ValidateRecvData(incoming_frame_size);
    if (error != GRPC_ERROR_NONE) return error;
    CommitRecvData(incoming_frame_size);
    return GRPC_ERROR_NONE;
  }

  // we have received a WINDOW_UPDATE frame for a transport
  void RecvUpdate(uint32_t size) override {
    FlowControlTrace trace("t updt recv", this, nullptr);
    remote_window_ += size;
  }

  // See comment above announced_stream_total_over_incoming_window_ for the
  // logic behind this decision.
  int64_t target_window() const override {
    return static_cast<uint32_t> GPR_MIN(
        (int64_t)((1u << 31) - 1),
        announced_stream_total_over_incoming_window_ +
            target_initial_window_size_);
  }

  const grpc_chttp2_transport* transport() const { return t_; }

  void PreUpdateAnnouncedWindowOverIncomingWindow(int64_t delta) {
    if (delta > 0) {
      announced_stream_total_over_incoming_window_ -= delta;
    }
  }

  void PostUpdateAnnouncedWindowOverIncomingWindow(int64_t delta) {
    if (delta > 0) {
      announced_stream_total_over_incoming_window_ += delta;
    }
  }

  BdpEstimator* bdp_estimator() override { return &bdp_estimator_; }

  void TestOnlyForceHugeWindow() override {
    announced_window_ = 1024 * 1024 * 1024;
    remote_window_ = 1024 * 1024 * 1024;
  }

 private:
  double TargetLogBdp();
  double SmoothLogBdp(double value);
  FlowControlAction::Urgency DeltaUrgency(int64_t value,
                                          grpc_chttp2_setting_id setting_id);

  FlowControlAction UpdateAction(FlowControlAction action) {
    if (announced_window_ < target_window() / 2) {
      action.set_send_transport_update(
          FlowControlAction::Urgency::UPDATE_IMMEDIATELY);
    }
    return action;
  }

  const grpc_chttp2_transport* const t_;

  /** calculating what we should give for local window:
      we track the total amount of flow control over initial window size
      across all streams: this is data that we want to receive right now (it
      has an outstanding read)
      and the total amount of flow control under initial window size across all
      streams: this is data we've read early
      we want to adjust incoming_window such that:
      incoming_window = total_over - max(bdp - total_under, 0) */
  int64_t announced_stream_total_over_incoming_window_ = 0;

  /** should we probe bdp? */
  const bool enable_bdp_probe_;

  /* bdp estimation */
  grpc_core::BdpEstimator bdp_estimator_;

  /* pid controller */
  grpc_core::PidController pid_controller_;
  grpc_millis last_pid_update_ = 0;
};

// Fat interface with all methods a stream flow control implementation needs
// to support. gRPC C Core does not support pure virtual functions, so instead
// we abort in any methods which require implementation in the base class.
class StreamFlowControlBase {
 public:
  StreamFlowControlBase() {}
  virtual ~StreamFlowControlBase() {}

  // Updates an action using the protected members.
  virtual FlowControlAction UpdateAction(FlowControlAction /* action */) {
    abort();
  }

  // Using the protected members, returns an Action for this stream to be
  // taken by the tranport.
  virtual FlowControlAction MakeAction() { abort(); }

  // Bookkeeping for when data is sent on this stream.
  virtual void SentData(int64_t /* outgoing_frame_size */) { abort(); }

  // Bookkeeping and error checking for when data is received by this stream.
  virtual grpc_error* RecvData(int64_t /* incoming_frame_size */) { abort(); }

  // Called to check if this stream needs to send a WINDOW_UPDATE frame.
  virtual uint32_t MaybeSendUpdate() { abort(); }

  // Bookkeeping for receiving a WINDOW_UPDATE from for this stream.
  virtual void RecvUpdate(uint32_t /* size */) { abort(); }

  // Bookkeeping for when a call pulls bytes out of the transport. At this
  // point we consider the data 'used' and can thus let out peer know we are
  // ready for more data.
  virtual void IncomingByteStreamUpdate(size_t /* max_size_hint */,
                                        size_t /* have_already */) {
    abort();
  }

  // Used in certain benchmarks in which we don't want FlowControl to be a
  // factor
  virtual void TestOnlyForceHugeWindow() {}

  // Getters
  int64_t remote_window_delta() { return remote_window_delta_; }
  int64_t local_window_delta() { return local_window_delta_; }
  int64_t announced_window_delta() { return announced_window_delta_; }

 protected:
  friend class ::grpc::testing::TrickledCHTTP2;
  int64_t remote_window_delta_ = 0;
  int64_t local_window_delta_ = 0;
  int64_t announced_window_delta_ = 0;
};

// Implementation of flow control that does NOTHING. Always returns maximum
// values, never initiates writes, and assumes that the remote peer is doing
// the same. To be used to narrow down on flow control as the cause of negative
// performance.
class StreamFlowControlDisabled : public StreamFlowControlBase {
 public:
  FlowControlAction UpdateAction(FlowControlAction action) override {
    return action;
  }
  FlowControlAction MakeAction() override { return FlowControlAction(); }
  void SentData(int64_t /* outgoing_frame_size */) override {}
  grpc_error* RecvData(int64_t /* incoming_frame_size */) override {
    return GRPC_ERROR_NONE;
  }
  uint32_t MaybeSendUpdate() override { return 0; }
  void RecvUpdate(uint32_t /* size */) override {}
  void IncomingByteStreamUpdate(size_t /* max_size_hint */,
                                size_t /* have_already */) override {}
};

// Implementation of flow control that abides to HTTP/2 spec and attempts
// to be as performant as possible.
class StreamFlowControl final : public StreamFlowControlBase {
 public:
  StreamFlowControl(TransportFlowControl* tfc, const grpc_chttp2_stream* s);
  ~StreamFlowControl() {
    tfc_->PreUpdateAnnouncedWindowOverIncomingWindow(announced_window_delta_);
  }

  FlowControlAction UpdateAction(FlowControlAction action) override;
  FlowControlAction MakeAction() override {
    return UpdateAction(tfc_->MakeAction());
  }

  // we have sent data on the wire, we must track this in our bookkeeping for
  // the remote peer's flow control.
  void SentData(int64_t outgoing_frame_size) override {
    FlowControlTrace tracer("  data sent", tfc_, this);
    tfc_->StreamSentData(outgoing_frame_size);
    remote_window_delta_ -= outgoing_frame_size;
  }

  // we have received data from the wire
  grpc_error* RecvData(int64_t incoming_frame_size) override;

  // returns an announce if we should send a stream update to our peer, else
  // returns zero
  uint32_t MaybeSendUpdate() override;

  // we have received a WINDOW_UPDATE frame for a stream
  void RecvUpdate(uint32_t size) override {
    FlowControlTrace trace("s updt recv", tfc_, this);
    remote_window_delta_ += size;
  }

  // the application is asking for a certain amount of bytes
  void IncomingByteStreamUpdate(size_t max_size_hint,
                                size_t have_already) override;

  int64_t remote_window_delta() const { return remote_window_delta_; }
  int64_t local_window_delta() const { return local_window_delta_; }
  int64_t announced_window_delta() const { return announced_window_delta_; }

  const grpc_chttp2_stream* stream() const { return s_; }

  void TestOnlyForceHugeWindow() override {
    announced_window_delta_ = 1024 * 1024 * 1024;
    local_window_delta_ = 1024 * 1024 * 1024;
    remote_window_delta_ = 1024 * 1024 * 1024;
  }

 private:
  TransportFlowControl* const tfc_;
  const grpc_chttp2_stream* const s_;

  void UpdateAnnouncedWindowDelta(TransportFlowControl* tfc, int64_t change) {
    tfc->PreUpdateAnnouncedWindowOverIncomingWindow(announced_window_delta_);
    announced_window_delta_ += change;
    tfc->PostUpdateAnnouncedWindowOverIncomingWindow(announced_window_delta_);
  }
};

}  // namespace chttp2
}  // namespace grpc_core

#endif
