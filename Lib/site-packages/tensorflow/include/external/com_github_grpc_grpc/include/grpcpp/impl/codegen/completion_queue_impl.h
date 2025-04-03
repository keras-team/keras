/*
 *
 * Copyright 2015-2016 gRPC authors.
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

/// A completion queue implements a concurrent producer-consumer queue, with
/// two main API-exposed methods: \a Next and \a AsyncNext. These
/// methods are the essential component of the gRPC C++ asynchronous API.
/// There is also a \a Shutdown method to indicate that a given completion queue
/// will no longer have regular events. This must be called before the
/// completion queue is destroyed.
/// All completion queue APIs are thread-safe and may be used concurrently with
/// any other completion queue API invocation; it is acceptable to have
/// multiple threads calling \a Next or \a AsyncNext on the same or different
/// completion queues, or to call these methods concurrently with a \a Shutdown
/// elsewhere.
/// \remark{All other API calls on completion queue should be completed before
/// a completion queue destructor is called.}
#ifndef GRPCPP_IMPL_CODEGEN_COMPLETION_QUEUE_IMPL_H
#define GRPCPP_IMPL_CODEGEN_COMPLETION_QUEUE_IMPL_H

#include <grpc/impl/codegen/atm.h>
#include <grpcpp/impl/codegen/completion_queue_tag.h>
#include <grpcpp/impl/codegen/core_codegen_interface.h>
#include <grpcpp/impl/codegen/grpc_library.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/time.h>

struct grpc_completion_queue;

namespace grpc_impl {

class Channel;
class Server;
class ServerBuilder;
template <class R>
class ClientReader;
template <class W>
class ClientWriter;
template <class W, class R>
class ClientReaderWriter;
template <class R>
class ServerReader;
template <class W>
class ServerWriter;
class ServerContextBase;
namespace internal {
template <class W, class R>
class ServerReaderWriterBody;

template <class ServiceType, class RequestType, class ResponseType>
class RpcMethodHandler;
template <class ServiceType, class RequestType, class ResponseType>
class ClientStreamingHandler;
template <class ServiceType, class RequestType, class ResponseType>
class ServerStreamingHandler;
template <class Streamer, bool WriteNeeded>
class TemplatedBidiStreamingHandler;
template <::grpc::StatusCode code>
class ErrorMethodHandler;
}  // namespace internal
}  // namespace grpc_impl
namespace grpc {

class ChannelInterface;
class ServerInterface;

namespace internal {
class CompletionQueueTag;
class RpcMethod;
template <class InputMessage, class OutputMessage>
class BlockingUnaryCallImpl;
template <class Op1, class Op2, class Op3, class Op4, class Op5, class Op6>
class CallOpSet;
}  // namespace internal

extern CoreCodegenInterface* g_core_codegen_interface;

}  // namespace grpc

namespace grpc_impl {

/// A thin wrapper around \ref grpc_completion_queue (see \ref
/// src/core/lib/surface/completion_queue.h).
/// See \ref doc/cpp/perf_notes.md for notes on best practices for high
/// performance servers.
class CompletionQueue : private ::grpc::GrpcLibraryCodegen {
 public:
  /// Default constructor. Implicitly creates a \a grpc_completion_queue
  /// instance.
  CompletionQueue()
      : CompletionQueue(grpc_completion_queue_attributes{
            GRPC_CQ_CURRENT_VERSION, GRPC_CQ_NEXT, GRPC_CQ_DEFAULT_POLLING,
            nullptr}) {}

  /// Wrap \a take, taking ownership of the instance.
  ///
  /// \param take The completion queue instance to wrap. Ownership is taken.
  explicit CompletionQueue(grpc_completion_queue* take);

  /// Destructor. Destroys the owned wrapped completion queue / instance.
  ~CompletionQueue() {
    ::grpc::g_core_codegen_interface->grpc_completion_queue_destroy(cq_);
  }

  /// Tri-state return for AsyncNext: SHUTDOWN, GOT_EVENT, TIMEOUT.
  enum NextStatus {
    SHUTDOWN,   ///< The completion queue has been shutdown and fully-drained
    GOT_EVENT,  ///< Got a new event; \a tag will be filled in with its
                ///< associated value; \a ok indicating its success.
    TIMEOUT     ///< deadline was reached.
  };

  /// Read from the queue, blocking until an event is available or the queue is
  /// shutting down.
  ///
  /// \param tag [out] Updated to point to the read event's tag.
  /// \param ok [out] true if read a successful event, false otherwise.
  ///
  /// Note that each tag sent to the completion queue (through RPC operations
  /// or alarms) will be delivered out of the completion queue by a call to
  /// Next (or a related method), regardless of whether the operation succeeded
  /// or not. Success here means that this operation completed in the normal
  /// valid manner.
  ///
  /// Server-side RPC request: \a ok indicates that the RPC has indeed
  /// been started. If it is false, the server has been Shutdown
  /// before this particular call got matched to an incoming RPC.
  ///
  /// Client-side StartCall/RPC invocation: \a ok indicates that the RPC is
  /// going to go to the wire. If it is false, it not going to the wire. This
  /// would happen if the channel is either permanently broken or
  /// transiently broken but with the fail-fast option. (Note that async unary
  /// RPCs don't post a CQ tag at this point, nor do client-streaming
  /// or bidi-streaming RPCs that have the initial metadata corked option set.)
  ///
  /// Client-side Write, Client-side WritesDone, Server-side Write,
  /// Server-side Finish, Server-side SendInitialMetadata (which is
  /// typically included in Write or Finish when not done explicitly):
  /// \a ok means that the data/metadata/status/etc is going to go to the
  /// wire. If it is false, it not going to the wire because the call
  /// is already dead (i.e., canceled, deadline expired, other side
  /// dropped the channel, etc).
  ///
  /// Client-side Read, Server-side Read, Client-side
  /// RecvInitialMetadata (which is typically included in Read if not
  /// done explicitly): \a ok indicates whether there is a valid message
  /// that got read. If not, you know that there are certainly no more
  /// messages that can ever be read from this stream. For the client-side
  /// operations, this only happens because the call is dead. For the
  /// server-sider operation, though, this could happen because the client
  /// has done a WritesDone already.
  ///
  /// Client-side Finish: \a ok should always be true
  ///
  /// Server-side AsyncNotifyWhenDone: \a ok should always be true
  ///
  /// Alarm: \a ok is true if it expired, false if it was canceled
  ///
  /// \return true if got an event, false if the queue is fully drained and
  ///         shut down.
  bool Next(void** tag, bool* ok) {
    return (AsyncNextInternal(tag, ok,
                              ::grpc::g_core_codegen_interface->gpr_inf_future(
                                  GPR_CLOCK_REALTIME)) != SHUTDOWN);
  }

  /// Read from the queue, blocking up to \a deadline (or the queue's shutdown).
  /// Both \a tag and \a ok are updated upon success (if an event is available
  /// within the \a deadline).  A \a tag points to an arbitrary location usually
  /// employed to uniquely identify an event.
  ///
  /// \param tag [out] Upon success, updated to point to the event's tag.
  /// \param ok [out] Upon success, true if a successful event, false otherwise
  ///        See documentation for CompletionQueue::Next for explanation of ok
  /// \param deadline [in] How long to block in wait for an event.
  ///
  /// \return The type of event read.
  template <typename T>
  NextStatus AsyncNext(void** tag, bool* ok, const T& deadline) {
    ::grpc::TimePoint<T> deadline_tp(deadline);
    return AsyncNextInternal(tag, ok, deadline_tp.raw_time());
  }

  /// EXPERIMENTAL
  /// First executes \a F, then reads from the queue, blocking up to
  /// \a deadline (or the queue's shutdown).
  /// Both \a tag and \a ok are updated upon success (if an event is available
  /// within the \a deadline).  A \a tag points to an arbitrary location usually
  /// employed to uniquely identify an event.
  ///
  /// \param f [in] Function to execute before calling AsyncNext on this queue.
  /// \param tag [out] Upon success, updated to point to the event's tag.
  /// \param ok [out] Upon success, true if read a regular event, false
  /// otherwise.
  /// \param deadline [in] How long to block in wait for an event.
  ///
  /// \return The type of event read.
  template <typename T, typename F>
  NextStatus DoThenAsyncNext(F&& f, void** tag, bool* ok, const T& deadline) {
    CompletionQueueTLSCache cache = CompletionQueueTLSCache(this);
    f();
    if (cache.Flush(tag, ok)) {
      return GOT_EVENT;
    } else {
      return AsyncNext(tag, ok, deadline);
    }
  }

  /// Request the shutdown of the queue.
  ///
  /// \warning This method must be called at some point if this completion queue
  /// is accessed with Next or AsyncNext. \a Next will not return false
  /// until this method has been called and all pending tags have been drained.
  /// (Likewise for \a AsyncNext returning \a NextStatus::SHUTDOWN .)
  /// Only once either one of these methods does that (that is, once the queue
  /// has been \em drained) can an instance of this class be destroyed.
  /// Also note that applications must ensure that no work is enqueued on this
  /// completion queue after this method is called.
  void Shutdown();

  /// Returns a \em raw pointer to the underlying \a grpc_completion_queue
  /// instance.
  ///
  /// \warning Remember that the returned instance is owned. No transfer of
  /// owership is performed.
  grpc_completion_queue* cq() { return cq_; }

 protected:
  /// Private constructor of CompletionQueue only visible to friend classes
  CompletionQueue(const grpc_completion_queue_attributes& attributes) {
    cq_ = ::grpc::g_core_codegen_interface->grpc_completion_queue_create(
        ::grpc::g_core_codegen_interface->grpc_completion_queue_factory_lookup(
            &attributes),
        &attributes, NULL);
    InitialAvalanching();  // reserve this for the future shutdown
  }

 private:
  // Friend synchronous wrappers so that they can access Pluck(), which is
  // a semi-private API geared towards the synchronous implementation.
  template <class R>
  friend class ::grpc_impl::ClientReader;
  template <class W>
  friend class ::grpc_impl::ClientWriter;
  template <class W, class R>
  friend class ::grpc_impl::ClientReaderWriter;
  template <class R>
  friend class ::grpc_impl::ServerReader;
  template <class W>
  friend class ::grpc_impl::ServerWriter;
  template <class W, class R>
  friend class ::grpc_impl::internal::ServerReaderWriterBody;
  template <class ServiceType, class RequestType, class ResponseType>
  friend class ::grpc_impl::internal::RpcMethodHandler;
  template <class ServiceType, class RequestType, class ResponseType>
  friend class ::grpc_impl::internal::ClientStreamingHandler;
  template <class ServiceType, class RequestType, class ResponseType>
  friend class ::grpc_impl::internal::ServerStreamingHandler;
  template <class Streamer, bool WriteNeeded>
  friend class ::grpc_impl::internal::TemplatedBidiStreamingHandler;
  template <::grpc::StatusCode code>
  friend class ::grpc_impl::internal::ErrorMethodHandler;
  friend class ::grpc_impl::Server;
  friend class ::grpc_impl::ServerContextBase;
  friend class ::grpc::ServerInterface;
  template <class InputMessage, class OutputMessage>
  friend class ::grpc::internal::BlockingUnaryCallImpl;

  // Friends that need access to constructor for callback CQ
  friend class ::grpc_impl::Channel;

  // For access to Register/CompleteAvalanching
  template <class Op1, class Op2, class Op3, class Op4, class Op5, class Op6>
  friend class ::grpc::internal::CallOpSet;

  /// EXPERIMENTAL
  /// Creates a Thread Local cache to store the first event
  /// On this completion queue queued from this thread.  Once
  /// initialized, it must be flushed on the same thread.
  class CompletionQueueTLSCache {
   public:
    CompletionQueueTLSCache(CompletionQueue* cq);
    ~CompletionQueueTLSCache();
    bool Flush(void** tag, bool* ok);

   private:
    CompletionQueue* cq_;
    bool flushed_;
  };

  NextStatus AsyncNextInternal(void** tag, bool* ok, gpr_timespec deadline);

  /// Wraps \a grpc_completion_queue_pluck.
  /// \warning Must not be mixed with calls to \a Next.
  bool Pluck(::grpc::internal::CompletionQueueTag* tag) {
    auto deadline =
        ::grpc::g_core_codegen_interface->gpr_inf_future(GPR_CLOCK_REALTIME);
    while (true) {
      auto ev = ::grpc::g_core_codegen_interface->grpc_completion_queue_pluck(
          cq_, tag, deadline, nullptr);
      bool ok = ev.success != 0;
      void* ignored = tag;
      if (tag->FinalizeResult(&ignored, &ok)) {
        GPR_CODEGEN_ASSERT(ignored == tag);
        return ok;
      }
    }
  }

  /// Performs a single polling pluck on \a tag.
  /// \warning Must not be mixed with calls to \a Next.
  ///
  /// TODO: sreek - This calls tag->FinalizeResult() even if the cq_ is already
  /// shutdown. This is most likely a bug and if it is a bug, then change this
  /// implementation to simple call the other TryPluck function with a zero
  /// timeout. i.e:
  ///      TryPluck(tag, gpr_time_0(GPR_CLOCK_REALTIME))
  void TryPluck(::grpc::internal::CompletionQueueTag* tag) {
    auto deadline =
        ::grpc::g_core_codegen_interface->gpr_time_0(GPR_CLOCK_REALTIME);
    auto ev = ::grpc::g_core_codegen_interface->grpc_completion_queue_pluck(
        cq_, tag, deadline, nullptr);
    if (ev.type == GRPC_QUEUE_TIMEOUT) return;
    bool ok = ev.success != 0;
    void* ignored = tag;
    // the tag must be swallowed if using TryPluck
    GPR_CODEGEN_ASSERT(!tag->FinalizeResult(&ignored, &ok));
  }

  /// Performs a single polling pluck on \a tag. Calls tag->FinalizeResult if
  /// the pluck() was successful and returned the tag.
  ///
  /// This exects tag->FinalizeResult (if called) to return 'false' i.e expects
  /// that the tag is internal not something that is returned to the user.
  void TryPluck(::grpc::internal::CompletionQueueTag* tag,
                gpr_timespec deadline) {
    auto ev = ::grpc::g_core_codegen_interface->grpc_completion_queue_pluck(
        cq_, tag, deadline, nullptr);
    if (ev.type == GRPC_QUEUE_TIMEOUT || ev.type == GRPC_QUEUE_SHUTDOWN) {
      return;
    }

    bool ok = ev.success != 0;
    void* ignored = tag;
    GPR_CODEGEN_ASSERT(!tag->FinalizeResult(&ignored, &ok));
  }

  /// Manage state of avalanching operations : completion queue tags that
  /// trigger other completion queue operations. The underlying core completion
  /// queue should not really shutdown until all avalanching operations have
  /// been finalized. Note that we maintain the requirement that an avalanche
  /// registration must take place before CQ shutdown (which must be maintained
  /// elsehwere)
  void InitialAvalanching() {
    gpr_atm_rel_store(&avalanches_in_flight_, static_cast<gpr_atm>(1));
  }
  void RegisterAvalanching() {
    gpr_atm_no_barrier_fetch_add(&avalanches_in_flight_,
                                 static_cast<gpr_atm>(1));
  }
  void CompleteAvalanching() {
    if (gpr_atm_no_barrier_fetch_add(&avalanches_in_flight_,
                                     static_cast<gpr_atm>(-1)) == 1) {
      ::grpc::g_core_codegen_interface->grpc_completion_queue_shutdown(cq_);
    }
  }

  grpc_completion_queue* cq_;  // owned

  gpr_atm avalanches_in_flight_;
};

/// A specific type of completion queue used by the processing of notifications
/// by servers. Instantiated by \a ServerBuilder.
class ServerCompletionQueue : public CompletionQueue {
 public:
  bool IsFrequentlyPolled() { return polling_type_ != GRPC_CQ_NON_LISTENING; }

 protected:
  /// Default constructor
  ServerCompletionQueue() : polling_type_(GRPC_CQ_DEFAULT_POLLING) {}

 private:
  /// \param completion_type indicates whether this is a NEXT or CALLBACK
  /// completion queue.
  /// \param polling_type Informs the GRPC library about the type of polling
  /// allowed on this completion queue. See grpc_cq_polling_type's description
  /// in grpc_types.h for more details.
  /// \param shutdown_cb is the shutdown callback used for CALLBACK api queues
  ServerCompletionQueue(grpc_cq_completion_type completion_type,
                        grpc_cq_polling_type polling_type,
                        grpc_experimental_completion_queue_functor* shutdown_cb)
      : CompletionQueue(grpc_completion_queue_attributes{
            GRPC_CQ_CURRENT_VERSION, completion_type, polling_type,
            shutdown_cb}),
        polling_type_(polling_type) {}

  grpc_cq_polling_type polling_type_;
  friend class ::grpc_impl::ServerBuilder;
  friend class ::grpc_impl::Server;
};

}  // namespace grpc_impl

#endif  // GRPCPP_IMPL_CODEGEN_COMPLETION_QUEUE_IMPL_H
