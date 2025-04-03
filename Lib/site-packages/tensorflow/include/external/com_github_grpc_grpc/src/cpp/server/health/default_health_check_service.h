/*
 *
 * Copyright 2016 gRPC authors.
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

#ifndef GRPC_INTERNAL_CPP_SERVER_DEFAULT_HEALTH_CHECK_SERVICE_H
#define GRPC_INTERNAL_CPP_SERVER_DEFAULT_HEALTH_CHECK_SERVICE_H

#include <atomic>
#include <set>

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/impl/codegen/async_generic_service.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/completion_queue.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/support/byte_buffer.h>

#include "src/core/lib/gprpp/sync.h"
#include "src/core/lib/gprpp/thd.h"

namespace grpc {

// Default implementation of HealthCheckServiceInterface. Server will create and
// own it.
class DefaultHealthCheckService final : public HealthCheckServiceInterface {
 public:
  enum ServingStatus { NOT_FOUND, SERVING, NOT_SERVING };

  // The service impl to register with the server.
  class HealthCheckServiceImpl : public Service {
   public:
    // Base class for call handlers.
    class CallHandler {
     public:
      virtual ~CallHandler() = default;
      virtual void SendHealth(std::shared_ptr<CallHandler> self,
                              ServingStatus status) = 0;
    };

    HealthCheckServiceImpl(DefaultHealthCheckService* database,
                           std::unique_ptr<ServerCompletionQueue> cq);

    ~HealthCheckServiceImpl();

    void StartServingThread();

   private:
    // A tag that can be called with a bool argument. It's tailored for
    // CallHandler's use. Before being used, it should be constructed with a
    // method of CallHandler and a shared pointer to the handler. The
    // shared pointer will be moved to the invoked function and the function
    // can only be invoked once. That makes ref counting of the handler easier,
    // because the shared pointer is not bound to the function and can be gone
    // once the invoked function returns (if not used any more).
    class CallableTag {
     public:
      using HandlerFunction =
          std::function<void(std::shared_ptr<CallHandler>, bool)>;

      CallableTag() {}

      CallableTag(HandlerFunction func, std::shared_ptr<CallHandler> handler)
          : handler_function_(std::move(func)), handler_(std::move(handler)) {
        GPR_ASSERT(handler_function_ != nullptr);
        GPR_ASSERT(handler_ != nullptr);
      }

      // Runs the tag. This should be called only once. The handler is no
      // longer owned by this tag after this method is invoked.
      void Run(bool ok) {
        GPR_ASSERT(handler_function_ != nullptr);
        GPR_ASSERT(handler_ != nullptr);
        handler_function_(std::move(handler_), ok);
      }

      // Releases and returns the shared pointer to the handler.
      std::shared_ptr<CallHandler> ReleaseHandler() {
        return std::move(handler_);
      }

     private:
      HandlerFunction handler_function_ = nullptr;
      std::shared_ptr<CallHandler> handler_;
    };

    // Call handler for Check method.
    // Each handler takes care of one call. It contains per-call data and it
    // will access the members of the parent class (i.e.,
    // DefaultHealthCheckService) for per-service health data.
    class CheckCallHandler : public CallHandler {
     public:
      // Instantiates a CheckCallHandler and requests the next health check
      // call. The handler object will manage its own lifetime, so no action is
      // needed from the caller any more regarding that object.
      static void CreateAndStart(ServerCompletionQueue* cq,
                                 DefaultHealthCheckService* database,
                                 HealthCheckServiceImpl* service);

      // This ctor is public because we want to use std::make_shared<> in
      // CreateAndStart(). This ctor shouldn't be used elsewhere.
      CheckCallHandler(ServerCompletionQueue* cq,
                       DefaultHealthCheckService* database,
                       HealthCheckServiceImpl* service);

      // Not used for Check.
      void SendHealth(std::shared_ptr<CallHandler> /*self*/,
                      ServingStatus /*status*/) override {}

     private:
      // Called when we receive a call.
      // Spawns a new handler so that we can keep servicing future calls.
      void OnCallReceived(std::shared_ptr<CallHandler> self, bool ok);

      // Called when Finish() is done.
      void OnFinishDone(std::shared_ptr<CallHandler> self, bool ok);

      // The members passed down from HealthCheckServiceImpl.
      ServerCompletionQueue* cq_;
      DefaultHealthCheckService* database_;
      HealthCheckServiceImpl* service_;

      ByteBuffer request_;
      GenericServerAsyncResponseWriter writer_;
      ServerContext ctx_;

      CallableTag next_;
    };

    // Call handler for Watch method.
    // Each handler takes care of one call. It contains per-call data and it
    // will access the members of the parent class (i.e.,
    // DefaultHealthCheckService) for per-service health data.
    class WatchCallHandler : public CallHandler {
     public:
      // Instantiates a WatchCallHandler and requests the next health check
      // call. The handler object will manage its own lifetime, so no action is
      // needed from the caller any more regarding that object.
      static void CreateAndStart(ServerCompletionQueue* cq,
                                 DefaultHealthCheckService* database,
                                 HealthCheckServiceImpl* service);

      // This ctor is public because we want to use std::make_shared<> in
      // CreateAndStart(). This ctor shouldn't be used elsewhere.
      WatchCallHandler(ServerCompletionQueue* cq,
                       DefaultHealthCheckService* database,
                       HealthCheckServiceImpl* service);

      void SendHealth(std::shared_ptr<CallHandler> self,
                      ServingStatus status) override;

     private:
      // Called when we receive a call.
      // Spawns a new handler so that we can keep servicing future calls.
      void OnCallReceived(std::shared_ptr<CallHandler> self, bool ok);

      // Requires holding send_mu_.
      void SendHealthLocked(std::shared_ptr<CallHandler> self,
                            ServingStatus status);

      // When sending a health result finishes.
      void OnSendHealthDone(std::shared_ptr<CallHandler> self, bool ok);

      void SendFinish(std::shared_ptr<CallHandler> self, const Status& status);

      // Requires holding service_->cq_shutdown_mu_.
      void SendFinishLocked(std::shared_ptr<CallHandler> self,
                            const Status& status);

      // Called when Finish() is done.
      void OnFinishDone(std::shared_ptr<CallHandler> self, bool ok);

      // Called when AsyncNotifyWhenDone() notifies us.
      void OnDoneNotified(std::shared_ptr<CallHandler> self, bool ok);

      // The members passed down from HealthCheckServiceImpl.
      ServerCompletionQueue* cq_;
      DefaultHealthCheckService* database_;
      HealthCheckServiceImpl* service_;

      ByteBuffer request_;
      grpc::string service_name_;
      GenericServerAsyncWriter stream_;
      ServerContext ctx_;

      grpc_core::Mutex send_mu_;
      bool send_in_flight_ = false;               // Guarded by mu_.
      ServingStatus pending_status_ = NOT_FOUND;  // Guarded by mu_.

      bool finish_called_ = false;
      CallableTag next_;
      CallableTag on_done_notified_;
      CallableTag on_finish_done_;
    };

    // Handles the incoming requests and drives the completion queue in a loop.
    static void Serve(void* arg);

    // Returns true on success.
    static bool DecodeRequest(const ByteBuffer& request,
                              grpc::string* service_name);
    static bool EncodeResponse(ServingStatus status, ByteBuffer* response);

    // Needed to appease Windows compilers, which don't seem to allow
    // nested classes to access protected members in the parent's
    // superclass.
    using Service::RequestAsyncServerStreaming;
    using Service::RequestAsyncUnary;

    DefaultHealthCheckService* database_;
    std::unique_ptr<ServerCompletionQueue> cq_;

    // To synchronize the operations related to shutdown state of cq_, so that
    // we don't enqueue new tags into cq_ after it is already shut down.
    grpc_core::Mutex cq_shutdown_mu_;
    std::atomic_bool shutdown_{false};
    std::unique_ptr<::grpc_core::Thread> thread_;
  };

  DefaultHealthCheckService();

  void SetServingStatus(const grpc::string& service_name,
                        bool serving) override;
  void SetServingStatus(bool serving) override;

  void Shutdown() override;

  ServingStatus GetServingStatus(const grpc::string& service_name) const;

  HealthCheckServiceImpl* GetHealthCheckService(
      std::unique_ptr<ServerCompletionQueue> cq);

 private:
  // Stores the current serving status of a service and any call
  // handlers registered for updates when the service's status changes.
  class ServiceData {
   public:
    void SetServingStatus(ServingStatus status);
    ServingStatus GetServingStatus() const { return status_; }
    void AddCallHandler(
        std::shared_ptr<HealthCheckServiceImpl::CallHandler> handler);
    void RemoveCallHandler(
        const std::shared_ptr<HealthCheckServiceImpl::CallHandler>& handler);
    bool Unused() const {
      return call_handlers_.empty() && status_ == NOT_FOUND;
    }

   private:
    ServingStatus status_ = NOT_FOUND;
    std::set<std::shared_ptr<HealthCheckServiceImpl::CallHandler>>
        call_handlers_;
  };

  void RegisterCallHandler(
      const grpc::string& service_name,
      std::shared_ptr<HealthCheckServiceImpl::CallHandler> handler);

  void UnregisterCallHandler(
      const grpc::string& service_name,
      const std::shared_ptr<HealthCheckServiceImpl::CallHandler>& handler);

  mutable grpc_core::Mutex mu_;
  bool shutdown_ = false;                             // Guarded by mu_.
  std::map<grpc::string, ServiceData> services_map_;  // Guarded by mu_.
  std::unique_ptr<HealthCheckServiceImpl> impl_;
};

}  // namespace grpc

#endif  // GRPC_INTERNAL_CPP_SERVER_DEFAULT_HEALTH_CHECK_SERVICE_H
