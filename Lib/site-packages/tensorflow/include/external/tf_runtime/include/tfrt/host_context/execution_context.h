/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file declares ExecutionContext.

#ifndef TFRT_HOST_CONTEXT_EXECUTION_CONTEXT_H_
#define TFRT_HOST_CONTEXT_EXECUTION_CONTEXT_H_

#include <utility>

#include "llvm/Support/Error.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/resource_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/map_by_type.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

class HostContext;
class ConcurrentWorkQueue;

class CancellationContext : public ReferenceCounted<CancellationContext> {
 public:
  CancellationContext() = default;
  CancellationContext(const CancellationContext&) = delete;
  CancellationContext& operator=(const CancellationContext&) = delete;

  ~CancellationContext();

  bool IsCancelled() const { return GetCancelAsyncValue(); }

  ErrorAsyncValue* GetCancelAsyncValue() const {
    return cancel_value_.load(std::memory_order_acquire);
  }

  void Cancel();

 private:
  std::atomic<ErrorAsyncValue*> cancel_value_{nullptr};
};

// A request refers to either a BEFFunction execution or an op execution.
// RequestContext holds per request information, such as the cancellation status
// and request priority. A RequestContext object is reference counted and is
// passed around during the execution of a request. This allows us to support
// per-request actions, such as canceling all pending ops for a request and
// assigning all tasks of a request to a particular priority.
//
// RequestContext can only be created by using RequestContextBuilder defined
// below.
class RequestContext : public ReferenceCounted<RequestContext> {
 public:
  using ContextData = MapByType<RequestContext>;

  bool IsCancelled() const { return cancellation_->IsCancelled(); }
  void Cancel();
  HostContext* host() const { return host_; }
  ResourceContext* resource_context() const { return resource_context_; }

  const RCReference<CancellationContext>& cancellation_context() const {
    return cancellation_;
  }

  // If the request has been canceled, return an ErrorAsyncValue for
  // the cancellation. Otherwise, return nullptr.
  ErrorAsyncValue* GetCancelAsyncValue() const {
    return cancellation_->GetCancelAsyncValue();
  }

  // Get context data by type. The returned reference T& is stable. The client
  // may store the reference/pointer if needed.
  template <typename T>
  T& GetData() {
    return context_data_.get<T>();
  }

  // Get context data by type. The returned reference T& is stable. The client
  // may store the reference/pointer if needed.
  template <typename T>
  T* GetDataIfExists() {
    return context_data_.getIfExists<T>();
  }

  int64_t id() const { return id_; }

 private:
  friend class RequestContextBuilder;

  RequestContext(HostContext* host, ResourceContext* resource_context,
                 ContextData ctx_data, int64_t id)
      : id_{id},
        host_{host},
        resource_context_{resource_context},
        context_data_{std::move(ctx_data)},
        cancellation_{TakeRef(new CancellationContext)} {}

  int64_t id_;
  HostContext* const host_ = nullptr;
  // Both ResourceContext and ContextData manages data used during the request
  // execution. ResourceContext is more flexible than ContextData at the cost of
  // performance. ResourceContext stores the data keyed by a string name. It
  // allows inserting data dynamically during the request execution and uses a
  // mutex to ensure thread-safety. In contrast, ContextData stores data keyed
  // by the data type and is populated only during the request initialization
  // time. The look up requires only a simple array index without
  // synchronization overhead.
  ResourceContext* const resource_context_ = nullptr;
  ContextData context_data_;

  RCReference<CancellationContext> cancellation_;
};

struct RequestOptions {
  using RequestPriority = int;

  RequestPriority priority = 0;
};

// A builder class for RequestContext.
// Sample usage:
// auto request_context = RequestContextBuilder(host, resource_context)
//                          .set_request_options(request_options)
//                          .build();
class RequestContextBuilder {
 public:
  RequestContextBuilder(HostContext* host, ResourceContext* resource_context,
                        int64_t id = 0, bool enable_cost_measurement = false)
      : id_{id},
        host_{host},
        resource_context_{resource_context},
        enable_cost_measurement_{enable_cost_measurement} {}

  RequestContextBuilder& set_request_options(RequestOptions request_options) & {
    request_options_ = std::move(request_options);
    return *this;
  }

  RequestContextBuilder&& set_request_options(
      RequestOptions request_options) && {
    request_options_ = std::move(request_options);
    return std::move(*this);
  }

  int64_t id() const { return id_; }
  HostContext* host() const { return host_; }
  ResourceContext* resource_context() const { return resource_context_; }
  const RequestOptions& request_options() const { return request_options_; }
  RequestContext::ContextData& context_data() { return context_data_; }

  // Build the RequestContext object.
  // This method is marked with &&, as it logically consumes this object. Once
  // the build() method is called, the RequestContextBuilder should no longer be
  // used.
  Expected<RCReference<RequestContext>> build() &&;

 private:
  int64_t id_;
  HostContext* host_;
  RequestOptions request_options_;
  ResourceContext* resource_context_ = nullptr;
  RequestContext::ContextData context_data_;
  bool enable_cost_measurement_ = false;
};

// ExecutionContext holds the context information for kernel and op execution,
// which currently includes the memory allocator, thread pool (memory allocator
// and thread pool are part of HostContext), and the location information. In
// the future, we plan to include other contextual information, such as client
// request id and request priority, and the request cancellation support, in the
// ExecutionContext as well.
//
// ExecutionContext is passed widely in the code base, as most code requires
// some of the facilities provided by ExecutionContext, e.g. memory allocation,
// dispatching async tasks, or reporting errors.

class ExecutionContext {
 public:
  explicit ExecutionContext(RCReference<RequestContext> req_ctx,
                            Location location = {});

  Location location() const { return location_; }
  HostContext* host() const { return request_ctx_->host(); }
  bool IsCancelled() const { return request_ctx_->IsCancelled(); }
  ErrorAsyncValue* GetCancelAsyncValue() const {
    return request_ctx_->GetCancelAsyncValue();
  }

  void set_location(Location location) { location_ = location; }

  // Set the work queue to use for dispatching async tasks.
  void set_work_queue(ConcurrentWorkQueue* work_queue) {
    assert(work_queue);
    work_queue_ = work_queue;
  }

  // Return the work queue to use for dispatching async tasks.
  ConcurrentWorkQueue& work_queue() const { return *work_queue_; }

  RequestContext* request_ctx() const { return request_ctx_.get(); }

  ResourceContext* resource_context() const {
    return request_ctx_->resource_context();
  }

 private:
  RCReference<RequestContext> request_ctx_;
  // If set, this work queue will be used for running async tasks in the
  // execution. Otherwise, the work queue in HostContext is used.
  ConcurrentWorkQueue* work_queue_ = nullptr;
  Location location_;
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_EXECUTION_CONTEXT_H_
