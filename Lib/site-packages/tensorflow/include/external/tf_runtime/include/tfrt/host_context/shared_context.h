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

// Shared context abstraction
//
// This file declares SharedContext.

#ifndef TFRT_HOST_CONTEXT_SHARED_CONTEXT_H_
#define TFRT_HOST_CONTEXT_SHARED_CONTEXT_H_

namespace tfrt {

// SharedContext is used to store shared data that has the same life-time as
// a HostContext. A subclass of SharedContext is required to have a constructor
// that takes a HostContext*.
//
// Example usage:
//
// class SampleSharedContext : public SharedContext {
//  public:
//   // A SharedContext is required to have a constructor that takes a
//   // HostContext*.
//   explicit SampleSharedContext(HostContext* host) {}
//   // ...
// };
//
// In the user code, to retrieve the shared context object, use
// HostContext::GetOrCreateSharedContext<SampleSharedContext>() as follows,
// where host is of type HostContext*.
//
// SampleSharedContext& sample_shared_context =
//             host->GetOrCreateSharedContext<SampleSharedContext>();
//
class SharedContext {
 public:
  virtual ~SharedContext();
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_SHARED_CONTEXT_H_
