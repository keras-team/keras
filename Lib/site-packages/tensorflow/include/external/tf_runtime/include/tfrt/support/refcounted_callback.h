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

// Defines ref-counted callback.

#ifndef TFRT_SUPPORT_REFCOUNTED_CALLBACK_H_
#define TFRT_SUPPORT_REFCOUNTED_CALLBACK_H_

#include <cstdint>

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Error.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
// Wrapper of a callback function that takes an llvm::Error argument. The
// callback function will be invoked with the internal Error status as argument
// when it gets destructed (i.e., loses the last reference).
//
// Example:
//   auto rc_done = MakeRef<RefCountedCallback>([...](Error e) mutable {
//       if (e) ...
//       ...
//     });
//   InvokeAsyncFunction(..., [rc_done = rc_done.CopyRef()](Error e) mutable {
//       rc_done->UpdateState(std::move(e));
//     });
class RefCountedCallback : public ReferenceCounted<RefCountedCallback> {
 public:
  explicit RefCountedCallback(llvm::unique_function<void(Error)> done)
      : done_(std::move(done)) {}

  ~RefCountedCallback() {
    if (errors_) {
      done_(Error(std::move(errors_)));
    } else {
      done_(Error::success());
    }
  }

  // Update the internal error status. If more than one errors are present, they
  // will be merged as an ErrorCollection in the callback.
  void UpdateState(Error e) {
    if (!e) return;
    mutex_lock l(mu_);
    if (errors_ == nullptr) {
      errors_ = std::make_unique<ErrorCollection>();
    }
    errors_->AddError(std::move(e));
  }

 private:
  llvm::unique_function<void(Error)> done_;
  mutex mu_;
  std::unique_ptr<ErrorCollection> errors_ TFRT_GUARDED_BY(mu_);
};
}  // namespace tfrt

#endif  // TFRT_SUPPORT_REFCOUNTED_CALLBACK_H_
