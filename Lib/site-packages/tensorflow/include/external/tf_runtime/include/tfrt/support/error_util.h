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

// This file defines utilities related to error handling.

#ifndef TFRT_SUPPORT_ERROR_UTIL_H_
#define TFRT_SUPPORT_ERROR_UTIL_H_

#include <memory>
#include <tuple>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/string_util.h"

// Concatenate 'left' and 'right'.
#define TFRT_CONCAT(left, right) TFRT_CONCAT_IMPL(left, right)
#define TFRT_CONCAT_IMPL(left, right) left##right

// Helper macro to get value from llvm::Expected.
//
// The result of 'expr' should be a llvm::Expected<T>. If it has a value, it
// is assigned to 'lhs'. Otherwise the error is returned.
//
// Usage: TFRT_ASSIGN_OR_RETURN(auto value, GetExpectedValue());
#define TFRT_ASSIGN_OR_RETURN(lhs, expr) \
  TFRT_ASSIGN_OR_RETURN_IMPL(TFRT_CONCAT(_expected_, __COUNTER__), lhs, expr)
#define TFRT_ASSIGN_OR_RETURN_IMPL(expected, lhs, expr) \
  auto expected = expr;                                 \
  if (!expected) return expected.takeError();           \
  lhs = std::move(*expected)

namespace tfrt {

enum class ErrorCode {
#define ERROR_TYPE(ENUM) k##ENUM,
#include "tfrt/support/error_type.def"
};

namespace internal {
void UseCharPointer(char const volatile*);
// Pimpl class holding a stack trace, see CreateStackTrace() below.
struct StackTraceImpl;
struct StackTraceDeleter {
  void operator()(StackTraceImpl* ptr) const;
};
// Print a previously captured stack trace to 'os'. Does not print anything
// if 'stack_trace' is a nullptr. Found through template ADL.
llvm::raw_ostream& operator<<(
    llvm::raw_ostream& os,
    const std::unique_ptr<StackTraceImpl, StackTraceDeleter>& stack_trace);
}  // namespace internal

// Holds a stack trace that can be written to a llvm::raw_ostream.
using StackTrace =
    std::unique_ptr<internal::StackTraceImpl, internal::StackTraceDeleter>;

// Fake use on the object to prevent the compiler from doing a tail-call, which
// messes up the expected stack sizes of the CreateStackTrace() function.
template <class T>
void DoNotOptimize(const T& var) {
#if defined(_MSC_VER)
  internal::UseCharPointer(reinterpret_cast<char const volatile*>(&var));
  _ReadWriteBarrier();
#else
  asm volatile("" : "+m"(const_cast<T&>(var)));
#endif
}

// Capture the current stack trace, without the first 'skip_count' frames. The
// result may be empty (i.e. does not print anything) if capturing traces is
// not supported.
StackTrace CreateStackTrace(int skip_count = 0);

namespace internal {
// TMP to prevent elements of temporary reference types (i.e. llvm::ArrayRef,
// llvm::StringRef) because the underlying data is likely to go away before
// the Error is printed.
template <typename T>
struct IsTempRef : public std::integral_constant<bool, false> {};
template <typename T>
struct IsTempRef<llvm::ArrayRef<T>>
    : public std::integral_constant<bool, true> {};
template <>
struct IsTempRef<llvm::StringRef> : public std::integral_constant<bool, true> {
};
}  // namespace internal

// ErrorInfo with a pack of elements that are logged to llvm::raw_ostream.
template <typename... Args>
class TupleErrorInfo : public llvm::ErrorInfo<TupleErrorInfo<Args...>> {
  using Tuple = decltype(std::make_tuple(std::declval<Args>()...));

  template <bool...>
  struct BoolPack;
  template <bool... Bs>
  using AllFalse = std::is_same<BoolPack<Bs..., false>, BoolPack<false, Bs...>>;
  static_assert(
      AllFalse<internal::IsTempRef<std::decay_t<Args>>::value...>::value,
      "Argument types should not be temporary references.");

 public:
  // Required field for all ErrorInfo derivatives.
  // NOLINTNEXTLINE(readability-identifier-naming)
  static char ID;

  explicit TupleErrorInfo(Args... args) : tuple_(std::forward<Args>(args)...) {}

  template <typename T>
  constexpr const T& get() const {
    return std::get<T>(tuple_);
  }
  template <std::size_t I>
  constexpr const auto& get() const {
    return std::get<I>(tuple_);
  }

  void log(llvm::raw_ostream& os) const override {
    log(os, std::make_index_sequence<sizeof...(Args)>());
  }

  std::string message() const override {
    std::string message;
    llvm::raw_string_ostream os(message);
    log(os);
    return os.str();
  }

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

 private:
  template <std::size_t... Is>
  void log(llvm::raw_ostream& os, std::index_sequence<Is...>) const {
    internal::ToStreamHelper(os, get<Is>()...);
  }

  Tuple tuple_;
};
template <typename... Args>
char TupleErrorInfo<Args...>::ID;

// Create error from args, which are written to an llvm::raw_ostream when the
// error is logged.
template <typename... Args>
llvm::Error MakeTupleError(Args&&... args) {
  return llvm::make_error<TupleErrorInfo<Args...>>(std::forward<Args>(args)...);
}

// Create error from args by writing them to an llvm::raw_ostream immediately.
template <typename... Args>
llvm::Error MakeStringError(Args&&... args) {
  return MakeTupleError(StrCat(std::forward<Args>(args)...));
}

// Translate the code to string.
string_view ErrorName(ErrorCode code);

using ErrorType = string_view;

// Error that can be serialized and transferred through different hosts.
class BaseTypedErrorInfo : public llvm::ErrorInfo<BaseTypedErrorInfo> {
 public:
  // Required field for all ErrorInfo derivatives.
  static char ID;  // NOLINT

  void log(llvm::raw_ostream& os) const override {
    os << type_ << ": " << message_;
    if (!task_name_.empty()) {
      os << " (from task: " << task_name_ << ")";
    }
  }

  ErrorType type() const { return type_; }
  std::string message() const override { return message_; }
  string_view task_name() const { return task_name_; }

  std::error_code convertToErrorCode() const final {
    return llvm::inconvertibleErrorCode();
  }

 protected:
  BaseTypedErrorInfo(ErrorType type, string_view message, string_view task_name)
      : type_(type), message_(message), task_name_(task_name) {}

 private:
  const ErrorType type_;
  const std::string message_;
  const std::string task_name_;
};

template <typename ErrorTrait>
class TypedErrorInfo
    : public llvm::ErrorInfo<TypedErrorInfo<ErrorTrait>, BaseTypedErrorInfo> {
 public:
  static char ID;  // NOLINT
  using llvm::ErrorInfo<TypedErrorInfo<ErrorTrait>,
                        BaseTypedErrorInfo>::ErrorInfo;
  explicit TypedErrorInfo<ErrorTrait>(string_view message,
                                      string_view task_name = "")
      : TypedErrorInfo<ErrorTrait>(ErrorTrait::name, message, task_name) {}
};

#define ERROR_TYPE(ENUM)     \
  struct ENUM##ErrorTrait {  \
    static const char* name; \
  };                         \
  using ENUM##ErrorInfo = TypedErrorInfo<ENUM##ErrorTrait>;
#include "tfrt/support/error_type.def"  // NOLINT

template <typename ErrorTrait>
char TypedErrorInfo<ErrorTrait>::ID;

void LogIfError(Error&& error);
void DieIfError(Error&& error);

// Basic implementation of a collection of typed errors. Not thread-safe.
// TODO(haoyuzhang): implement more advanced error aggregation with lineage.
class ErrorCollection : public llvm::ErrorInfo<ErrorCollection> {
 public:
  // Required field for all ErrorInfo derivatives.
  static char ID;  // NOLINT

  std::error_code convertToErrorCode() const final {
    return llvm::inconvertibleErrorCode();
  }

  // Add an error to the collection. If the added error is an ErrorCollection,
  // merge the errors into the current collection.
  void AddError(Error error);

  // Get reference to all error info instances contained in this collection.
  const llvm::SmallVector<std::unique_ptr<BaseTypedErrorInfo>, 4>&
  GetAllErrors() const;

  // Log summarized error information for the collection of errors. Sample
  // summarized error message logging:
  //
  //   RpcDeadlineExceeded: Found 2 errors:
  //     (1) RpcDeadlineExceeded: Timeout registering function on remote host.
  //     (2) Cancelled: Remote execute is cancelled
  void log(raw_ostream& OS) const override;

 private:
  llvm::SmallVector<std::unique_ptr<BaseTypedErrorInfo>, 4> errors_;
};

}  // namespace tfrt

#endif  // TFRT_SUPPORT_ERROR_UTIL_H_
