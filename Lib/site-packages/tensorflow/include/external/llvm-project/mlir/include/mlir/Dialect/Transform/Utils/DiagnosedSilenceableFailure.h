//===- DiagnosedSilenceableFailure.h - Tri-state result ----------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the DiagnosedSilenceableFailure class allowing to store
// a tri-state result (definite failure, recoverable failure, success) with an
// optional associated list of diagnostics.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include <optional>

#ifndef MLIR_DIALECT_TRANSFORM_UTILS_DIAGNOSEDSILENCEABLEFAILURE_H
#define MLIR_DIALECT_TRANSFORM_UTILS_DIAGNOSEDSILENCEABLEFAILURE_H

namespace mlir {
/// The result of a transform IR operation application. This can have one of the
/// three states:
///   - success;
///   - silenceable (recoverable) failure with yet-unreported diagnostic;
///   - definite failure.
/// Silenceable failure is intended to communicate information about
/// transformations that did not apply but in a way that supports recovery,
/// for example, they did not modify the payload IR or modified it in some
/// predictable way. They are associated with a Diagnostic that provides more
/// details on the failure. Silenceable failure can be discarded, turning the
/// result into success, or "reported", emitting the diagnostic and turning the
/// result into definite failure.
/// Transform IR operations containing other operations are allowed to do either
/// with the results of the nested transformations, but must propagate definite
/// failures as their diagnostics have been already reported to the user.
class [[nodiscard]] DiagnosedSilenceableFailure {
public:
  DiagnosedSilenceableFailure(const DiagnosedSilenceableFailure &) = delete;
  DiagnosedSilenceableFailure &
  operator=(const DiagnosedSilenceableFailure &) = delete;
  DiagnosedSilenceableFailure(DiagnosedSilenceableFailure &&) = default;
  DiagnosedSilenceableFailure &
  operator=(DiagnosedSilenceableFailure &&) = default;

  /// Constructs a DiagnosedSilenceableFailure in the success state.
  static DiagnosedSilenceableFailure success() {
    return DiagnosedSilenceableFailure(::mlir::success());
  }

  /// Constructs a DiagnosedSilenceableFailure in the failure state. Typically,
  /// a diagnostic has been emitted before this.
  static DiagnosedSilenceableFailure definiteFailure() {
    return DiagnosedSilenceableFailure(::mlir::failure());
  }

  /// Constructs a DiagnosedSilenceableFailure in the silenceable failure state,
  /// ready to emit the given diagnostic. This is considered a failure
  /// regardless of the diagnostic severity.
  static DiagnosedSilenceableFailure silenceableFailure(Diagnostic &&diag) {
    return DiagnosedSilenceableFailure(std::forward<Diagnostic>(diag));
  }
  static DiagnosedSilenceableFailure
  silenceableFailure(SmallVector<Diagnostic> &&diag) {
    return DiagnosedSilenceableFailure(
        std::forward<SmallVector<Diagnostic>>(diag));
  }

  /// Converts all kinds of failure into a LogicalResult failure, emitting the
  /// diagnostic if necessary. Must not be called more than once.
  LogicalResult checkAndReport();

  /// Returns `true` if this is a success.
  bool succeeded() const {
    return ::mlir::succeeded(result) && diagnostics.empty();
  }

  /// Returns `true` if this is a definite failure.
  bool isDefiniteFailure() const {
    return ::mlir::failed(result) && diagnostics.empty();
  }

  /// Returns `true` if this is a silenceable failure.
  bool isSilenceableFailure() const { return !diagnostics.empty(); }

  /// Returns the diagnostic message without emitting it. Expects this object
  /// to be a silenceable failure.
  std::string getMessage() const {
    std::string res;
    for (auto &diagnostic : diagnostics) {
      res.append(diagnostic.str());
      res.append("\n");
    }
    return res;
  }

  /// Returns a string representation of the failure mode (for error reporting).
  std::string getStatusString() const {
    if (succeeded())
      return "success";
    if (isSilenceableFailure())
      return "silenceable failure";
    return "definite failure";
  }

  /// Converts silenceable failure into LogicalResult success without reporting
  /// the diagnostic, preserves the other states.
  LogicalResult silence() {
    if (!diagnostics.empty()) {
      diagnostics.clear();
      result = ::mlir::success();
    }
    return result;
  }

  /// Take the diagnostics and silence.
  void takeDiagnostics(SmallVectorImpl<Diagnostic> &diags) {
    assert(!diagnostics.empty() && "expected a diagnostic to be present");
    diags.append(std::make_move_iterator(diagnostics.begin()),
                 std::make_move_iterator(diagnostics.end()));
  }

  /// Streams the given values into the last diagnostic.
  /// Expects this object to be a silenceable failure.
  template <typename T>
  DiagnosedSilenceableFailure &operator<<(T &&value) & {
    assert(isSilenceableFailure() &&
           "can only append output in silenceable failure state");
    diagnostics.back() << std::forward<T>(value);
    return *this;
  }
  template <typename T>
  DiagnosedSilenceableFailure &&operator<<(T &&value) && {
    return std::move(this->operator<<(std::forward<T>(value)));
  }

  /// Attaches a note to the last diagnostic.
  /// Expects this object to be a silenceable failure.
  Diagnostic &attachNote(std::optional<Location> loc = std::nullopt) {
    assert(isSilenceableFailure() &&
           "can only attach notes to silenceable failures");
    return diagnostics.back().attachNote(loc);
  }

private:
  explicit DiagnosedSilenceableFailure(LogicalResult result) : result(result) {}
  explicit DiagnosedSilenceableFailure(Diagnostic &&diagnostic)
      : result(failure()) {
    diagnostics.emplace_back(std::move(diagnostic));
  }
  explicit DiagnosedSilenceableFailure(SmallVector<Diagnostic> &&diagnostics)
      : diagnostics(std::move(diagnostics)), result(failure()) {}

  /// The diagnostics associated with this object. If non-empty, the object is
  /// considered to be in the silenceable failure state regardless of the
  /// `result` field.
  SmallVector<Diagnostic, 1> diagnostics;

  /// The "definite" logical state, either success or failure.
  /// Ignored if the diagnostics message is present.
  LogicalResult result;

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  /// Whether the associated diagnostics have been reported.
  /// Diagnostics reporting consumes the diagnostics, so we need a mechanism to
  /// differentiate reported diagnostics from a state where it was never
  /// created.
  bool reported = false;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
};

class DiagnosedDefiniteFailure;

DiagnosedDefiniteFailure emitDefiniteFailure(Location loc,
                                             const Twine &message = {});

/// A compatibility class connecting `InFlightDiagnostic` to
/// `DiagnosedSilenceableFailure` while providing an interface similar to the
/// former. Implicitly convertible to `DiagnosticSilenceableFailure` in definite
/// failure state and to `LogicalResult` failure. Reports the error on
/// conversion or on destruction. Instances of this class can be created by
/// `emitDefiniteFailure()`.
class DiagnosedDefiniteFailure {
  friend DiagnosedDefiniteFailure emitDefiniteFailure(Location loc,
                                                      const Twine &message);

public:
  /// Only move-constructible because it carries an in-flight diagnostic.
  DiagnosedDefiniteFailure(DiagnosedDefiniteFailure &&) = default;

  /// Forward the message to the diagnostic.
  template <typename T>
  DiagnosedDefiniteFailure &operator<<(T &&value) & {
    diag << std::forward<T>(value);
    return *this;
  }
  template <typename T>
  DiagnosedDefiniteFailure &&operator<<(T &&value) && {
    return std::move(this->operator<<(std::forward<T>(value)));
  }

  /// Attaches a note to the error.
  Diagnostic &attachNote(std::optional<Location> loc = std::nullopt) {
    return diag.attachNote(loc);
  }

  /// Implicit conversion to DiagnosedSilenceableFailure in the definite failure
  /// state. Reports the error.
  operator DiagnosedSilenceableFailure() {
    diag.report();
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  /// Implicit conversion to LogicalResult in the failure state. Reports the
  /// error.
  operator LogicalResult() {
    diag.report();
    return failure();
  }

private:
  /// Constructs a definite failure at the given location with the given
  /// message.
  explicit DiagnosedDefiniteFailure(Location loc, const Twine &message)
      : diag(emitError(loc, message)) {}

  /// Copy-construction and any assignment is disallowed to prevent repeated
  /// error reporting.
  DiagnosedDefiniteFailure(const DiagnosedDefiniteFailure &) = delete;
  DiagnosedDefiniteFailure &
  operator=(const DiagnosedDefiniteFailure &) = delete;
  DiagnosedDefiniteFailure &operator=(DiagnosedDefiniteFailure &&) = delete;

  /// The error message.
  InFlightDiagnostic diag;
};

/// Emits a definite failure with the given message. The returned object allows
/// for last-minute modification to the error message, such as attaching notes
/// and completing the message. It will be reported when the object is
/// destructed or converted.
inline DiagnosedDefiniteFailure emitDefiniteFailure(Location loc,
                                                    const Twine &message) {
  return DiagnosedDefiniteFailure(loc, message);
}
inline DiagnosedDefiniteFailure emitDefiniteFailure(Operation *op,
                                                    const Twine &message = {}) {
  return emitDefiniteFailure(op->getLoc(), message);
}

/// Emits a silenceable failure with the given message. A silenceable failure
/// must be either suppressed or converted into a definite failure and reported
/// to the user.
inline DiagnosedSilenceableFailure
emitSilenceableFailure(Location loc, const Twine &message = {}) {
  Diagnostic diag(loc, DiagnosticSeverity::Error);
  diag << message;
  return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
}
inline DiagnosedSilenceableFailure
emitSilenceableFailure(Operation *op, const Twine &message = {}) {
  return emitSilenceableFailure(op->getLoc(), message);
}
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_UTILS_DIAGNOSEDSILENCEABLEFAILURE_H
