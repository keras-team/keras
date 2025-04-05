//===- Timing.h - Execution time measurement facilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Facilities to measure and provide statistics on execution time.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_TIMING_H
#define MLIR_SUPPORT_TIMING_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMapEntry.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace mlir {

class Timer;
class TimingManager;
class TimingScope;
class DefaultTimingManager;
namespace detail {
class TimingManagerImpl;
class DefaultTimingManagerImpl;
} // namespace detail

//===----------------------------------------------------------------------===//
// TimingIdentifier
//===----------------------------------------------------------------------===//

/// This class represesents a uniqued string owned by a `TimingManager`. Most
/// importantly, instances of this class provide a stable opaque pointer that
/// is guaranteed to be reproduced by later interning of the same string. The
/// `TimingManager` uses this mechanism to provide timers with an opaque id
/// even when the user of the API merely provided a string as identification
/// (instead of a pass for example).
///
/// This is a POD type with pointer size, so it should be passed around by
/// value. The underlying data is owned by the `TimingManager`.
class TimingIdentifier {
  using EntryType = llvm::StringMapEntry<std::nullopt_t>;

public:
  TimingIdentifier(const TimingIdentifier &) = default;
  TimingIdentifier &operator=(const TimingIdentifier &other) = default;

  /// Return an identifier for the specified string.
  static TimingIdentifier get(StringRef str, TimingManager &tm);

  /// Return a `StringRef` for the string.
  StringRef strref() const { return entry->first(); }

  /// Return an `std::string`.
  std::string str() const { return strref().str(); }

  /// Return the opaque pointer that corresponds to this identifier.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(entry);
  }

private:
  const EntryType *entry;
  explicit TimingIdentifier(const EntryType *entry) : entry(entry) {}
};

//===----------------------------------------------------------------------===//
// TimingManager
//===----------------------------------------------------------------------===//

/// This class represents facilities to measure execution time.
///
/// Libraries and infrastructure code operate on opque `Timer` handles returned
/// by various functions of this manager. Timers are started and stopped to
/// demarcate regions in the code where execution time is of interest, and they
/// can be nested to provide more detailed timing resolution. Calls to the timer
/// start, stop, and nesting functions must be balanced. To facilitate this,
/// users are encouraged to leverage the `TimingScope` RAII-style wrapper around
/// `Timer`s.
///
/// Users can provide their own implementation of `TimingManager`, or use the
/// default `DefaultTimingManager` implementation in MLIR. Implementations
/// override the various protected virtual functions to create, nest, start, and
/// stop timers. A common pattern is for subclasses to provide a custom timer
/// class and simply pass pointers to instances of this class around as the
/// opaque timer handle. The manager itself can then forward callbacks to the
/// this class. Alternatively, external timing libraries may return their own
/// opaque handles for timing scopes.
///
/// For example:
/// ```
/// void doWork(TimingManager &tm) {
///   auto root = tm.getRootScope();
///
///   {
///     auto scope = root.nest("First");
///     doSomeWork();
///     // <-- "First" timer stops here
///   }
///
///   auto scope = root.nest("Second");
///   doEvenMoreWork();
///   scope.stop(); // <-- "Second" timer stops here
///
///   // <-- Root timer stops here
/// }
/// ```
class TimingManager {
public:
  explicit TimingManager();
  virtual ~TimingManager();

  /// Get the root timer of this timing manager. The returned timer must be
  /// started and stopped manually. Execution time can be measured by nesting
  /// timers within this root timer and starting/stopping them as appropriate.
  /// Use this function only if you need access to the timer itself. Otherwise
  /// consider the more convenient `getRootScope()` which offers an RAII-style
  /// wrapper around the timer.
  Timer getRootTimer();

  /// Get the root timer of this timing manager wrapped in a `TimingScope` for
  /// convenience. Automatically starts the timer and stops it as soon as the
  /// `TimingScope` is destroyed, e.g. when it goes out of scope.
  TimingScope getRootScope();

protected:
  // Allow `Timer` access to the protected callbacks.
  friend class Timer;

  //===--------------------------------------------------------------------===//
  // Callbacks
  //
  // See the corresponding functions in `Timer` for additional details.

  /// Return the root timer. Implementations should return `std::nullopt` if the
  /// collection of timing samples is disabled. This will cause the timers
  /// constructed from the manager to be tombstones which can be skipped
  /// quickly.
  virtual std::optional<void *> rootTimer() = 0;

  /// Start the timer with the given handle.
  virtual void startTimer(void *handle) = 0;

  /// Stop the timer with the given handle.
  virtual void stopTimer(void *handle) = 0;

  /// Create a child timer nested within the one with the given handle. The `id`
  /// parameter is used to uniquely identify the timer within its parent.
  /// Multiple calls to this function with the same `handle` and `id` should
  /// return the same timer, or at least cause the samples of the returned
  /// timers to be combined for the final timing results.
  virtual void *nestTimer(void *handle, const void *id,
                          function_ref<std::string()> nameBuilder) = 0;

  /// Hide the timer in timing reports and directly show its children. This is
  /// merely a hint that implementations are free to ignore.
  virtual void hideTimer(void *handle) {}

protected:
  const std::unique_ptr<detail::TimingManagerImpl> impl;

  // Allow `TimingIdentifier::get` access to the private impl details.
  friend class TimingIdentifier;

private:
  // Disallow copying the manager.
  TimingManager(const TimingManager &) = delete;
  void operator=(const TimingManager &) = delete;
};

//===----------------------------------------------------------------------===//
// Timer
//===----------------------------------------------------------------------===//

/// A handle for a timer in a `TimingManager`.
///
/// This class encapsulates a pointer to a `TimingManager` and an opaque handle
/// to a timer running within that manager. Libraries and infrastructure code
/// operate on `Timer` rather than any concrete classes handed out by custom
/// manager implementations.
class Timer {
public:
  Timer() = default;
  Timer(const Timer &other) = default;
  Timer(Timer &&other) : Timer(other) {
    other.tm = nullptr;
    other.handle = nullptr;
  }

  Timer &operator=(Timer &&other) {
    tm = other.tm;
    handle = other.handle;
    other.tm = nullptr;
    other.handle = nullptr;
    return *this;
  }

  /// Returns whether this is a valid timer handle. Invalid timer handles are
  /// used when timing is disabled in the `TimingManager` to keep the impact on
  /// performance low.
  explicit operator bool() const { return tm != nullptr; }

  /// Start the timer. This must be accompanied by a corresponding call to
  /// `stop()` at a later point.
  void start() {
    if (tm)
      tm->startTimer(handle);
  }

  /// Stop the timer. This must have been preceded by a corresponding call to
  /// `start()` at an earlier point.
  void stop() {
    if (tm)
      tm->stopTimer(handle);
  }

  /// Create a child timer nested within this one. Multiple calls to this
  /// function with the same unique identifier `id` will return the same child
  /// timer. The timer must have been started when calling this function.
  ///
  /// This function can be called from other threads, as long as this timer
  /// is not stopped before any uses of the child timer on the other thread are
  /// stopped.
  ///
  /// The `nameBuilder` function is not guaranteed to be called.
  Timer nest(const void *id, function_ref<std::string()> nameBuilder) {
    return tm ? Timer(*tm, tm->nestTimer(handle, id, nameBuilder)) : Timer();
  }

  /// See above.
  Timer nest(TimingIdentifier name) {
    return tm ? nest(name.getAsOpaquePointer(), [=]() { return name.str(); })
              : Timer();
  }

  /// See above.
  Timer nest(StringRef name) {
    return tm ? nest(TimingIdentifier::get(name, *tm)) : Timer();
  }

  /// Hide the timer in timing reports and directly show its children.
  void hide() {
    if (tm)
      tm->hideTimer(handle);
  }

protected:
  Timer(TimingManager &tm, void *handle) : tm(&tm), handle(handle) {}

  // Allow the `TimingManager` access to the above constructor.
  friend class TimingManager;

private:
  /// The associated timing manager.
  TimingManager *tm = nullptr;
  /// An opaque handle that identifies the timer in the timing manager
  /// implementation.
  void *handle = nullptr;
};

//===----------------------------------------------------------------------===//
// TimingScope
//===----------------------------------------------------------------------===//

/// An RAII-style wrapper around a timer that ensures the timer is properly
/// started and stopped.
class TimingScope {
public:
  TimingScope() {}
  TimingScope(const Timer &other) : timer(other) {
    if (timer)
      timer.start();
  }
  TimingScope(Timer &&other) : timer(std::move(other)) {
    if (timer)
      timer.start();
  }
  TimingScope(TimingScope &&other) : timer(std::move(other.timer)) {}
  ~TimingScope() { stop(); }

  TimingScope &operator=(TimingScope &&other) {
    stop();
    timer = std::move(other.timer);
    return *this;
  }

  /// Check if the timing scope actually contains a valid timer.
  explicit operator bool() const { return bool(timer); }

  // Disable copying of the `TimingScope`.
  TimingScope(const TimingScope &) = delete;
  TimingScope &operator=(const TimingScope &) = delete;

  /// Manually stop the timer early.
  void stop() {
    timer.stop();
    timer = Timer();
  }

  /// Create a nested timing scope.
  ///
  /// This returns a new `TimingScope` with a timer nested within the current
  /// scope. In this fashion, the time in this scope may be further subdivided
  /// in a more fine-grained fashion.
  template <typename... Args>
  TimingScope nest(Args... args) {
    return TimingScope(std::move(timer.nest(std::forward<Args>(args)...)));
  }

  /// Hide the timer in timing reports and directly show its children.
  void hide() { timer.hide(); }

private:
  /// The wrapped timer.
  Timer timer;
};

//===----------------------------------------------------------------------===//
// OutputStrategy
//===----------------------------------------------------------------------===//

/// Simple record class to record timing information.
struct TimeRecord {
  TimeRecord(double wall = 0.0, double user = 0.0) : wall(wall), user(user) {}

  TimeRecord &operator+=(const TimeRecord &other) {
    wall += other.wall;
    user += other.user;
    return *this;
  }

  TimeRecord &operator-=(const TimeRecord &other) {
    wall -= other.wall;
    user -= other.user;
    return *this;
  }

  double wall, user;
};

/// Facilities for printing timing reports to various output formats.
///
/// This is an abstract class that serves as the foundation for printing.
/// Users can implement additional output formats by extending this abstract
/// class.
class OutputStrategy {
public:
  OutputStrategy(raw_ostream &os) : os(os) {}
  virtual ~OutputStrategy() = default;

  virtual void printHeader(const TimeRecord &total) = 0;
  virtual void printFooter() = 0;
  virtual void printTime(const TimeRecord &time, const TimeRecord &total) = 0;
  virtual void printListEntry(StringRef name, const TimeRecord &time,
                              const TimeRecord &total,
                              bool lastEntry = false) = 0;
  virtual void printTreeEntry(unsigned indent, StringRef name,
                              const TimeRecord &time,
                              const TimeRecord &total) = 0;
  virtual void printTreeEntryEnd(unsigned indent, bool lastEntry = false) = 0;

  raw_ostream &os;
};

//===----------------------------------------------------------------------===//
// DefaultTimingManager
//===----------------------------------------------------------------------===//

/// Facilities for time measurement and report printing to an output stream.
///
/// This is MLIR's default implementation of a `TimingManager`. Prints an
/// execution time report upon destruction, or manually through `print()`. By
/// default the results are printed in `DisplayMode::Tree` mode to stderr.
/// Use `setEnabled(true)` to enable collection of timing samples; it is
/// disabled by default.
///
/// You should only instantiate a `DefaultTimingManager` if you are writing a
/// tool and want to pass a timing manager to the remaining infrastructure. If
/// you are writing library or infrastructure code, you should rather accept
/// the `TimingManager` base class to allow for users of your code to substitute
/// their own timing implementations. Also, if you only intend to collect time
/// samples, consider accepting a `Timer` or `TimingScope` instead.
class DefaultTimingManager : public TimingManager {
public:
  /// The different display modes for printing the timers.
  enum class DisplayMode {
    /// In this mode the results are displayed in a list sorted by total time,
    /// with timers aggregated into one unique result per timer name.
    List,

    /// In this mode the results are displayed in a tree view, with child timers
    /// nested under their parents.
    Tree,
  };

  /// The different output formats for printing the timers.
  enum class OutputFormat {
    /// In this format the results are displayed in text format.
    Text,

    /// In this format the results are displayed in JSON format.
    Json,
  };

  DefaultTimingManager();
  DefaultTimingManager(DefaultTimingManager &&rhs);
  ~DefaultTimingManager() override;

  // Disable copying of the `DefaultTimingManager`.
  DefaultTimingManager(const DefaultTimingManager &rhs) = delete;
  DefaultTimingManager &operator=(const DefaultTimingManager &rhs) = delete;

  /// Enable or disable execution time sampling.
  void setEnabled(bool enabled);

  /// Return whether execution time sampling is enabled.
  bool isEnabled() const;

  /// Change the display mode.
  void setDisplayMode(DisplayMode displayMode);

  /// Return the current display mode;
  DisplayMode getDisplayMode() const;

  /// Change the stream where the output will be printed to.
  void setOutput(std::unique_ptr<OutputStrategy> output);

  /// Print and clear the timing results. Only call this when there are no more
  /// references to nested timers around, as printing post-processes and clears
  /// the timers.
  void print();

  /// Clear the timing results. Only call this when there are no more references
  /// to nested timers around, as clearing invalidates them.
  void clear();

  /// Debug print the timer data structures to an output stream.
  void dumpTimers(raw_ostream &os = llvm::errs());

  /// Debug print the timers as a list. Only call this when there are no more
  /// references to nested timers around.
  void dumpAsList(raw_ostream &os = llvm::errs());

  /// Debug print the timers as a tree. Only call this when there are no
  /// more references to nested timers around.
  void dumpAsTree(raw_ostream &os = llvm::errs());

protected:
  // `TimingManager` callbacks
  std::optional<void *> rootTimer() override;
  void startTimer(void *handle) override;
  void stopTimer(void *handle) override;
  void *nestTimer(void *handle, const void *id,
                  function_ref<std::string()> nameBuilder) override;
  void hideTimer(void *handle) override;

private:
  const std::unique_ptr<detail::DefaultTimingManagerImpl> impl;
  std::unique_ptr<OutputStrategy> out;
};

/// Register a set of useful command-line options that can be used to configure
/// a `DefaultTimingManager`. The values of these options can be applied via the
/// `applyDefaultTimingManagerCLOptions` method.
void registerDefaultTimingManagerCLOptions();

/// Apply any values that were registered with
/// 'registerDefaultTimingManagerOptions' to a `DefaultTimingManager`.
void applyDefaultTimingManagerCLOptions(DefaultTimingManager &tm);

} // namespace mlir

#endif // MLIR_SUPPORT_TIMING_H
