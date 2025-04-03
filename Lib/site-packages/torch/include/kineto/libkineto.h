/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Mediator for initialization and profiler control

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "ActivityProfilerInterface.h"
#include "ActivityTraceInterface.h"
#include "ActivityType.h"
#include "ClientInterface.h"
#include "GenericTraceActivity.h"
#include "IActivityProfiler.h"
#include "ILoggerObserver.h"
#include "LoggingAPI.h"
#include "TraceSpan.h"

#include "ThreadUtil.h"

extern "C" {
void suppressLibkinetoLogMessages();
int InitializeInjection(void);
void libkineto_init(bool cpuOnly, bool logOnError);
bool hasTestEnvVar();
}

namespace libkineto {

class Config;
class ConfigLoader;

struct CpuTraceBuffer {
  template <class... Args>
  void emplace_activity(Args&&... args) {
    activities.emplace_back(
        std::make_unique<GenericTraceActivity>(std::forward<Args>(args)...));
  }

  static GenericTraceActivity& toRef(
      std::unique_ptr<GenericTraceActivity>& ref) {
    return *ref;
  }

  static const GenericTraceActivity& toRef(
      const std::unique_ptr<GenericTraceActivity>& ref) {
    return *ref;
  }

  TraceSpan span{0, 0, "none"};
  int gpuOpCount;
  std::deque<std::unique_ptr<GenericTraceActivity>> activities;
};

using ChildActivityProfilerFactory =
    std::function<std::unique_ptr<IActivityProfiler>()>;

class LibkinetoApi {
 public:
  explicit LibkinetoApi(ConfigLoader& configLoader)
      : configLoader_(configLoader) {}

  // Called by client that supports tracing API.
  // libkineto can still function without this.
  void registerClient(ClientInterface* client);

  // Called by libkineto on init
  void registerProfiler(std::unique_ptr<ActivityProfilerInterface> profiler) {
    activityProfiler_ = std::move(profiler);
    initClientIfRegistered();
  }

  ActivityProfilerInterface& activityProfiler() {
    return *activityProfiler_;
  }

  ClientInterface* client() {
    return client_;
  }

  void initProfilerIfRegistered() {
    static std::once_flag once;
    if (activityProfiler_) {
      std::call_once(once, [this] {
        if (!activityProfiler_->isInitialized()) {
          activityProfiler_->init();
          initChildActivityProfilers();
        }
      });
    }
  }

  bool isProfilerInitialized() const {
    return activityProfiler_ && activityProfiler_->isInitialized();
  }

  bool isProfilerRegistered() const {
    return activityProfiler_ != nullptr;
  }

  void suppressLogMessages() {
    suppressLibkinetoLogMessages();
  }

  void resetKinetoTLS() {
    resetTLS();
  }

  // Provides access to profier configuration manaegement
  ConfigLoader& configLoader() {
    return configLoader_;
  }

  void registerProfilerFactory(ChildActivityProfilerFactory factory) {
    if (isProfilerInitialized()) {
      activityProfiler_->addChildActivityProfiler(factory());
    } else {
      childProfilerFactories_.push_back(factory);
    }
  }

 private:
  void initChildActivityProfilers() {
    if (!isProfilerInitialized()) {
      return;
    }
    for (const auto& factory : childProfilerFactories_) {
      activityProfiler_->addChildActivityProfiler(factory());
    }
    childProfilerFactories_.clear();
  }

  // Client is initialized once both it and libkineto has registered
  void initClientIfRegistered();

  ConfigLoader& configLoader_;
  std::unique_ptr<ActivityProfilerInterface> activityProfiler_{};
  ClientInterface* client_{};
  int32_t clientRegisterThread_{0};

  std::vector<ChildActivityProfilerFactory> childProfilerFactories_;
};

// Singleton
LibkinetoApi& api();

} // namespace libkineto
