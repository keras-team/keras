/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ITraceActivity.h"
#include "ThreadUtil.h"
#include "TraceSpan.h"

namespace libkineto {

// Link type, used in GenericTraceActivity.flow.type
constexpr unsigned int kLinkFwdBwd = 1;
constexpr unsigned int kLinkAsyncCpuGpu = 2;

// @lint-ignore-every CLANGTIDY
// cppcoreguidelines-non-private-member-variables-in-classes
// @lint-ignore-every CLANGTIDY cppcoreguidelines-pro-type-member-init
class GenericTraceActivity : public ITraceActivity {
 public:
  GenericTraceActivity()
      : activityType(ActivityType::ENUM_COUNT), traceSpan_(nullptr) {}

  GenericTraceActivity(
      const TraceSpan& trace,
      ActivityType type,
      const std::string& name)
      : activityType(type), activityName(name), traceSpan_(&trace) {}

  int64_t deviceId() const override {
    return device;
  }

  int64_t resourceId() const override {
    return resource;
  }

  int32_t getThreadId() const override {
    return threadId;
  }

  int64_t timestamp() const override {
    return startTime;
  }

  int64_t duration() const override {
    return endTime - startTime;
  }

  int64_t correlationId() const override {
    return id;
  }

  ActivityType type() const override {
    return activityType;
  }

  const ITraceActivity* linkedActivity() const override {
    return linked;
  }

  int flowType() const override {
    return flow.type;
  }

  int flowId() const override {
    return flow.id;
  }

  bool flowStart() const override {
    return flow.start;
  }

  const std::string name() const override {
    return activityName;
  }

  const TraceSpan* traceSpan() const override {
    return traceSpan_;
  }

  void log(ActivityLogger& logger) const override;

  // Encode client side metadata as a key/value
  template <typename ValType>
  void addMetadata(const std::string& key, const ValType& value) {
    metadataMap_.emplace(key, std::make_pair(fmt::format("{}", value), false));
  }

  void addMetadataQuoted(const std::string& key, const std::string& value) {
    metadataMap_.emplace(key, std::make_pair(value, true));
  }

  const std::string getMetadataValue(const std::string& key) const override {
    if (auto it = metadataMap_.find(key); it != metadataMap_.end()) {
      return it->second.first;
    }
    return "";
  }

  const std::string metadataJson() const override {
    std::stringstream json;
    bool first = true;
    for (const auto& [key, val] : metadataMap_) {
      if (!first) {
        json << ", ";
      }
      val.second ? json << fmt::format("\"{}\": \"{}\"", key, val.first)
                 : json << fmt::format("\"{}\": {}", key, val.first);
      first = false;
    }
    return json.str();
  }

  virtual ~GenericTraceActivity() override {}

  int64_t startTime{0};
  int64_t endTime{0};
  int32_t id{0};
  int32_t device{0};
  int32_t resource{0};
  int32_t threadId{0};
  ActivityType activityType;
  std::string activityName;
  struct Flow {
    Flow() : id(0), type(0), start(0) {}
    // Ids must be unique within each type
    uint32_t id : 27;
    // Type will be used to connect flows between profilers, as
    // well as look up flow information (name etc)
    uint32_t type : 4;
    uint32_t start : 1;
  } flow;
  const ITraceActivity* linked{nullptr};

 private:
  const TraceSpan* traceSpan_;
  // Metadata map: { key: (value, quoted)}
  std::unordered_map<std::string, std::pair<std::string, bool>> metadataMap_;
};

} // namespace libkineto
