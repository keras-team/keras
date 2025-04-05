/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

// Stages in libkineto used when pushing logs to UST Logger.
constexpr char kWarmUpStage[] = "Warm Up";
constexpr char kCollectionStage[] = "Collection";
constexpr char kPostProcessingStage[] = "Post Processing";

// Special string in UST for determining if traces are empty
constexpr char kEmptyTrace[] =
    "No Valid Trace Events (CPU/GPU) found. Outputting empty trace.";

#if !USE_GOOGLE_LOG

#include <map>
#include <vector>

#include <stdint.h>

namespace libkineto {

enum LoggerOutputType {
  VERBOSE = 0,
  INFO = 1,
  WARNING = 2,
  STAGE = 3,
  ERROR = 4,
  ENUM_COUNT = 5
};

const char* toString(LoggerOutputType t);
LoggerOutputType toLoggerOutputType(const std::string& str);

constexpr int LoggerTypeCount = (int)LoggerOutputType::ENUM_COUNT;

class ILoggerObserver {
 public:
  virtual ~ILoggerObserver() = default;
  virtual void write(const std::string& message, LoggerOutputType ot) = 0;
  virtual const std::map<LoggerOutputType, std::vector<std::string>>
  extractCollectorMetadata() = 0;
  virtual void reset() = 0;
  virtual void addDevice(const int64_t device) = 0;
  virtual void setTraceDurationMS(const int64_t duration) = 0;
  virtual void addEventCount(const int64_t count) = 0;
  virtual void setTraceID(const std::string&) {}
  virtual void setGroupTraceID(const std::string&) {}
  virtual void addDestination(const std::string& dest) = 0;
  virtual void setTriggerOnDemand() {}
  virtual void addMetadata(
      const std::string& key,
      const std::string& value) = 0;
};

} // namespace libkineto

#endif // !USE_GOOGLE_LOG
