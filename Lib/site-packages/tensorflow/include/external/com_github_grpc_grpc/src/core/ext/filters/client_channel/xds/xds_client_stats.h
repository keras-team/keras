/*
 *
 * Copyright 2018 gRPC authors.
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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_XDS_XDS_CLIENT_STATS_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_XDS_XDS_CLIENT_STATS_H

#include <grpc/support/port_platform.h>

#include <grpc/support/string_util.h>

#include "src/core/lib/gprpp/atomic.h"
#include "src/core/lib/gprpp/inlined_vector.h"
#include "src/core/lib/gprpp/map.h"
#include "src/core/lib/gprpp/memory.h"
#include "src/core/lib/gprpp/ref_counted.h"
#include "src/core/lib/gprpp/sync.h"
#include "src/core/lib/iomgr/exec_ctx.h"

namespace grpc_core {

class XdsLocalityName : public RefCounted<XdsLocalityName> {
 public:
  struct Less {
    bool operator()(const RefCountedPtr<XdsLocalityName>& lhs,
                    const RefCountedPtr<XdsLocalityName>& rhs) const {
      int cmp_result = strcmp(lhs->region_.get(), rhs->region_.get());
      if (cmp_result != 0) return cmp_result < 0;
      cmp_result = strcmp(lhs->zone_.get(), rhs->zone_.get());
      if (cmp_result != 0) return cmp_result < 0;
      return strcmp(lhs->sub_zone_.get(), rhs->sub_zone_.get()) < 0;
    }
  };

  XdsLocalityName(grpc_core::UniquePtr<char> region,
                  grpc_core::UniquePtr<char> zone,
                  grpc_core::UniquePtr<char> subzone)
      : region_(std::move(region)),
        zone_(std::move(zone)),
        sub_zone_(std::move(subzone)) {}

  bool operator==(const XdsLocalityName& other) const {
    return strcmp(region_.get(), other.region_.get()) == 0 &&
           strcmp(zone_.get(), other.zone_.get()) == 0 &&
           strcmp(sub_zone_.get(), other.sub_zone_.get()) == 0;
  }

  const char* region() const { return region_.get(); }
  const char* zone() const { return zone_.get(); }
  const char* sub_zone() const { return sub_zone_.get(); }

  const char* AsHumanReadableString() {
    if (human_readable_string_ == nullptr) {
      char* tmp;
      gpr_asprintf(&tmp, "{region=\"%s\", zone=\"%s\", sub_zone=\"%s\"}",
                   region_.get(), zone_.get(), sub_zone_.get());
      human_readable_string_.reset(tmp);
    }
    return human_readable_string_.get();
  }

 private:
  grpc_core::UniquePtr<char> region_;
  grpc_core::UniquePtr<char> zone_;
  grpc_core::UniquePtr<char> sub_zone_;
  grpc_core::UniquePtr<char> human_readable_string_;
};

// The stats classes (i.e., XdsClientStats, LocalityStats, and LoadMetric) can
// be taken a snapshot (and reset) to populate the load report. The snapshots
// are contained in the respective Snapshot structs. The Snapshot structs have
// no synchronization. The stats classes use several different synchronization
// methods. 1. Most of the counters are Atomic<>s for performance. 2. Some of
// the Map<>s are protected by Mutex if we are not guaranteed that the accesses
// to them are synchronized by the callers. 3. The Map<>s to which the accesses
// are already synchronized by the callers do not have additional
// synchronization here. Note that the Map<>s we mentioned in 2 and 3 refer to
// the map's tree structure rather than the content in each tree node.
class XdsClientStats {
 public:
  class LocalityStats : public RefCounted<LocalityStats> {
   public:
    class LoadMetric {
     public:
      struct Snapshot {
        bool IsAllZero() const;

        uint64_t num_requests_finished_with_metric;
        double total_metric_value;
      };

      // Returns a snapshot of this instance and reset all the accumulative
      // counters.
      Snapshot GetSnapshotAndReset();

     private:
      uint64_t num_requests_finished_with_metric_{0};
      double total_metric_value_{0};
    };

    using LoadMetricMap =
        std::map<grpc_core::UniquePtr<char>, LoadMetric, StringLess>;
    using LoadMetricSnapshotMap =
        std::map<grpc_core::UniquePtr<char>, LoadMetric::Snapshot, StringLess>;

    struct Snapshot {
      // TODO(juanlishen): Change this to const method when const_iterator is
      // added to Map<>.
      bool IsAllZero();

      uint64_t total_successful_requests;
      uint64_t total_requests_in_progress;
      uint64_t total_error_requests;
      uint64_t total_issued_requests;
      LoadMetricSnapshotMap load_metric_stats;
    };

    // Returns a snapshot of this instance and reset all the accumulative
    // counters.
    Snapshot GetSnapshotAndReset();

    // Each XdsLb::PickerWrapper holds a ref to the perspective LocalityStats.
    // If the refcount is 0, there won't be new calls recorded to the
    // LocalityStats, so the LocalityStats can be safely deleted when all the
    // in-progress calls have finished.
    // Only be called from the control plane combiner.
    void RefByPicker() { picker_refcount_.FetchAdd(1, MemoryOrder::ACQ_REL); }
    // Might be called from the control plane combiner or the data plane
    // combiner.
    // TODO(juanlishen): Once https://github.com/grpc/grpc/pull/19390 is merged,
    //  this method will also only be invoked in the control plane combiner.
    //  We may then be able to simplify the LocalityStats' lifetime by making it
    //  RefCounted<> and populating the protobuf in its dtor.
    void UnrefByPicker() { picker_refcount_.FetchSub(1, MemoryOrder::ACQ_REL); }
    // Only be called from the control plane combiner.
    // The only place where the picker_refcount_ can be increased is
    // RefByPicker(), which also can only be called from the control plane
    // combiner. Also, if the picker_refcount_ is 0, total_requests_in_progress_
    // can't be increased from 0. So it's safe to delete the LocalityStats right
    // after this method returns true.
    bool IsSafeToDelete() {
      return picker_refcount_.FetchAdd(0, MemoryOrder::ACQ_REL) == 0 &&
             total_requests_in_progress_.FetchAdd(0, MemoryOrder::ACQ_REL) == 0;
    }

    void AddCallStarted();
    void AddCallFinished(bool fail = false);

   private:
    Atomic<uint64_t> total_successful_requests_{0};
    Atomic<uint64_t> total_requests_in_progress_{0};
    // Requests that were issued (not dropped) but failed.
    Atomic<uint64_t> total_error_requests_{0};
    Atomic<uint64_t> total_issued_requests_{0};
    // Protects load_metric_stats_. A mutex is necessary because the length of
    // load_metric_stats_ can be accessed by both the callback intercepting the
    // call's recv_trailing_metadata (not from any combiner) and the load
    // reporting thread (from the control plane combiner).
    Mutex load_metric_stats_mu_;
    LoadMetricMap load_metric_stats_;
    // Can be accessed from either the control plane combiner or the data plane
    // combiner.
    Atomic<uint8_t> picker_refcount_{0};
  };

  // TODO(juanlishen): The value type of Map<> must be movable in current
  // implementation. To avoid making LocalityStats movable, we wrap it by
  // std::unique_ptr<>. We should remove this wrapper if the value type of Map<>
  // doesn't have to be movable.
  using LocalityStatsMap =
      std::map<RefCountedPtr<XdsLocalityName>, RefCountedPtr<LocalityStats>,
               XdsLocalityName::Less>;
  using LocalityStatsSnapshotMap =
      std::map<RefCountedPtr<XdsLocalityName>, LocalityStats::Snapshot,
               XdsLocalityName::Less>;
  using DroppedRequestsMap =
      std::map<grpc_core::UniquePtr<char>, uint64_t, StringLess>;
  using DroppedRequestsSnapshotMap = DroppedRequestsMap;

  struct Snapshot {
    // TODO(juanlishen): Change this to const method when const_iterator is
    // added to Map<>.
    bool IsAllZero();

    LocalityStatsSnapshotMap upstream_locality_stats;
    uint64_t total_dropped_requests;
    DroppedRequestsSnapshotMap dropped_requests;
    // The actual load report interval.
    grpc_millis load_report_interval;
  };

  // Returns a snapshot of this instance and reset all the accumulative
  // counters.
  Snapshot GetSnapshotAndReset();

  void MaybeInitLastReportTime();
  RefCountedPtr<LocalityStats> FindLocalityStats(
      const RefCountedPtr<XdsLocalityName>& locality_name);
  void PruneLocalityStats();
  void AddCallDropped(const grpc_core::UniquePtr<char>& category);

 private:
  // The stats for each locality.
  LocalityStatsMap upstream_locality_stats_;
  Atomic<uint64_t> total_dropped_requests_{0};
  // Protects dropped_requests_. A mutex is necessary because the length of
  // dropped_requests_ can be accessed by both the picker (from data plane
  // combiner) and the load reporting thread (from the control plane combiner).
  Mutex dropped_requests_mu_;
  DroppedRequestsMap dropped_requests_;
  // The timestamp of last reporting. For the LB-policy-wide first report, the
  // last_report_time is the time we scheduled the first reporting timer.
  grpc_millis last_report_time_ = -1;
};

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_XDS_XDS_CLIENT_STATS_H */
