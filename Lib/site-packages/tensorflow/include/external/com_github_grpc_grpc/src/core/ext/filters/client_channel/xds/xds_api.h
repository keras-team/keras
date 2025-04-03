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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_XDS_XDS_API_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_XDS_XDS_API_H

#include <grpc/support/port_platform.h>

#include <stdint.h>

#include <grpc/slice_buffer.h>

#include "src/core/ext/filters/client_channel/server_address.h"
#include "src/core/ext/filters/client_channel/xds/xds_bootstrap.h"
#include "src/core/ext/filters/client_channel/xds/xds_client_stats.h"

namespace grpc_core {

class XdsPriorityListUpdate {
 public:
  struct LocalityMap {
    struct Locality {
      bool operator==(const Locality& other) const {
        return *name == *other.name && serverlist == other.serverlist &&
               lb_weight == other.lb_weight && priority == other.priority;
      }

      // This comparator only compares the locality names.
      struct Less {
        bool operator()(const Locality& lhs, const Locality& rhs) const {
          return XdsLocalityName::Less()(lhs.name, rhs.name);
        }
      };

      RefCountedPtr<XdsLocalityName> name;
      ServerAddressList serverlist;
      uint32_t lb_weight;
      uint32_t priority;
    };

    bool Contains(const RefCountedPtr<XdsLocalityName>& name) const {
      return localities.find(name) != localities.end();
    }

    size_t size() const { return localities.size(); }

    std::map<RefCountedPtr<XdsLocalityName>, Locality, XdsLocalityName::Less>
        localities;
  };

  bool operator==(const XdsPriorityListUpdate& other) const;
  bool operator!=(const XdsPriorityListUpdate& other) const {
    return !(*this == other);
  }

  void Add(LocalityMap::Locality locality);

  const LocalityMap* Find(uint32_t priority) const;

  bool Contains(uint32_t priority) const {
    return priority < priorities_.size();
  }
  bool Contains(const RefCountedPtr<XdsLocalityName>& name);

  bool empty() const { return priorities_.empty(); }
  size_t size() const { return priorities_.size(); }

  // Callers should make sure the priority list is non-empty.
  uint32_t LowestPriority() const {
    return static_cast<uint32_t>(priorities_.size()) - 1;
  }

 private:
  InlinedVector<LocalityMap, 2> priorities_;
};

// There are two phases of accessing this class's content:
// 1. to initialize in the control plane combiner;
// 2. to use in the data plane combiner.
// So no additional synchronization is needed.
class XdsDropConfig : public RefCounted<XdsDropConfig> {
 public:
  struct DropCategory {
    bool operator==(const DropCategory& other) const {
      return strcmp(name.get(), other.name.get()) == 0 &&
             parts_per_million == other.parts_per_million;
    }

    grpc_core::UniquePtr<char> name;
    const uint32_t parts_per_million;
  };

  using DropCategoryList = InlinedVector<DropCategory, 2>;

  void AddCategory(grpc_core::UniquePtr<char> name,
                   uint32_t parts_per_million) {
    drop_category_list_.emplace_back(
        DropCategory{std::move(name), parts_per_million});
  }

  // The only method invoked from the data plane combiner.
  bool ShouldDrop(const grpc_core::UniquePtr<char>** category_name) const;

  const DropCategoryList& drop_category_list() const {
    return drop_category_list_;
  }

  bool operator==(const XdsDropConfig& other) const {
    return drop_category_list_ == other.drop_category_list_;
  }
  bool operator!=(const XdsDropConfig& other) const {
    return !(*this == other);
  }

 private:
  DropCategoryList drop_category_list_;
};

struct EdsUpdate {
  XdsPriorityListUpdate priority_list_update;
  RefCountedPtr<XdsDropConfig> drop_config;
  bool drop_all = false;
};

struct CdsUpdate {
  // The name to use in the EDS request.
  // If null, the cluster name will be used.
  grpc_core::UniquePtr<char> eds_service_name;
  // The LRS server to use for load reporting.
  // If null, load reporting will be disabled.
  // If set to the empty string, will use the same server we obtained
  // the CDS data from.
  grpc_core::UniquePtr<char> lrs_load_reporting_server_name;
};

// Creates an EDS request querying \a service_name.
grpc_slice XdsEdsRequestCreateAndEncode(const char* server_name,
                                        const XdsBootstrap::Node* node,
                                        const char* build_version);

// Parses the EDS response and returns the args to update locality map. If there
// is any error, the output update is invalid.
grpc_error* XdsEdsResponseDecodeAndParse(const grpc_slice& encoded_response,
                                         EdsUpdate* update);

// Creates an LRS request querying \a server_name.
grpc_slice XdsLrsRequestCreateAndEncode(const char* server_name,
                                        const XdsBootstrap::Node* node,
                                        const char* build_version);

// Creates an LRS request sending client-side load reports. If all the counters
// in \a client_stats are zero, returns empty slice.
grpc_slice XdsLrsRequestCreateAndEncode(const char* server_name,
                                        XdsClientStats* client_stats);

// Parses the LRS response and returns \a cluster_name and \a
// load_reporting_interval for client-side load reporting. If there is any
// error, the output config is invalid.
grpc_error* XdsLrsResponseDecodeAndParse(
    const grpc_slice& encoded_response,
    grpc_core::UniquePtr<char>* cluster_name,
    grpc_millis* load_reporting_interval);

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_XDS_XDS_API_H */
