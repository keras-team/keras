//
// Copyright 2019 gRPC authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_XDS_XDS_BOOTSTRAP_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_XDS_XDS_BOOTSTRAP_H

#include <grpc/support/port_platform.h>

#include <vector>

#include <grpc/impl/codegen/slice.h>

#include "src/core/lib/gprpp/inlined_vector.h"
#include "src/core/lib/gprpp/map.h"
#include "src/core/lib/gprpp/memory.h"
#include "src/core/lib/iomgr/error.h"
#include "src/core/lib/json/json.h"

namespace grpc_core {

class XdsBootstrap {
 public:
  struct MetadataValue {
    enum class Type { MD_NULL, DOUBLE, STRING, BOOL, STRUCT, LIST };
    Type type = Type::MD_NULL;
    // TODO(roth): Once we can use C++17, these can be in a std::variant.
    double double_value;
    const char* string_value;
    bool bool_value;
    std::map<const char*, MetadataValue, StringLess> struct_value;
    std::vector<MetadataValue> list_value;
  };

  struct Node {
    const char* id = nullptr;
    const char* cluster = nullptr;
    const char* locality_region = nullptr;
    const char* locality_zone = nullptr;
    const char* locality_subzone = nullptr;
    std::map<const char*, MetadataValue, StringLess> metadata;
  };

  struct ChannelCreds {
    const char* type = nullptr;
    grpc_json* config = nullptr;
  };

  struct XdsServer {
    const char* server_uri = nullptr;
    InlinedVector<ChannelCreds, 1> channel_creds;
  };

  // If *error is not GRPC_ERROR_NONE after returning, then there was an
  // error reading the file.
  static std::unique_ptr<XdsBootstrap> ReadFromFile(grpc_error** error);

  // Do not instantiate directly -- use ReadFromFile() above instead.
  XdsBootstrap(grpc_slice contents, grpc_error** error);
  ~XdsBootstrap();

  // TODO(roth): We currently support only one server. Fix this when we
  // add support for fallback for the xds channel.
  const XdsServer& server() const { return servers_[0]; }
  const Node* node() const { return node_.get(); }

 private:
  grpc_error* ParseXdsServerList(grpc_json* json);
  grpc_error* ParseXdsServer(grpc_json* json, size_t idx);
  grpc_error* ParseChannelCredsArray(grpc_json* json, XdsServer* server);
  grpc_error* ParseChannelCreds(grpc_json* json, size_t idx, XdsServer* server);
  grpc_error* ParseNode(grpc_json* json);
  grpc_error* ParseLocality(grpc_json* json);

  InlinedVector<grpc_error*, 1> ParseMetadataStruct(
      grpc_json* json,
      std::map<const char*, MetadataValue, StringLess>* result);
  InlinedVector<grpc_error*, 1> ParseMetadataList(
      grpc_json* json, std::vector<MetadataValue>* result);
  grpc_error* ParseMetadataValue(grpc_json* json, size_t idx,
                                 MetadataValue* result);

  grpc_slice contents_;
  grpc_json* tree_ = nullptr;

  InlinedVector<XdsServer, 1> servers_;
  std::unique_ptr<Node> node_;
};

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_XDS_XDS_BOOTSTRAP_H */
