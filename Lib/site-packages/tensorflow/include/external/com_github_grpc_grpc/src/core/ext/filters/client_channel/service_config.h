//
// Copyright 2016 gRPC authors.
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

#ifndef GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_SERVICE_CONFIG_H
#define GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_SERVICE_CONFIG_H

#include <grpc/support/port_platform.h>

#include <grpc/impl/codegen/grpc_types.h>
#include <grpc/support/string_util.h>

#include "src/core/lib/gprpp/inlined_vector.h"
#include "src/core/lib/gprpp/ref_counted.h"
#include "src/core/lib/gprpp/ref_counted_ptr.h"
#include "src/core/lib/iomgr/error.h"
#include "src/core/lib/json/json.h"
#include "src/core/lib/slice/slice_hash_table.h"

// The main purpose of the code here is to parse the service config in
// JSON form, which will look like this:
//
// {
//   "loadBalancingPolicy": "string",  // optional
//   "methodConfig": [  // array of one or more method_config objects
//     {
//       "name": [  // array of one or more name objects
//         {
//           "service": "string",  // required
//           "method": "string",  // optional
//         }
//       ],
//       // remaining fields are optional.
//       // see https://developers.google.com/protocol-buffers/docs/proto3#json
//       // for format details.
//       "waitForReady": bool,
//       "timeout": "duration_string",
//       "maxRequestMessageBytes": "int64_string",
//       "maxResponseMessageBytes": "int64_string",
//     }
//   ]
// }

namespace grpc_core {

class ServiceConfig : public RefCounted<ServiceConfig> {
 public:
  /// This is the base class that all service config parsers MUST use to store
  /// parsed service config data.
  class ParsedConfig {
   public:
    virtual ~ParsedConfig() = default;
  };

  /// This is the base class that all service config parsers should derive from.
  class Parser {
   public:
    virtual ~Parser() = default;

    virtual std::unique_ptr<ParsedConfig> ParseGlobalParams(
        const grpc_json* /* json */, grpc_error** error) {
      // Avoid unused parameter warning on debug-only parameter
      (void)error;
      GPR_DEBUG_ASSERT(error != nullptr);
      return nullptr;
    }

    virtual std::unique_ptr<ParsedConfig> ParsePerMethodParams(
        const grpc_json* /* json */, grpc_error** error) {
      // Avoid unused parameter warning on debug-only parameter
      (void)error;
      GPR_DEBUG_ASSERT(error != nullptr);
      return nullptr;
    }
  };

  static constexpr int kNumPreallocatedParsers = 4;
  typedef InlinedVector<std::unique_ptr<ParsedConfig>, kNumPreallocatedParsers>
      ParsedConfigVector;

  /// When a service config is applied to a call in the client_channel_filter,
  /// we create an instance of this object and store it in the call_data for
  /// client_channel. A pointer to this object is also stored in the
  /// call_context, so that future filters can easily access method and global
  /// parameters for the call.
  class CallData {
   public:
    CallData() = default;
    CallData(RefCountedPtr<ServiceConfig> svc_cfg, const grpc_slice& path)
        : service_config_(std::move(svc_cfg)) {
      if (service_config_ != nullptr) {
        method_params_vector_ =
            service_config_->GetMethodParsedConfigVector(path);
      }
    }

    ServiceConfig* service_config() { return service_config_.get(); }

    ParsedConfig* GetMethodParsedConfig(size_t index) const {
      return method_params_vector_ != nullptr
                 ? (*method_params_vector_)[index].get()
                 : nullptr;
    }

    ParsedConfig* GetGlobalParsedConfig(size_t index) const {
      return service_config_->GetGlobalParsedConfig(index);
    }

   private:
    RefCountedPtr<ServiceConfig> service_config_;
    const ParsedConfigVector* method_params_vector_ = nullptr;
  };

  /// Creates a new service config from parsing \a json_string.
  /// Returns null on parse error.
  static RefCountedPtr<ServiceConfig> Create(const char* json,
                                             grpc_error** error);

  // Takes ownership of \a json_tree.
  ServiceConfig(grpc_core::UniquePtr<char> service_config_json,
                grpc_core::UniquePtr<char> json_string, grpc_json* json_tree,
                grpc_error** error);
  ~ServiceConfig();

  const char* service_config_json() const { return service_config_json_.get(); }

  /// Retrieves the global parsed config at index \a index. The
  /// lifetime of the returned object is tied to the lifetime of the
  /// ServiceConfig object.
  ParsedConfig* GetGlobalParsedConfig(size_t index) {
    GPR_DEBUG_ASSERT(index < parsed_global_configs_.size());
    return parsed_global_configs_[index].get();
  }

  /// Retrieves the vector of parsed configs for the method identified
  /// by \a path.  The lifetime of the returned vector and contained objects
  /// is tied to the lifetime of the ServiceConfig object.
  const ParsedConfigVector* GetMethodParsedConfigVector(const grpc_slice& path);

  /// Globally register a service config parser. On successful registration, it
  /// returns the index at which the parser was registered. On failure, -1 is
  /// returned. Each new service config update will go through all the
  /// registered parser. Each parser is responsible for reading the service
  /// config json and returning a parsed config. This parsed config can later be
  /// retrieved using the same index that was returned at registration time.
  static size_t RegisterParser(std::unique_ptr<Parser> parser);

  static void Init();

  static void Shutdown();

 private:
  // Helper functions to parse the service config
  grpc_error* ParseGlobalParams(const grpc_json* json_tree);
  grpc_error* ParsePerMethodParams(const grpc_json* json_tree);

  // Returns the number of names specified in the method config \a json.
  static int CountNamesInMethodConfig(grpc_json* json);

  // Returns a path string for the JSON name object specified by \a json.
  // Returns null on error, and stores error in \a error.
  static grpc_core::UniquePtr<char> ParseJsonMethodName(grpc_json* json,
                                                        grpc_error** error);

  grpc_error* ParseJsonMethodConfigToServiceConfigVectorTable(
      const grpc_json* json,
      SliceHashTable<const ParsedConfigVector*>::Entry* entries, size_t* idx);

  grpc_core::UniquePtr<char> service_config_json_;
  grpc_core::UniquePtr<char> json_string_;  // Underlying storage for json_tree.
  grpc_json* json_tree_;

  InlinedVector<std::unique_ptr<ParsedConfig>, kNumPreallocatedParsers>
      parsed_global_configs_;
  // A map from the method name to the parsed config vector. Note that we are
  // using a raw pointer and not a unique pointer so that we can use the same
  // vector for multiple names.
  RefCountedPtr<SliceHashTable<const ParsedConfigVector*>>
      parsed_method_configs_table_;
  // Storage for all the vectors that are being used in
  // parsed_method_configs_table_.
  InlinedVector<std::unique_ptr<ParsedConfigVector>, 32>
      parsed_method_config_vectors_storage_;
};

}  // namespace grpc_core

#endif /* GRPC_CORE_EXT_FILTERS_CLIENT_CHANNEL_SERVICE_CONFIG_H */
