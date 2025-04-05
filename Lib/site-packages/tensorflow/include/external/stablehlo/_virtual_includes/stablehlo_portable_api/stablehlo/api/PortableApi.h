/* Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_API_PORTABLEAPI_H
#define STABLEHLO_API_PORTABLEAPI_H

#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace stablehlo {

/// Return the current version for portable API.
/// Increments on all meaningful changes to this file.
inline int64_t getApiVersion() { return 9; }

// Get the smaller version between version1 and version2.
FailureOr<std::string> getSmallerVersion(llvm::StringRef version1,
                                         llvm::StringRef version2);

// Get the current StableHLO version.
//
// This value can be used as the `targetVersion` argument to
// `serializePortableArtifact`.
std::string getCurrentVersion();

// Get the minimum supported StableHLO version.
//
// This value can be used as the `targetVersion` argument to
// `serializePortableArtifact`.
//
// Each StableHLO version `producer_version` has a compatibility window,
// i.e. range of versions [`consumer_version_min`, `consumer_version_max`],
// where StableHLO portable artifacts serialized by `producer_version`
// can be deserialized by `consumer_version` within the window.
// See https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md
// for the exact extent of these compatibility guarantees.
//
// This function returns `consumer_version_min` for the current StableHLO
// version. It can be used maximize forward compatibility, i.e. to maximize how
// far into the past we can go and still have the payloads produced by
// `serializePortableArtifact` compatible with potential consumers from the past
std::string getMinimumVersion();

// Write a StableHLO program expressed as a string (either prettyprinted MLIR
// module or MLIR bytecode) to a portable artifact.
// Can fail if `moduleStr` cannot be parsed, or if it cannot be expressed in the
// `targetVersion` version of StableHLO, e.g. if it's using new or removed
// features, or if it involves unsupported dialects.
LogicalResult serializePortableArtifact(llvm::StringRef moduleStr,
                                        llvm::StringRef targetVersion,
                                        llvm::raw_ostream& os);

// Read a StableHLO program from a portable artifact, returning the module as
// MLIR bytecode. Note, this bytecode returned is not a portable artifact,
// and has the stability of returning textual assembly format. Bytecode is
// returned here since it is more compact and faster to read and write.
// Can fail if `artifactStr` cannot be expressed in the current version of
// StableHLO, e.g. if it's using incompatible features. Returns failure if
// `artifactStr` is invalid or fails to deserialize.
LogicalResult deserializePortableArtifact(llvm::StringRef artifactStr,
                                          llvm::raw_ostream& os);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_API_PORTABLEAPI_H
