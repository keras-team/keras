/* Copyright 2023-2024 The StableHLO Authors.

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

#ifndef STABLEHLO_REFERENCE_PROCESS_H
#define STABLEHLO_REFERENCE_PROCESS_H

#include <cstdint>

#include "stablehlo/reference/ProcessGrid.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

/// StableHLO process.
class Process {
 public:
  /// \name Constructors
  /// @{
  Process(ProcessId id, ProcessGrid *grid);
  /// @}

  /// See `ProcessGrid::crossPartition`.
  ProcessGroups crossPartition(
      SmallVector<SmallVector<uint32_t>> partitionGroups);

  /// See `ProcessGrid::crossReplica`.
  ProcessGroups crossReplica(SmallVector<SmallVector<uint32_t>> replicaGroups);

  /// See `ProcessGrid::crossReplicaAndPartition`.
  ProcessGroups crossReplicaAndPartition(
      SmallVector<SmallVector<uint32_t>> replicaGroups);

  /// See `ProcessGrid::flattenedIds`.
  ProcessGroups flattenedIds(
      SmallVector<SmallVector<uint32_t>> flattenedIdGroups);

  /// See "ProcessGrid::infeed".
  StringAttr infeed();

  /// Getter for the underlying StableHLO `process_id`.
  ProcessId getId();

  /// See `ProcessGrid::outfeed`.
  void outfeed(ArrayRef<Tensor> inputs);

  /// See `ProcessGrid::recv`.
  SmallVector<Tensor> recv(ChannelId channelId);

  /// See `ProcessGrid::rendezvous`.
  RendezvousResult rendezvous(ProcessGroup processGroup, ChannelId channelId,
                              ArrayRef<Tensor> operands);

  /// See `ProcessGrid::send`.
  void send(ArrayRef<Tensor> inputs, ChannelId channelId);

 private:
  /// StableHLO `process_id`.
  ProcessId id_;

  /// See ProcessGrid. The pointer is used to gain access to allow
  /// synchronization among participating processes.
  ProcessGrid *grid_;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_PROCESS_H
