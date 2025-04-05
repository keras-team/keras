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

#ifndef STABLEHLO_REFERENCE_PROCESSGRID_H
#define STABLEHLO_REFERENCE_PROCESSGRID_H

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <utility>

#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

struct ProcessId;

/// Represents a result of a `ProcessGrid::rendezvous` where multiple processes
/// synchronize at a barrier and contribute same number of Tensors.
/// This class is pretty much a map from ProcessId to set of Tensors, with the
/// map-like API.
class RendezvousResult {
 public:
  RendezvousResult() = default;
  RendezvousResult(std::map<ProcessId, SmallVector<Tensor>> const &results);

  /// Iterates through the (ProcessId, SmallVector<Tensor>) map entires and
  /// returns a vector of Tensors sorted by ProcessId--(replicaId, partitionId)
  /// pair--in lexicographical order.
  SmallVector<SmallVector<Tensor>> getSortedTensors() const;

  /// Inserts `SmallVector<tensor>` into the map using the key `processId`.
  void insert(ProcessId processId, SmallVector<Tensor> tensor);

  /// Iterates through the map and returns the value associated with the key
  /// `processId`. If key is not found, return an empty `SmallVector<Tensor>`.
  SmallVector<Tensor> lookup(ProcessId processId) const;

  /// Iterates through the (ProcessId, SmallVector<Tensor>) map entires and
  /// return true if all processes contributed same number of operand Tensors
  bool hasMatchingOperandsCount() const;

 private:
  /// Internal map representation of the result of `ProcessGrid::rendezvous`.
  std::map<ProcessId, SmallVector<Tensor>> results_;
};

namespace detail {

/// Internal storage used in `rendezvous` to manage concurrent access to the
/// shared resource. Processes contribute their data to `values` concurrently.
/// Once all processes have added their data, the data in `values` is moved to
/// `result` that multiple processes can concurrently read from.
struct RendezvousState {
  /// Synchronization primitive used to manage concurrent access to this
  /// object.
  std::mutex mutex;

  /// Internal storage used to store data contributed by the processes.
  std::map<ProcessId, SmallVector<Tensor>> values;

  /// Internal state management counter which counts the number of processes
  /// that contributed already.
  size_t useCount;

  /// Stores the result of `rendezvous`.
  RendezvousResult result;
};

struct SendRecvState {
  /// Synchronization primitive used to manage concurrent access to this
  /// object.
  std::mutex mutex;
  /// Internal storage used to store data contributed by the processes.
  SmallVector<Tensor> result;
};

/// Stores the result of `rendezvous` represented as a map that allows
/// concurrent access.
/// Each call to `rendezvous`, i.e. each combination `processGroup` and
/// `channelId`, has its own key in the map. Within the implementation of
/// `rendezvous`, the value corresponding to this key is gradually populated
/// with tensors arriving from different processes in the process group.
template <typename K, typename V>
class ThreadSafeMap {
 public:
  /// Returns a reference to the data associated with the `key`.
  V &operator[](const K &key);

 private:
  /// Synchronization primitive used to manage concurrent access to the map.
  std::mutex lock_;
  /// Internal storage used to implement `rendezvous`.
  std::map<K, V> map_;
};

/// Internal set that manages concurrent access to implement `send` and
/// `recv`.
template <typename T>
class ThreadSafeSet {
 public:
  /// Returns whether the element `value` exists in the set.
  bool contains(T value);

  /// Remove `value` from the set.
  void erase(T value);

  /// Add `value` to the set.
  void insert(T value);

 private:
  /// Synchronization primitive used to manage concurrent access to the set.
  std::mutex lock_;

  /// Internal storage used to manage `send` and `recv` order.
  std::set<T> set_;
};

/// StableHLO `infeed` and `outfeed` represented as a queue that allows
/// concurrent access.
template <typename T>
class ThreadSafeQueue {
 public:
  /// \name Constructors
  /// @{
  ThreadSafeQueue() = default;
  ThreadSafeQueue(const std::queue<T> &queue);
  /// @}

  /// Remove the first element of the queue and return it.
  T pop();

  /// Add `inputs` to the end of the queue.
  void push(T inputs);

 private:
  /// Synchronization primitive used to manage concurrent access to the queue.
  std::mutex lock_;

  /// Internal storage used to implement StableHLO `infeed` and `outfeed`.
  std::queue<T> queue_;
};

}  // namespace detail

using ChannelId = int64_t;

/// StableHLO `process_id`.
struct ProcessId {
  /// StableHLO `replica_id`.
  uint32_t replicaId;

  /// StableHLO `partition_id`.
  uint32_t partitionId;

  /// Overloaded inequality operator.
  bool operator!=(const ProcessId &other) const;

  /// The sort order for ProcessId is not defined in StableHLO, and it's
  /// internally used in ProcessGrid::rendezvous as part of a sorted key on the
  /// map. This operator is conveniently used to help define the ordering since
  /// ordering is defined for StableHLO process group.
  bool operator<(const ProcessId &other) const;

  /// Overloaded equality operator.
  bool operator==(const ProcessId &other) const;
};

/// StableHLO `process_group`.
class ProcessGroup : public SmallVector<ProcessId> {};

/// StableHLO `process_groups`.
class ProcessGroups : public SmallVector<ProcessGroup> {
 public:
  /// Iterates through the ProcessGroups and finds the first ProcessGroup
  /// containing the `processId`. If the group is not found, std::nullopt is
  /// returned.
  std::optional<ProcessGroup> findGroup(ProcessId processId);
};

/// StableHLO process grid.
class ProcessGrid {
 public:
  /// \name Constructors
  /// @{
  ProcessGrid(uint32_t numReplicas, uint32_t numPartitions,
              std::queue<StringAttr> &infeed);
  /// @}

  /// StableHLO `cross_partition` communication strategy.
  ProcessGroups crossPartition(
      SmallVector<SmallVector<uint32_t>> partitionGroups);

  /// StableHLO `cross_replica` communication strategy.
  ProcessGroups crossReplica(SmallVector<SmallVector<uint32_t>> replicaGroups);

  /// StableHLO `cross_replica_and_partition` communication strategy.
  ProcessGroups crossReplicaAndPartition(
      SmallVector<SmallVector<uint32_t>> replicaGroups);

  /// StableHLO `flattened_ids` communication strategy.
  ProcessGroups flattenedIds(
      SmallVector<SmallVector<uint32_t>> flattenedIdGroups);

  /// Retrieves input strings from StableHLO `infeed`.
  StringAttr infeed();

  /// Inserts `inputs` to StableHLO `outfeed`.
  void outfeed(ArrayRef<Tensor> inputs);

  /// Receives data from a channel with `channelId` and returns the data.
  /// `recv` has to be called first before `send` to indicate to the sending
  /// process that the receiver is ready to receive data. The process then waits
  /// until there is data in the channel. The data in the channel with
  /// `channelId` is returned.
  SmallVector<Tensor> recv(ChannelId channelId, ProcessId processId);

  /// Synchronize a StableHLO process with the `processId` with other StableHLO
  /// processes in the `processGroup` using a `channelId`.
  ///
  /// A call to this method represents a barrier, i.e. it blocks the calling
  /// OS thread until all StableHLO processes from the `processGroup` call this
  /// method with the same `channelId`. If the calling OS thread doesn't
  /// correspond to the StableHLO process with `processId`, the behavior is
  /// undefined.
  ///
  /// If any of the StableHLO processes from `processGroup` fail to arrive
  /// at the barrier within 3 seconds, the `rendezvous` fails with a fatal
  /// error for all calling OS threads. This is to make sure that errors in
  /// underlying StableHLO programs or bugs in the StableHLO interpreter don't
  /// deadlock the interpreter.
  ///
  /// At the barrier, each StableHLO process contribute any number of tensors,
  /// and these tensors are accumulated in `RendezvousResult` whose shared
  /// pointer is returned to all callers once the barrier has been reached by
  /// all StableHLO processes.
  RendezvousResult rendezvous(ProcessGroup processGroup, ChannelId channelId,
                              ProcessId processId, ArrayRef<Tensor> operands);

  /// Sends `inputs` to a channel with `channelId`.
  /// The channel with `channelId` is emptied before the receiving process can
  /// receive values. If there are multiple processes sending data to a
  /// duplciate `channelId`, the behavior is undefined.
  void send(ArrayRef<Tensor> inputs, ChannelId channelId, ProcessId processId);

 private:
  /// StableHLO `num_replicas`.
  const uint32_t numReplicas_;

  /// StableHLO `num_partitions`.
  const uint32_t numPartitions_;

  /// Internal queue of strings which represents `func::FuncOp` mnemonic that
  /// returns a vector of Tensor. The function name is stored instead of the
  /// vector of tensors to save memory. See `ThreadSafeQueue`.
  detail::ThreadSafeQueue<StringAttr> infeed_;

  /// Internal queue of vector of Tensor which represents `inputs` stored in
  /// StableHLO `outfeed`. See `ThreadSafeQueue`.
  detail::ThreadSafeQueue<SmallVector<Tensor>> outfeed_;

  /// Internal storage used to implement `send` and `recv`.
  /// `send` can write its data to the channel with ChannelId once the ops are
  /// ready to communicate. `recv` receives data from the same channel once the
  /// data is ready to read.
  detail::ThreadSafeMap<ChannelId, detail::SendRecvState> sendRecvChannels_;

  /// Synchronization primitive used to manage concurrent access to
  /// `sendRecvChannels_`.
  std::map<ChannelId, std::condition_variable> sendRecvConditions_;

  /// Synchronization primitive used to signal send and recv operations are
  /// ready to communicate.
  /// The presence of a ChannelId in the set indicates that the receiving
  /// process is ready to receive data using this ChannelId from the sender
  /// process.
  detail::ThreadSafeSet<ChannelId> sendRecvReady_;

  /// See `ThreadSafeMap`.
  detail::ThreadSafeMap<std::pair<ProcessGroup, ChannelId>,
                        detail::RendezvousState>
      channels_;

  /// Synchronization primitive used to manage concurrent access to `channels_`.
  detail::ThreadSafeMap<std::pair<ProcessGroup, ChannelId>,
                        std::condition_variable>
      channelConditions_;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_PROCESSGRID_H
