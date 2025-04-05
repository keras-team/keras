// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_DATA_FLOW_OPS_H_
#define TENSORFLOW_CC_OPS_DATA_FLOW_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup data_flow_ops Data Flow Ops
/// @{

/// Applies a gradient to a given accumulator.
///
/// Does not add if local_step is lesser than the accumulator's global_step.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a accumulator.
/// * local_step: The local_step value at which the gradient was computed.
/// * gradient: A tensor of the gradient to be accumulated.
///
/// Returns:
/// * the created `Operation`
class AccumulatorApplyGradient {
 public:
  AccumulatorApplyGradient(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         handle, ::tensorflow::Input local_step,
                         ::tensorflow::Input gradient);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Returns the number of gradients aggregated in the given accumulators.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to an accumulator.
///
/// Returns:
/// * `Output`: The number of gradients aggregated in the given accumulator.
class AccumulatorNumAccumulated {
 public:
  AccumulatorNumAccumulated(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          handle);
  operator ::tensorflow::Output() const { return num_accumulated; }
  operator ::tensorflow::Input() const { return num_accumulated; }
  ::tensorflow::Node* node() const { return num_accumulated.node(); }

  Operation operation;
  ::tensorflow::Output num_accumulated;
};

/// Updates the accumulator with a new value for global_step.
///
/// Logs warning if the accumulator's value is already higher than
/// new_global_step.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to an accumulator.
/// * new_global_step: The new global_step value to set.
///
/// Returns:
/// * the created `Operation`
class AccumulatorSetGlobalStep {
 public:
  AccumulatorSetGlobalStep(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         handle, ::tensorflow::Input new_global_step);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Extracts the average gradient in the given ConditionalAccumulator.
///
/// The op blocks until sufficient (i.e., more than num_required)
/// gradients have been accumulated.  If the accumulator has already
/// aggregated more than num_required gradients, it returns the average of
/// the accumulated gradients.  Also automatically increments the recorded
/// global_step in the accumulator by 1, and resets the aggregate to 0.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to an accumulator.
/// * num_required: Number of gradients required before we return an aggregate.
/// * dtype: The data type of accumulated gradients. Needs to correspond to the type
/// of the accumulator.
///
/// Returns:
/// * `Output`: The average of the accumulated gradients.
class AccumulatorTakeGradient {
 public:
  AccumulatorTakeGradient(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        handle, ::tensorflow::Input num_required, DataType
                        dtype);
  operator ::tensorflow::Output() const { return average; }
  operator ::tensorflow::Input() const { return average; }
  ::tensorflow::Node* node() const { return average.node(); }

  Operation operation;
  ::tensorflow::Output average;
};

/// Defines a barrier that persists across different graph executions.
///
/// A barrier represents a key-value map, where each key is a string, and
/// each value is a tuple of tensors.
///
/// At runtime, the barrier contains 'complete' and 'incomplete'
/// elements. A complete element has defined tensors for all components of
/// its value tuple, and may be accessed using BarrierTakeMany. An
/// incomplete element has some undefined components in its value tuple,
/// and may be updated using BarrierInsertMany.
///
/// Args:
/// * scope: A Scope object
/// * component_types: The type of each component in a value.
///
/// Optional attributes (see `Attrs`):
/// * shapes: The shape of each component in a value. Each shape must be 1 in the
/// first dimension. The length of this attr must be the same as the length of
/// component_types.
/// * capacity: The capacity of the barrier.  The default capacity is MAX_INT32,
/// which is the largest capacity of the underlying queue.
/// * container: If non-empty, this barrier is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this barrier will be shared under the given name
/// across multiple sessions.
///
/// Returns:
/// * `Output`: The handle to the barrier.
class Barrier {
 public:
  /// Optional attribute setters for Barrier
  struct Attrs {
    /// The shape of each component in a value. Each shape must be 1 in the
    /// first dimension. The length of this attr must be the same as the length of
    /// component_types.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs Shapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.shapes_ = x;
      return ret;
    }

    /// The capacity of the barrier.  The default capacity is MAX_INT32,
    /// which is the largest capacity of the underlying queue.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// If non-empty, this barrier is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this barrier will be shared under the given name
    /// across multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    gtl::ArraySlice<PartialTensorShape> shapes_ = {};
    int64 capacity_ = -1;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  Barrier(const ::tensorflow::Scope& scope, const DataTypeSlice& component_types);
  Barrier(const ::tensorflow::Scope& scope, const DataTypeSlice& component_types,
        const Barrier::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Shapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().Shapes(x);
  }
  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Closes the given barrier.
///
/// This operation signals that no more new elements will be inserted in the
/// given barrier. Subsequent InsertMany that try to introduce a new key will fail.
/// Subsequent InsertMany operations that just add missing components to already
/// existing elements will continue to succeed. Subsequent TakeMany operations will
/// continue to succeed if sufficient completed elements remain in the barrier.
/// Subsequent TakeMany operations that would block will fail immediately.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a barrier.
///
/// Optional attributes (see `Attrs`):
/// * cancel_pending_enqueues: If true, all pending enqueue requests that are
/// blocked on the barrier's queue will be canceled. InsertMany will fail, even
/// if no new key is introduced.
///
/// Returns:
/// * the created `Operation`
class BarrierClose {
 public:
  /// Optional attribute setters for BarrierClose
  struct Attrs {
    /// If true, all pending enqueue requests that are
    /// blocked on the barrier's queue will be canceled. InsertMany will fail, even
    /// if no new key is introduced.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs CancelPendingEnqueues(bool x) {
      Attrs ret = *this;
      ret.cancel_pending_enqueues_ = x;
      return ret;
    }

    bool cancel_pending_enqueues_ = false;
  };
  BarrierClose(const ::tensorflow::Scope& scope, ::tensorflow::Input handle);
  BarrierClose(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
             const BarrierClose::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs CancelPendingEnqueues(bool x) {
    return Attrs().CancelPendingEnqueues(x);
  }

  Operation operation;
};

/// Computes the number of incomplete elements in the given barrier.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a barrier.
///
/// Returns:
/// * `Output`: The number of incomplete elements (i.e. those with some of their value
/// components not set) in the barrier.
class BarrierIncompleteSize {
 public:
  BarrierIncompleteSize(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      handle);
  operator ::tensorflow::Output() const { return size; }
  operator ::tensorflow::Input() const { return size; }
  ::tensorflow::Node* node() const { return size.node(); }

  Operation operation;
  ::tensorflow::Output size;
};

/// For each key, assigns the respective value to the specified component.
///
/// If a key is not found in the barrier, this operation will create a new
/// incomplete element. If a key is found in the barrier, and the element
/// already has a value at component_index, this operation will fail with
/// INVALID_ARGUMENT, and leave the barrier in an undefined state.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a barrier.
/// * keys: A one-dimensional tensor of keys, with length n.
/// * values: An any-dimensional tensor of values, which are associated with the
/// respective keys. The 0th dimension must have length n.
/// * component_index: The component of the barrier elements that is being assigned.
///
/// Returns:
/// * the created `Operation`
class BarrierInsertMany {
 public:
  BarrierInsertMany(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                  ::tensorflow::Input keys, ::tensorflow::Input values, int64
                  component_index);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Computes the number of complete elements in the given barrier.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a barrier.
///
/// Returns:
/// * `Output`: The number of complete elements (i.e. those with all of their value
/// components set) in the barrier.
class BarrierReadySize {
 public:
  BarrierReadySize(const ::tensorflow::Scope& scope, ::tensorflow::Input handle);
  operator ::tensorflow::Output() const { return size; }
  operator ::tensorflow::Input() const { return size; }
  ::tensorflow::Node* node() const { return size.node(); }

  Operation operation;
  ::tensorflow::Output size;
};

/// Takes the given number of completed elements from a barrier.
///
/// This operation concatenates completed-element component tensors along
/// the 0th dimension to make a single component tensor.
///
/// Elements come out of the barrier when they are complete, and in the order
/// in which they were placed into the barrier.  The indices output provides
/// information about the batch in which each element was originally inserted
/// into the barrier.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a barrier.
/// * num_elements: A single-element tensor containing the number of elements to
/// take.
/// * component_types: The type of each component in a value.
///
/// Optional attributes (see `Attrs`):
/// * allow_small_batch: Allow to return less than num_elements items if barrier is
/// already closed.
/// * timeout_ms: If the queue is empty, this operation will block for up to
/// timeout_ms milliseconds.
/// Note: This option is not supported yet.
///
/// Returns:
/// * `Output` indices: A one-dimensional tensor of indices, with length num_elems.
/// These indices refer to the batch in which the values were placed into the
/// barrier (starting with MIN_LONG and increasing with each BarrierInsertMany).
/// * `Output` keys: A one-dimensional tensor of keys, with length num_elements.
/// * `OutputList` values: One any-dimensional tensor per component in a barrier element. All
/// values have length num_elements in the 0th dimension.
class BarrierTakeMany {
 public:
  /// Optional attribute setters for BarrierTakeMany
  struct Attrs {
    /// Allow to return less than num_elements items if barrier is
    /// already closed.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs AllowSmallBatch(bool x) {
      Attrs ret = *this;
      ret.allow_small_batch_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs WaitForIncomplete(bool x) {
      Attrs ret = *this;
      ret.wait_for_incomplete_ = x;
      return ret;
    }

    /// If the queue is empty, this operation will block for up to
    /// timeout_ms milliseconds.
    /// Note: This option is not supported yet.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs TimeoutMs(int64 x) {
      Attrs ret = *this;
      ret.timeout_ms_ = x;
      return ret;
    }

    bool allow_small_batch_ = false;
    bool wait_for_incomplete_ = false;
    int64 timeout_ms_ = -1;
  };
  BarrierTakeMany(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                ::tensorflow::Input num_elements, const DataTypeSlice&
                component_types);
  BarrierTakeMany(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                ::tensorflow::Input num_elements, const DataTypeSlice&
                component_types, const BarrierTakeMany::Attrs& attrs);

  static Attrs AllowSmallBatch(bool x) {
    return Attrs().AllowSmallBatch(x);
  }
  static Attrs WaitForIncomplete(bool x) {
    return Attrs().WaitForIncomplete(x);
  }
  static Attrs TimeoutMs(int64 x) {
    return Attrs().TimeoutMs(x);
  }

  Operation operation;
  ::tensorflow::Output indices;
  ::tensorflow::Output keys;
  ::tensorflow::OutputList values;
};

/// A conditional accumulator for aggregating gradients.
///
/// The accumulator accepts gradients marked with local_step greater or
/// equal to the most recent global_step known to the accumulator. The
/// average can be extracted from the accumulator, provided sufficient
/// gradients have been accumulated. Extracting the average automatically
/// resets the aggregate to 0, and increments the global_step recorded by
/// the accumulator.
///
/// Args:
/// * scope: A Scope object
/// * dtype: The type of the value being accumulated.
/// * shape: The shape of the values, can be [], in which case shape is unknown.
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this accumulator is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this accumulator will be shared under the
/// given name across multiple sessions.
///
/// Returns:
/// * `Output`: The handle to the accumulator.
class ConditionalAccumulator {
 public:
  /// Optional attribute setters for ConditionalAccumulator
  struct Attrs {
    /// If non-empty, this accumulator is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this accumulator will be shared under the
    /// given name across multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    /// Defaults to "MEAN"
    TF_MUST_USE_RESULT Attrs ReductionType(StringPiece x) {
      Attrs ret = *this;
      ret.reduction_type_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
    StringPiece reduction_type_ = "MEAN";
  };
  ConditionalAccumulator(const ::tensorflow::Scope& scope, DataType dtype,
                       PartialTensorShape shape);
  ConditionalAccumulator(const ::tensorflow::Scope& scope, DataType dtype,
                       PartialTensorShape shape, const
                       ConditionalAccumulator::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }
  static Attrs ReductionType(StringPiece x) {
    return Attrs().ReductionType(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Delete the tensor specified by its handle in the session.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle for a tensor stored in the session state.
///
/// Returns:
/// * the created `Operation`
class DeleteSessionTensor {
 public:
  DeleteSessionTensor(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    handle);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Partitions `data` into `num_partitions` tensors using indices from `partitions`.
///
/// For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
/// becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
/// are placed in `outputs[i]` in lexicographic order of `js`, and the first
/// dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
/// In detail,
///
/// ```python
///     outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]
///
///     outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
/// ```
///
/// `data.shape` must start with `partitions.shape`.
///
/// For example:
///
/// ```python
///     # Scalar partitions.
///     partitions = 1
///     num_partitions = 2
///     data = [10, 20]
///     outputs[0] = []  # Empty with shape [0, 2]
///     outputs[1] = [[10, 20]]
///
///     # Vector partitions.
///     partitions = [0, 0, 1, 1, 0]
///     num_partitions = 2
///     data = [10, 20, 30, 40, 50]
///     outputs[0] = [10, 20, 50]
///     outputs[1] = [30, 40]
/// ```
///
/// See `dynamic_stitch` for an example on how to merge partitions back.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicPartition.png" alt>
/// </div>
///
///
/// Raises:
///   * `InvalidArgumentError` in following cases:
///     - If partitions is not in range `[0, num_partiions)`
///     - If `partitions.shape` does not match prefix of `data.shape` argument.
///
///
/// Args:
/// * scope: A Scope object
/// * partitions: Any shape.  Indices in the range `[0, num_partitions)`.
/// * num_partitions: The number of partitions to output.
///
/// Returns:
/// * `OutputList`: The outputs tensor.
class DynamicPartition {
 public:
  DynamicPartition(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
                 ::tensorflow::Input partitions, int64 num_partitions);
  ::tensorflow::Output operator[](size_t index) const { return outputs[index]; }


  Operation operation;
  ::tensorflow::OutputList outputs;
};

/// Interleave the values from the `data` tensors into a single tensor.
///
/// Builds a merged tensor such that
///
/// ```python
///     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
/// ```
///
/// For example, if each `indices[m]` is scalar or vector, we have
///
/// ```python
///     # Scalar indices:
///     merged[indices[m], ...] = data[m][...]
///
///     # Vector indices:
///     merged[indices[m][i], ...] = data[m][i, ...]
/// ```
///
/// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
/// and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
/// must have `data[i].shape = indices[i].shape + constant`.  In terms of this
/// `constant`, the output shape is
///
///     merged.shape = [max(indices) + 1] + constant
///
/// Values are merged in order, so if an index appears in both `indices[m][i]` and
/// `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
/// merged result. If you do not need this guarantee, ParallelDynamicStitch might
/// perform better on some devices.
///
/// For example:
///
/// ```python
///     indices[0] = 6
///     indices[1] = [4, 1]
///     indices[2] = [[5, 2], [0, 3]]
///     data[0] = [61, 62]
///     data[1] = [[41, 42], [11, 12]]
///     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
///     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
///               [51, 52], [61, 62]]
/// ```
///
/// This method can be used to merge partitions created by `dynamic_partition`
/// as illustrated on the following example:
///
/// ```python
///     # Apply function (increments x_i) on elements for which a certain condition
///     # apply (x_i != -1 in this example).
///     x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
///     condition_mask=tf.not_equal(x,tf.constant(-1.))
///     partitioned_data = tf.dynamic_partition(
///         x, tf.cast(condition_mask, tf.int32) , 2)
///     partitioned_data[1] = partitioned_data[1] + 1.0
///     condition_indices = tf.dynamic_partition(
///         tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
///     x = tf.dynamic_stitch(condition_indices, partitioned_data)
///     # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
///     # unchanged.
/// ```
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
/// </div>
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The merged tensor.
class DynamicStitch {
 public:
  DynamicStitch(const ::tensorflow::Scope& scope, ::tensorflow::InputList
              indices, ::tensorflow::InputList data);
  operator ::tensorflow::Output() const { return merged; }
  operator ::tensorflow::Input() const { return merged; }
  ::tensorflow::Node* node() const { return merged.node(); }

  Operation operation;
  ::tensorflow::Output merged;
};

/// A queue that produces elements in first-in first-out order.
///
/// Args:
/// * scope: A Scope object
/// * component_types: The type of each component in a value.
///
/// Optional attributes (see `Attrs`):
/// * shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types. If the length of
/// this attr is 0, the shapes of queue elements are not constrained, and
/// only one element may be dequeued at a time.
/// * capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// * container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
///
/// Returns:
/// * `Output`: The handle to the queue.
class FIFOQueue {
 public:
  /// Optional attribute setters for FIFOQueue
  struct Attrs {
    /// The shape of each component in a value. The length of this attr must
    /// be either 0 or the same as the length of component_types. If the length of
    /// this attr is 0, the shapes of queue elements are not constrained, and
    /// only one element may be dequeued at a time.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs Shapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.shapes_ = x;
      return ret;
    }

    /// The upper bound on the number of elements in this queue.
    /// Negative numbers mean no limit.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// If non-empty, this queue is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this queue will be shared under the given name
    /// across multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    gtl::ArraySlice<PartialTensorShape> shapes_ = {};
    int64 capacity_ = -1;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  FIFOQueue(const ::tensorflow::Scope& scope, const DataTypeSlice&
          component_types);
  FIFOQueue(const ::tensorflow::Scope& scope, const DataTypeSlice&
          component_types, const FIFOQueue::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Shapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().Shapes(x);
  }
  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Store the input tensor in the state of the current session.
///
/// Args:
/// * scope: A Scope object
/// * value: The tensor to be stored.
///
/// Returns:
/// * `Output`: The handle for the tensor stored in the session state, represented
/// as a string.
class GetSessionHandle {
 public:
  GetSessionHandle(const ::tensorflow::Scope& scope, ::tensorflow::Input value);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Store the input tensor in the state of the current session.
///
/// Args:
/// * scope: A Scope object
/// * value: The tensor to be stored.
///
/// Returns:
/// * `Output`: The handle for the tensor stored in the session state, represented
/// as a ResourceHandle object.
class GetSessionHandleV2 {
 public:
  GetSessionHandleV2(const ::tensorflow::Scope& scope, ::tensorflow::Input value);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Get the value of the tensor specified by its handle.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle for a tensor stored in the session state.
/// * dtype: The type of the output value.
///
/// Returns:
/// * `Output`: The tensor for the given handle.
class GetSessionTensor {
 public:
  GetSessionTensor(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                 DataType dtype);
  operator ::tensorflow::Output() const { return value; }
  operator ::tensorflow::Input() const { return value; }
  ::tensorflow::Node* node() const { return value.node(); }

  Operation operation;
  ::tensorflow::Output value;
};

/// Op removes all elements in the underlying container.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class MapClear {
 public:
  /// Optional attribute setters for MapClear
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  MapClear(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes);
  MapClear(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes, const
         MapClear::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
};

/// Op returns the number of incomplete elements in the underlying container.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The size tensor.
class MapIncompleteSize {
 public:
  /// Optional attribute setters for MapIncompleteSize
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  MapIncompleteSize(const ::tensorflow::Scope& scope, const DataTypeSlice&
                  dtypes);
  MapIncompleteSize(const ::tensorflow::Scope& scope, const DataTypeSlice&
                  dtypes, const MapIncompleteSize::Attrs& attrs);
  operator ::tensorflow::Output() const { return size; }
  operator ::tensorflow::Input() const { return size; }
  ::tensorflow::Node* node() const { return size.node(); }

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output size;
};

/// Op peeks at the values at the specified key.  If the
///
/// underlying container does not contain this key
/// this op will block until it does.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The values tensor.
class MapPeek {
 public:
  /// Optional attribute setters for MapPeek
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  MapPeek(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
        ::tensorflow::Input indices, const DataTypeSlice& dtypes);
  MapPeek(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
        ::tensorflow::Input indices, const DataTypeSlice& dtypes, const
        MapPeek::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return values[index]; }


  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::OutputList values;
};

/// Op returns the number of elements in the underlying container.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The size tensor.
class MapSize {
 public:
  /// Optional attribute setters for MapSize
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  MapSize(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes);
  MapSize(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes, const
        MapSize::Attrs& attrs);
  operator ::tensorflow::Output() const { return size; }
  operator ::tensorflow::Input() const { return size; }
  ::tensorflow::Node* node() const { return size.node(); }

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output size;
};

/// Stage (key, values) in the underlying container which behaves like a hashtable.
///
/// Args:
/// * scope: A Scope object
/// * key: int64
/// * values: a list of tensors
/// dtypes A list of data types that inserted values should adhere to.
///
/// Optional attributes (see `Attrs`):
/// * capacity: Maximum number of elements in the Staging Area. If > 0, inserts
/// on the container will block when the capacity is reached.
/// * container: If non-empty, this queue is placed in the given container. Otherwise,
/// a default container is used.
/// * shared_name: It is necessary to match this name to the matching Unstage Op.
///
/// Returns:
/// * the created `Operation`
class MapStage {
 public:
  /// Optional attribute setters for MapStage
  struct Attrs {
    /// Maximum number of elements in the Staging Area. If > 0, inserts
    /// on the container will block when the capacity is reached.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// If non-empty, this queue is placed in the given container. Otherwise,
    /// a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// It is necessary to match this name to the matching Unstage Op.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  MapStage(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
         ::tensorflow::Input indices, ::tensorflow::InputList values, const
         DataTypeSlice& dtypes);
  MapStage(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
         ::tensorflow::Input indices, ::tensorflow::InputList values, const
         DataTypeSlice& dtypes, const MapStage::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
};

/// Op removes and returns the values associated with the key
///
/// from the underlying container.   If the underlying container
/// does not contain this key, the op will block until it does.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The values tensor.
class MapUnstage {
 public:
  /// Optional attribute setters for MapUnstage
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  MapUnstage(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
           ::tensorflow::Input indices, const DataTypeSlice& dtypes);
  MapUnstage(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
           ::tensorflow::Input indices, const DataTypeSlice& dtypes, const
           MapUnstage::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return values[index]; }


  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::OutputList values;
};

/// Op removes and returns a random (key, value)
///
/// from the underlying container.   If the underlying container
/// does not contain elements, the op will block until it does.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` key
/// * `OutputList` values
class MapUnstageNoKey {
 public:
  /// Optional attribute setters for MapUnstageNoKey
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  MapUnstageNoKey(const ::tensorflow::Scope& scope, ::tensorflow::Input indices,
                const DataTypeSlice& dtypes);
  MapUnstageNoKey(const ::tensorflow::Scope& scope, ::tensorflow::Input indices,
                const DataTypeSlice& dtypes, const MapUnstageNoKey::Attrs&
                attrs);

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output key;
  ::tensorflow::OutputList values;
};

/// Op removes all elements in the underlying container.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class OrderedMapClear {
 public:
  /// Optional attribute setters for OrderedMapClear
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  OrderedMapClear(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes);
  OrderedMapClear(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes,
                const OrderedMapClear::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
};

/// Op returns the number of incomplete elements in the underlying container.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The size tensor.
class OrderedMapIncompleteSize {
 public:
  /// Optional attribute setters for OrderedMapIncompleteSize
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  OrderedMapIncompleteSize(const ::tensorflow::Scope& scope, const DataTypeSlice&
                         dtypes);
  OrderedMapIncompleteSize(const ::tensorflow::Scope& scope, const DataTypeSlice&
                         dtypes, const OrderedMapIncompleteSize::Attrs& attrs);
  operator ::tensorflow::Output() const { return size; }
  operator ::tensorflow::Input() const { return size; }
  ::tensorflow::Node* node() const { return size.node(); }

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output size;
};

/// Op peeks at the values at the specified key.  If the
///
/// underlying container does not contain this key
/// this op will block until it does.   This Op is optimized for
/// performance.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The values tensor.
class OrderedMapPeek {
 public:
  /// Optional attribute setters for OrderedMapPeek
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  OrderedMapPeek(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
               ::tensorflow::Input indices, const DataTypeSlice& dtypes);
  OrderedMapPeek(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
               ::tensorflow::Input indices, const DataTypeSlice& dtypes, const
               OrderedMapPeek::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return values[index]; }


  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::OutputList values;
};

/// Op returns the number of elements in the underlying container.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The size tensor.
class OrderedMapSize {
 public:
  /// Optional attribute setters for OrderedMapSize
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  OrderedMapSize(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes);
  OrderedMapSize(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes,
               const OrderedMapSize::Attrs& attrs);
  operator ::tensorflow::Output() const { return size; }
  operator ::tensorflow::Input() const { return size; }
  ::tensorflow::Node* node() const { return size.node(); }

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output size;
};

/// Stage (key, values) in the underlying container which behaves like a ordered
///
/// associative container.   Elements are ordered by key.
///
/// Args:
/// * scope: A Scope object
/// * key: int64
/// * values: a list of tensors
/// dtypes A list of data types that inserted values should adhere to.
///
/// Optional attributes (see `Attrs`):
/// * capacity: Maximum number of elements in the Staging Area. If > 0, inserts
/// on the container will block when the capacity is reached.
/// * container: If non-empty, this queue is placed in the given container. Otherwise,
/// a default container is used.
/// * shared_name: It is necessary to match this name to the matching Unstage Op.
///
/// Returns:
/// * the created `Operation`
class OrderedMapStage {
 public:
  /// Optional attribute setters for OrderedMapStage
  struct Attrs {
    /// Maximum number of elements in the Staging Area. If > 0, inserts
    /// on the container will block when the capacity is reached.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// If non-empty, this queue is placed in the given container. Otherwise,
    /// a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// It is necessary to match this name to the matching Unstage Op.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  OrderedMapStage(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
                ::tensorflow::Input indices, ::tensorflow::InputList values,
                const DataTypeSlice& dtypes);
  OrderedMapStage(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
                ::tensorflow::Input indices, ::tensorflow::InputList values,
                const DataTypeSlice& dtypes, const OrderedMapStage::Attrs&
                attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
};

/// Op removes and returns the values associated with the key
///
/// from the underlying container.   If the underlying container
/// does not contain this key, the op will block until it does.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The values tensor.
class OrderedMapUnstage {
 public:
  /// Optional attribute setters for OrderedMapUnstage
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  OrderedMapUnstage(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
                  ::tensorflow::Input indices, const DataTypeSlice& dtypes);
  OrderedMapUnstage(const ::tensorflow::Scope& scope, ::tensorflow::Input key,
                  ::tensorflow::Input indices, const DataTypeSlice& dtypes,
                  const OrderedMapUnstage::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return values[index]; }


  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::OutputList values;
};

/// Op removes and returns the (key, value) element with the smallest
///
/// key from the underlying container.   If the underlying container
/// does not contain elements, the op will block until it does.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` key
/// * `OutputList` values
class OrderedMapUnstageNoKey {
 public:
  /// Optional attribute setters for OrderedMapUnstageNoKey
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  OrderedMapUnstageNoKey(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       indices, const DataTypeSlice& dtypes);
  OrderedMapUnstageNoKey(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       indices, const DataTypeSlice& dtypes, const
                       OrderedMapUnstageNoKey::Attrs& attrs);

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output key;
  ::tensorflow::OutputList values;
};

/// A queue that produces elements in first-in first-out order.
///
/// Variable-size shapes are allowed by setting the corresponding shape dimensions
/// to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
/// size of any given element in the minibatch.  See below for details.
///
/// Args:
/// * scope: A Scope object
/// * component_types: The type of each component in a value.
///
/// Optional attributes (see `Attrs`):
/// * shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types.
/// Shapes of fixed rank but variable size are allowed by setting
/// any shape dimension to -1.  In this case, the inputs' shape may vary along
/// the given dimension, and DequeueMany will pad the given dimension with
/// zeros up to the maximum shape of all elements in the given batch.
/// If the length of this attr is 0, different queue elements may have
/// different ranks and shapes, but only one element may be dequeued at a time.
/// * capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// * container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
///
/// Returns:
/// * `Output`: The handle to the queue.
class PaddingFIFOQueue {
 public:
  /// Optional attribute setters for PaddingFIFOQueue
  struct Attrs {
    /// The shape of each component in a value. The length of this attr must
    /// be either 0 or the same as the length of component_types.
    /// Shapes of fixed rank but variable size are allowed by setting
    /// any shape dimension to -1.  In this case, the inputs' shape may vary along
    /// the given dimension, and DequeueMany will pad the given dimension with
    /// zeros up to the maximum shape of all elements in the given batch.
    /// If the length of this attr is 0, different queue elements may have
    /// different ranks and shapes, but only one element may be dequeued at a time.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs Shapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.shapes_ = x;
      return ret;
    }

    /// The upper bound on the number of elements in this queue.
    /// Negative numbers mean no limit.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// If non-empty, this queue is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this queue will be shared under the given name
    /// across multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    gtl::ArraySlice<PartialTensorShape> shapes_ = {};
    int64 capacity_ = -1;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  PaddingFIFOQueue(const ::tensorflow::Scope& scope, const DataTypeSlice&
                 component_types);
  PaddingFIFOQueue(const ::tensorflow::Scope& scope, const DataTypeSlice&
                 component_types, const PaddingFIFOQueue::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Shapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().Shapes(x);
  }
  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Interleave the values from the `data` tensors into a single tensor.
///
/// Builds a merged tensor such that
///
/// ```python
///     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
/// ```
///
/// For example, if each `indices[m]` is scalar or vector, we have
///
/// ```python
///     # Scalar indices:
///     merged[indices[m], ...] = data[m][...]
///
///     # Vector indices:
///     merged[indices[m][i], ...] = data[m][i, ...]
/// ```
///
/// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
/// and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
/// must have `data[i].shape = indices[i].shape + constant`.  In terms of this
/// `constant`, the output shape is
///
///     merged.shape = [max(indices)] + constant
///
/// Values may be merged in parallel, so if an index appears in both `indices[m][i]`
/// and `indices[n][j]`, the result may be invalid. This differs from the normal
/// DynamicStitch operator that defines the behavior in that case.
///
/// For example:
///
/// ```python
///     indices[0] = 6
///     indices[1] = [4, 1]
///     indices[2] = [[5, 2], [0, 3]]
///     data[0] = [61, 62]
///     data[1] = [[41, 42], [11, 12]]
///     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
///     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
///               [51, 52], [61, 62]]
/// ```
///
/// This method can be used to merge partitions created by `dynamic_partition`
/// as illustrated on the following example:
///
/// ```python
///     # Apply function (increments x_i) on elements for which a certain condition
///     # apply (x_i != -1 in this example).
///     x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
///     condition_mask=tf.not_equal(x,tf.constant(-1.))
///     partitioned_data = tf.dynamic_partition(
///         x, tf.cast(condition_mask, tf.int32) , 2)
///     partitioned_data[1] = partitioned_data[1] + 1.0
///     condition_indices = tf.dynamic_partition(
///         tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
///     x = tf.dynamic_stitch(condition_indices, partitioned_data)
///     # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
///     # unchanged.
/// ```
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
/// </div>
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The merged tensor.
class ParallelDynamicStitch {
 public:
  ParallelDynamicStitch(const ::tensorflow::Scope& scope, ::tensorflow::InputList
                      indices, ::tensorflow::InputList data);
  operator ::tensorflow::Output() const { return merged; }
  operator ::tensorflow::Input() const { return merged; }
  ::tensorflow::Node* node() const { return merged.node(); }

  Operation operation;
  ::tensorflow::Output merged;
};

/// A queue that produces elements sorted by the first component value.
///
/// Note that the PriorityQueue requires the first component of any element
/// to be a scalar int64, in addition to the other elements declared by
/// component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
/// and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
/// entry in their input (resp. output) lists.
///
/// Args:
/// * scope: A Scope object
/// * shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types. If the length of
/// this attr is 0, the shapes of queue elements are not constrained, and
/// only one element may be dequeued at a time.
///
/// Optional attributes (see `Attrs`):
/// * component_types: The type of each component in a value.
/// * capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// * container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
///
/// Returns:
/// * `Output`: The handle to the queue.
class PriorityQueue {
 public:
  /// Optional attribute setters for PriorityQueue
  struct Attrs {
    /// The type of each component in a value.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs ComponentTypes(const DataTypeSlice& x) {
      Attrs ret = *this;
      ret.component_types_ = x;
      return ret;
    }

    /// The upper bound on the number of elements in this queue.
    /// Negative numbers mean no limit.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// If non-empty, this queue is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this queue will be shared under the given name
    /// across multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    DataTypeSlice component_types_ = {};
    int64 capacity_ = -1;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  PriorityQueue(const ::tensorflow::Scope& scope, const
              gtl::ArraySlice<PartialTensorShape>& shapes);
  PriorityQueue(const ::tensorflow::Scope& scope, const
              gtl::ArraySlice<PartialTensorShape>& shapes, const
              PriorityQueue::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs ComponentTypes(const DataTypeSlice& x) {
    return Attrs().ComponentTypes(x);
  }
  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Closes the given queue.
///
/// This operation signals that no more elements will be enqueued in the
/// given queue. Subsequent Enqueue(Many) operations will fail.
/// Subsequent Dequeue(Many) operations will continue to succeed if
/// sufficient elements remain in the queue. Subsequent Dequeue(Many)
/// operations that would block will fail immediately.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a queue.
///
/// Optional attributes (see `Attrs`):
/// * cancel_pending_enqueues: If true, all pending enqueue requests that are
/// blocked on the given queue will be canceled.
///
/// Returns:
/// * the created `Operation`
class QueueClose {
 public:
  /// Optional attribute setters for QueueClose
  struct Attrs {
    /// If true, all pending enqueue requests that are
    /// blocked on the given queue will be canceled.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs CancelPendingEnqueues(bool x) {
      Attrs ret = *this;
      ret.cancel_pending_enqueues_ = x;
      return ret;
    }

    bool cancel_pending_enqueues_ = false;
  };
  QueueClose(const ::tensorflow::Scope& scope, ::tensorflow::Input handle);
  QueueClose(const ::tensorflow::Scope& scope, ::tensorflow::Input handle, const
           QueueClose::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs CancelPendingEnqueues(bool x) {
    return Attrs().CancelPendingEnqueues(x);
  }

  Operation operation;
};

/// Dequeues `n` tuples of one or more tensors from the given queue.
///
/// If the queue is closed and there are fewer than `n` elements, then an
/// OutOfRange error is returned.
///
/// This operation concatenates queue-element component tensors along the
/// 0th dimension to make a single component tensor.  All of the components
/// in the dequeued tuple will have size `n` in the 0th dimension.
///
/// This operation has `k` outputs, where `k` is the number of components in
/// the tuples stored in the given queue, and output `i` is the ith
/// component of the dequeued tuple.
///
/// N.B. If the queue is empty, this operation will block until `n` elements
/// have been dequeued (or 'timeout_ms' elapses, if specified).
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a queue.
/// * n: The number of tuples to dequeue.
/// * component_types: The type of each component in a tuple.
///
/// Optional attributes (see `Attrs`):
/// * timeout_ms: If the queue has fewer than n elements, this operation
/// will block for up to timeout_ms milliseconds.
/// Note: This option is not supported yet.
///
/// Returns:
/// * `OutputList`: One or more tensors that were dequeued as a tuple.
class QueueDequeueMany {
 public:
  /// Optional attribute setters for QueueDequeueMany
  struct Attrs {
    /// If the queue has fewer than n elements, this operation
    /// will block for up to timeout_ms milliseconds.
    /// Note: This option is not supported yet.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs TimeoutMs(int64 x) {
      Attrs ret = *this;
      ret.timeout_ms_ = x;
      return ret;
    }

    int64 timeout_ms_ = -1;
  };
  QueueDequeueMany(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                 ::tensorflow::Input n, const DataTypeSlice& component_types);
  QueueDequeueMany(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                 ::tensorflow::Input n, const DataTypeSlice& component_types,
                 const QueueDequeueMany::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return components[index]; }


  static Attrs TimeoutMs(int64 x) {
    return Attrs().TimeoutMs(x);
  }

  Operation operation;
  ::tensorflow::OutputList components;
};

/// Dequeues `n` tuples of one or more tensors from the given queue.
///
/// This operation is not supported by all queues.  If a queue does not support
/// DequeueUpTo, then an Unimplemented error is returned.
///
/// If the queue is closed and there are more than 0 but less than `n`
/// elements remaining, then instead of returning an OutOfRange error like
/// QueueDequeueMany, less than `n` elements are returned immediately.  If
/// the queue is closed and there are 0 elements left in the queue, then
/// an OutOfRange error is returned just like in QueueDequeueMany.
/// Otherwise the behavior is identical to QueueDequeueMany:
///
/// This operation concatenates queue-element component tensors along the
/// 0th dimension to make a single component tensor.  All of the components
/// in the dequeued tuple will have size n in the 0th dimension.
///
/// This operation has `k` outputs, where `k` is the number of components in
/// the tuples stored in the given queue, and output `i` is the ith
/// component of the dequeued tuple.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a queue.
/// * n: The number of tuples to dequeue.
/// * component_types: The type of each component in a tuple.
///
/// Optional attributes (see `Attrs`):
/// * timeout_ms: If the queue has fewer than n elements, this operation
/// will block for up to timeout_ms milliseconds.
/// Note: This option is not supported yet.
///
/// Returns:
/// * `OutputList`: One or more tensors that were dequeued as a tuple.
class QueueDequeueUpTo {
 public:
  /// Optional attribute setters for QueueDequeueUpTo
  struct Attrs {
    /// If the queue has fewer than n elements, this operation
    /// will block for up to timeout_ms milliseconds.
    /// Note: This option is not supported yet.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs TimeoutMs(int64 x) {
      Attrs ret = *this;
      ret.timeout_ms_ = x;
      return ret;
    }

    int64 timeout_ms_ = -1;
  };
  QueueDequeueUpTo(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                 ::tensorflow::Input n, const DataTypeSlice& component_types);
  QueueDequeueUpTo(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                 ::tensorflow::Input n, const DataTypeSlice& component_types,
                 const QueueDequeueUpTo::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return components[index]; }


  static Attrs TimeoutMs(int64 x) {
    return Attrs().TimeoutMs(x);
  }

  Operation operation;
  ::tensorflow::OutputList components;
};

/// Dequeues a tuple of one or more tensors from the given queue.
///
/// This operation has k outputs, where k is the number of components
/// in the tuples stored in the given queue, and output i is the ith
/// component of the dequeued tuple.
///
/// N.B. If the queue is empty, this operation will block until an element
/// has been dequeued (or 'timeout_ms' elapses, if specified).
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a queue.
/// * component_types: The type of each component in a tuple.
///
/// Optional attributes (see `Attrs`):
/// * timeout_ms: If the queue is empty, this operation will block for up to
/// timeout_ms milliseconds.
/// Note: This option is not supported yet.
///
/// Returns:
/// * `OutputList`: One or more tensors that were dequeued as a tuple.
class QueueDequeue {
 public:
  /// Optional attribute setters for QueueDequeue
  struct Attrs {
    /// If the queue is empty, this operation will block for up to
    /// timeout_ms milliseconds.
    /// Note: This option is not supported yet.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs TimeoutMs(int64 x) {
      Attrs ret = *this;
      ret.timeout_ms_ = x;
      return ret;
    }

    int64 timeout_ms_ = -1;
  };
  QueueDequeue(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
             const DataTypeSlice& component_types);
  QueueDequeue(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
             const DataTypeSlice& component_types, const QueueDequeue::Attrs&
             attrs);
  ::tensorflow::Output operator[](size_t index) const { return components[index]; }


  static Attrs TimeoutMs(int64 x) {
    return Attrs().TimeoutMs(x);
  }

  Operation operation;
  ::tensorflow::OutputList components;
};

/// Enqueues zero or more tuples of one or more tensors in the given queue.
///
/// This operation slices each component tensor along the 0th dimension to
/// make multiple queue elements. All of the tuple components must have the
/// same size in the 0th dimension.
///
/// The components input has k elements, which correspond to the components of
/// tuples stored in the given queue.
///
/// N.B. If the queue is full, this operation will block until the given
/// elements have been enqueued (or 'timeout_ms' elapses, if specified).
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a queue.
/// * components: One or more tensors from which the enqueued tensors should
/// be taken.
///
/// Optional attributes (see `Attrs`):
/// * timeout_ms: If the queue is too full, this operation will block for up
/// to timeout_ms milliseconds.
/// Note: This option is not supported yet.
///
/// Returns:
/// * the created `Operation`
class QueueEnqueueMany {
 public:
  /// Optional attribute setters for QueueEnqueueMany
  struct Attrs {
    /// If the queue is too full, this operation will block for up
    /// to timeout_ms milliseconds.
    /// Note: This option is not supported yet.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs TimeoutMs(int64 x) {
      Attrs ret = *this;
      ret.timeout_ms_ = x;
      return ret;
    }

    int64 timeout_ms_ = -1;
  };
  QueueEnqueueMany(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                 ::tensorflow::InputList components);
  QueueEnqueueMany(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                 ::tensorflow::InputList components, const
                 QueueEnqueueMany::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs TimeoutMs(int64 x) {
    return Attrs().TimeoutMs(x);
  }

  Operation operation;
};

/// Enqueues a tuple of one or more tensors in the given queue.
///
/// The components input has k elements, which correspond to the components of
/// tuples stored in the given queue.
///
/// N.B. If the queue is full, this operation will block until the given
/// element has been enqueued (or 'timeout_ms' elapses, if specified).
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a queue.
/// * components: One or more tensors from which the enqueued tensors should be taken.
///
/// Optional attributes (see `Attrs`):
/// * timeout_ms: If the queue is full, this operation will block for up to
/// timeout_ms milliseconds.
/// Note: This option is not supported yet.
///
/// Returns:
/// * the created `Operation`
class QueueEnqueue {
 public:
  /// Optional attribute setters for QueueEnqueue
  struct Attrs {
    /// If the queue is full, this operation will block for up to
    /// timeout_ms milliseconds.
    /// Note: This option is not supported yet.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs TimeoutMs(int64 x) {
      Attrs ret = *this;
      ret.timeout_ms_ = x;
      return ret;
    }

    int64 timeout_ms_ = -1;
  };
  QueueEnqueue(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
             ::tensorflow::InputList components);
  QueueEnqueue(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
             ::tensorflow::InputList components, const QueueEnqueue::Attrs&
             attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs TimeoutMs(int64 x) {
    return Attrs().TimeoutMs(x);
  }

  Operation operation;
};

/// Returns true if queue is closed.
///
/// This operation returns true if the queue is closed and false if the queue
/// is open.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a queue.
///
/// Returns:
/// * `Output`: The is_closed tensor.
class QueueIsClosed {
 public:
  QueueIsClosed(const ::tensorflow::Scope& scope, ::tensorflow::Input handle);
  operator ::tensorflow::Output() const { return is_closed; }
  operator ::tensorflow::Input() const { return is_closed; }
  ::tensorflow::Node* node() const { return is_closed.node(); }

  Operation operation;
  ::tensorflow::Output is_closed;
};

/// Returns true if queue is closed.
///
/// This operation returns true if the queue is closed and false if the queue
/// is open.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a queue.
///
/// Returns:
/// * `Output`: The is_closed tensor.
class QueueIsClosedV2 {
 public:
  QueueIsClosedV2(const ::tensorflow::Scope& scope, ::tensorflow::Input handle);
  operator ::tensorflow::Output() const { return is_closed; }
  operator ::tensorflow::Input() const { return is_closed; }
  ::tensorflow::Node* node() const { return is_closed.node(); }

  Operation operation;
  ::tensorflow::Output is_closed;
};

/// Computes the number of elements in the given queue.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a queue.
///
/// Returns:
/// * `Output`: The number of elements in the given queue.
class QueueSize {
 public:
  QueueSize(const ::tensorflow::Scope& scope, ::tensorflow::Input handle);
  operator ::tensorflow::Output() const { return size; }
  operator ::tensorflow::Input() const { return size; }
  ::tensorflow::Node* node() const { return size.node(); }

  Operation operation;
  ::tensorflow::Output size;
};

/// A queue that randomizes the order of elements.
///
/// Args:
/// * scope: A Scope object
/// * component_types: The type of each component in a value.
///
/// Optional attributes (see `Attrs`):
/// * shapes: The shape of each component in a value. The length of this attr must
/// be either 0 or the same as the length of component_types. If the length of
/// this attr is 0, the shapes of queue elements are not constrained, and
/// only one element may be dequeued at a time.
/// * capacity: The upper bound on the number of elements in this queue.
/// Negative numbers mean no limit.
/// * min_after_dequeue: Dequeue will block unless there would be this
/// many elements after the dequeue or the queue is closed. This
/// ensures a minimum level of mixing of elements.
/// * seed: If either seed or seed2 is set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, a random seed is used.
/// * seed2: A second seed to avoid seed collision.
/// * container: If non-empty, this queue is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this queue will be shared under the given name
/// across multiple sessions.
///
/// Returns:
/// * `Output`: The handle to the queue.
class RandomShuffleQueue {
 public:
  /// Optional attribute setters for RandomShuffleQueue
  struct Attrs {
    /// The shape of each component in a value. The length of this attr must
    /// be either 0 or the same as the length of component_types. If the length of
    /// this attr is 0, the shapes of queue elements are not constrained, and
    /// only one element may be dequeued at a time.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs Shapes(const gtl::ArraySlice<PartialTensorShape>& x) {
      Attrs ret = *this;
      ret.shapes_ = x;
      return ret;
    }

    /// The upper bound on the number of elements in this queue.
    /// Negative numbers mean no limit.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Dequeue will block unless there would be this
    /// many elements after the dequeue or the queue is closed. This
    /// ensures a minimum level of mixing of elements.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MinAfterDequeue(int64 x) {
      Attrs ret = *this;
      ret.min_after_dequeue_ = x;
      return ret;
    }

    /// If either seed or seed2 is set to be non-zero, the random number
    /// generator is seeded by the given seed.  Otherwise, a random seed is used.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed(int64 x) {
      Attrs ret = *this;
      ret.seed_ = x;
      return ret;
    }

    /// A second seed to avoid seed collision.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed2(int64 x) {
      Attrs ret = *this;
      ret.seed2_ = x;
      return ret;
    }

    /// If non-empty, this queue is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this queue will be shared under the given name
    /// across multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    gtl::ArraySlice<PartialTensorShape> shapes_ = {};
    int64 capacity_ = -1;
    int64 min_after_dequeue_ = 0;
    int64 seed_ = 0;
    int64 seed2_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  RandomShuffleQueue(const ::tensorflow::Scope& scope, const DataTypeSlice&
                   component_types);
  RandomShuffleQueue(const ::tensorflow::Scope& scope, const DataTypeSlice&
                   component_types, const RandomShuffleQueue::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Shapes(const gtl::ArraySlice<PartialTensorShape>& x) {
    return Attrs().Shapes(x);
  }
  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MinAfterDequeue(int64 x) {
    return Attrs().MinAfterDequeue(x);
  }
  static Attrs Seed(int64 x) {
    return Attrs().Seed(x);
  }
  static Attrs Seed2(int64 x) {
    return Attrs().Seed2(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Emits randomized records.
///
/// Args:
/// * scope: A Scope object
/// * file_pattern: Glob pattern for the data files.
///
/// Optional attributes (see `Attrs`):
/// * file_random_seed: Random seeds used to produce randomized records.
/// * file_shuffle_shift_ratio: Shifts the list of files after the list is randomly
/// shuffled.
/// * file_buffer_size: The randomization shuffling buffer.
/// * file_parallelism: How many sstables are opened and concurrently iterated over.
/// * batch_size: The batch size.
/// * compression_type: The type of compression for the file. Currently ZLIB and
/// GZIP are supported. Defaults to none.
///
/// Returns:
/// * `Output`: A tensor of shape [batch_size].
class RecordInput {
 public:
  /// Optional attribute setters for RecordInput
  struct Attrs {
    /// Random seeds used to produce randomized records.
    ///
    /// Defaults to 301
    TF_MUST_USE_RESULT Attrs FileRandomSeed(int64 x) {
      Attrs ret = *this;
      ret.file_random_seed_ = x;
      return ret;
    }

    /// Shifts the list of files after the list is randomly
    /// shuffled.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs FileShuffleShiftRatio(float x) {
      Attrs ret = *this;
      ret.file_shuffle_shift_ratio_ = x;
      return ret;
    }

    /// The randomization shuffling buffer.
    ///
    /// Defaults to 10000
    TF_MUST_USE_RESULT Attrs FileBufferSize(int64 x) {
      Attrs ret = *this;
      ret.file_buffer_size_ = x;
      return ret;
    }

    /// How many sstables are opened and concurrently iterated over.
    ///
    /// Defaults to 16
    TF_MUST_USE_RESULT Attrs FileParallelism(int64 x) {
      Attrs ret = *this;
      ret.file_parallelism_ = x;
      return ret;
    }

    /// The batch size.
    ///
    /// Defaults to 32
    TF_MUST_USE_RESULT Attrs BatchSize(int64 x) {
      Attrs ret = *this;
      ret.batch_size_ = x;
      return ret;
    }

    /// The type of compression for the file. Currently ZLIB and
    /// GZIP are supported. Defaults to none.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs CompressionType(StringPiece x) {
      Attrs ret = *this;
      ret.compression_type_ = x;
      return ret;
    }

    int64 file_random_seed_ = 301;
    float file_shuffle_shift_ratio_ = 0.0f;
    int64 file_buffer_size_ = 10000;
    int64 file_parallelism_ = 16;
    int64 batch_size_ = 32;
    StringPiece compression_type_ = "";
  };
  RecordInput(const ::tensorflow::Scope& scope, StringPiece file_pattern);
  RecordInput(const ::tensorflow::Scope& scope, StringPiece file_pattern, const
            RecordInput::Attrs& attrs);
  operator ::tensorflow::Output() const { return records; }
  operator ::tensorflow::Input() const { return records; }
  ::tensorflow::Node* node() const { return records.node(); }

  static Attrs FileRandomSeed(int64 x) {
    return Attrs().FileRandomSeed(x);
  }
  static Attrs FileShuffleShiftRatio(float x) {
    return Attrs().FileShuffleShiftRatio(x);
  }
  static Attrs FileBufferSize(int64 x) {
    return Attrs().FileBufferSize(x);
  }
  static Attrs FileParallelism(int64 x) {
    return Attrs().FileParallelism(x);
  }
  static Attrs BatchSize(int64 x) {
    return Attrs().BatchSize(x);
  }
  static Attrs CompressionType(StringPiece x) {
    return Attrs().CompressionType(x);
  }

  Operation operation;
  ::tensorflow::Output records;
};

/// Applies a sparse gradient to a given accumulator.
///
/// Does not add if local_step is smaller than the accumulator's
/// global_step.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a accumulator.
/// * local_step: The local_step value at which the sparse gradient was computed.
/// * gradient_indices: Indices of the sparse gradient to be accumulated. Must be a
/// vector.
/// * gradient_values: Values are the non-zero slices of the gradient, and must have
/// the same first dimension as indices, i.e., the nnz represented by indices and
/// values must be consistent.
/// * gradient_shape: Shape of the sparse gradient to be accumulated.
/// * has_known_shape: Boolean indicating whether gradient_shape is unknown, in which
/// case the input is ignored during validation.
///
/// Returns:
/// * the created `Operation`
class SparseAccumulatorApplyGradient {
 public:
  SparseAccumulatorApplyGradient(const ::tensorflow::Scope& scope,
                               ::tensorflow::Input handle, ::tensorflow::Input
                               local_step, ::tensorflow::Input
                               gradient_indices, ::tensorflow::Input
                               gradient_values, ::tensorflow::Input
                               gradient_shape, bool has_known_shape);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Extracts the average sparse gradient in a SparseConditionalAccumulator.
///
/// The op will blocks until sufficient (i.e., more than num_required)
/// gradients have been accumulated. If the accumulator has already
/// aggregated more than num_required gradients, it will return its
/// average of the accumulated gradients.  Also automatically increments
/// the recorded global_step in the accumulator by 1, and resets the
/// aggregate to 0.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a SparseConditionalAccumulator.
/// * num_required: Number of gradients required before we return an aggregate.
/// * dtype: The data type of accumulated gradients. Needs to correspond to the type
/// of the accumulator.
///
/// Returns:
/// * `Output` indices: Indices of the average of the accumulated sparse gradients.
/// * `Output` values: Values of the average of the accumulated sparse gradients.
/// * `Output` shape: Shape of the average of the accumulated sparse gradients.
class SparseAccumulatorTakeGradient {
 public:
  SparseAccumulatorTakeGradient(const ::tensorflow::Scope& scope,
                              ::tensorflow::Input handle, ::tensorflow::Input
                              num_required, DataType dtype);

  Operation operation;
  ::tensorflow::Output indices;
  ::tensorflow::Output values;
  ::tensorflow::Output shape;
};

/// A conditional accumulator for aggregating sparse gradients.
///
/// The accumulator accepts gradients marked with local_step greater or
/// equal to the most recent global_step known to the accumulator. The
/// average can be extracted from the accumulator, provided sufficient
/// gradients have been accumulated. Extracting the average automatically
/// resets the aggregate to 0, and increments the global_step recorded by
/// the accumulator.
///
/// Args:
/// * scope: A Scope object
/// * dtype: The type of the value being accumulated.
/// * shape: The shape of the values.
///
/// Optional attributes (see `Attrs`):
/// * container: If non-empty, this accumulator is placed in the given container.
/// Otherwise, a default container is used.
/// * shared_name: If non-empty, this accumulator will be shared under the given name
/// across multiple sessions.
///
/// Returns:
/// * `Output`: The handle to the accumulator.
class SparseConditionalAccumulator {
 public:
  /// Optional attribute setters for SparseConditionalAccumulator
  struct Attrs {
    /// If non-empty, this accumulator is placed in the given container.
    /// Otherwise, a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// If non-empty, this accumulator will be shared under the given name
    /// across multiple sessions.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    /// Defaults to "MEAN"
    TF_MUST_USE_RESULT Attrs ReductionType(StringPiece x) {
      Attrs ret = *this;
      ret.reduction_type_ = x;
      return ret;
    }

    StringPiece container_ = "";
    StringPiece shared_name_ = "";
    StringPiece reduction_type_ = "MEAN";
  };
  SparseConditionalAccumulator(const ::tensorflow::Scope& scope, DataType dtype,
                             PartialTensorShape shape);
  SparseConditionalAccumulator(const ::tensorflow::Scope& scope, DataType dtype,
                             PartialTensorShape shape, const
                             SparseConditionalAccumulator::Attrs& attrs);
  operator ::tensorflow::Output() const { return handle; }
  operator ::tensorflow::Input() const { return handle; }
  ::tensorflow::Node* node() const { return handle.node(); }

  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }
  static Attrs ReductionType(StringPiece x) {
    return Attrs().ReductionType(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
};

/// Stage values similar to a lightweight Enqueue.
///
/// The basic functionality of this Op is similar to a queue with many
/// fewer capabilities and options.  This Op is optimized for performance.
///
/// Args:
/// * scope: A Scope object
/// * values: a list of tensors
/// dtypes A list of data types that inserted values should adhere to.
///
/// Optional attributes (see `Attrs`):
/// * capacity: Maximum number of elements in the Staging Area. If > 0, inserts
/// on the container will block when the capacity is reached.
/// * memory_limit: The maximum number of bytes allowed for Tensors in the Staging Area.
/// If > 0, inserts will block until sufficient space is available.
/// * container: If non-empty, this queue is placed in the given container. Otherwise,
/// a default container is used.
/// * shared_name: It is necessary to match this name to the matching Unstage Op.
///
/// Returns:
/// * the created `Operation`
class Stage {
 public:
  /// Optional attribute setters for Stage
  struct Attrs {
    /// Maximum number of elements in the Staging Area. If > 0, inserts
    /// on the container will block when the capacity is reached.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// The maximum number of bytes allowed for Tensors in the Staging Area.
    /// If > 0, inserts will block until sufficient space is available.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// If non-empty, this queue is placed in the given container. Otherwise,
    /// a default container is used.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// It is necessary to match this name to the matching Unstage Op.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  Stage(const ::tensorflow::Scope& scope, ::tensorflow::InputList values);
  Stage(const ::tensorflow::Scope& scope, ::tensorflow::InputList values, const
      Stage::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
};

/// Op removes all elements in the underlying container.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class StageClear {
 public:
  /// Optional attribute setters for StageClear
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  StageClear(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes);
  StageClear(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes, const
           StageClear::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
};

/// Op peeks at the values at the specified index.  If the
///
/// underlying container does not contain sufficient elements
/// this op will block until it does.   This Op is optimized for
/// performance.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The values tensor.
class StagePeek {
 public:
  /// Optional attribute setters for StagePeek
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  StagePeek(const ::tensorflow::Scope& scope, ::tensorflow::Input index, const
          DataTypeSlice& dtypes);
  StagePeek(const ::tensorflow::Scope& scope, ::tensorflow::Input index, const
          DataTypeSlice& dtypes, const StagePeek::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return values[index]; }


  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::OutputList values;
};

/// Op returns the number of elements in the underlying container.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The size tensor.
class StageSize {
 public:
  /// Optional attribute setters for StageSize
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  StageSize(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes);
  StageSize(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes, const
          StageSize::Attrs& attrs);
  operator ::tensorflow::Output() const { return size; }
  operator ::tensorflow::Input() const { return size; }
  ::tensorflow::Node* node() const { return size.node(); }

  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::Output size;
};

/// Delete the TensorArray from its resource container.
///
/// This enables the user to close and release the resource in the middle
/// of a step/run.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
///
/// Returns:
/// * the created `Operation`
class TensorArrayClose {
 public:
  TensorArrayClose(const ::tensorflow::Scope& scope, ::tensorflow::Input handle);
  operator ::tensorflow::Operation() const { return operation; }

  Operation operation;
};

/// Concat the elements from the TensorArray into value `value`.
///
/// Takes `T` elements of shapes
///
///   ```
///   (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
///   ```
///
/// and concatenates them into a Tensor of shape:
///
///   ```
///   (n0 + n1 + ... + n(T-1) x d0 x d1 x ...)
///   ```
///
/// All elements must have the same shape (excepting the first dimension).
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a TensorArray.
/// * flow_in: A float scalar that enforces proper chaining of operations.
/// * dtype: The type of the elem that is returned.
///
/// Optional attributes (see `Attrs`):
/// * element_shape_except0: The expected shape of an element, if known,
/// excluding the first dimension. Used to validate the shapes of
/// TensorArray elements. If this shape is not fully specified, concatenating
/// zero-size TensorArrays is an error.
///
/// Returns:
/// * `Output` value: All of the elements in the TensorArray, concatenated along the first
/// axis.
/// * `Output` lengths: A vector of the row sizes of the original T elements in the
/// value output.  In the example above, this would be the values:
/// `(n1, n2, ..., n(T-1))`.
class TensorArrayConcat {
 public:
  /// Optional attribute setters for TensorArrayConcat
  struct Attrs {
    /// The expected shape of an element, if known,
    /// excluding the first dimension. Used to validate the shapes of
    /// TensorArray elements. If this shape is not fully specified, concatenating
    /// zero-size TensorArrays is an error.
    ///
    /// Defaults to <unknown>
    TF_MUST_USE_RESULT Attrs ElementShapeExcept0(PartialTensorShape x) {
      Attrs ret = *this;
      ret.element_shape_except0_ = x;
      return ret;
    }

    PartialTensorShape element_shape_except0_ = ::tensorflow::PartialTensorShape() /* unknown */;
  };
  TensorArrayConcat(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                  ::tensorflow::Input flow_in, DataType dtype);
  TensorArrayConcat(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                  ::tensorflow::Input flow_in, DataType dtype, const
                  TensorArrayConcat::Attrs& attrs);

  static Attrs ElementShapeExcept0(PartialTensorShape x) {
    return Attrs().ElementShapeExcept0(x);
  }

  Operation operation;
  ::tensorflow::Output value;
  ::tensorflow::Output lengths;
};

/// Gather specific elements from the TensorArray into output `value`.
///
/// All elements selected by `indices` must have the same shape.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a TensorArray.
/// * indices: The locations in the TensorArray from which to read tensor elements.
/// * flow_in: A float scalar that enforces proper chaining of operations.
/// * dtype: The type of the elem that is returned.
///
/// Optional attributes (see `Attrs`):
/// * element_shape: The expected shape of an element, if known. Used to
/// validate the shapes of TensorArray elements. If this shape is not
/// fully specified, gathering zero-size TensorArrays is an error.
///
/// Returns:
/// * `Output`: All of the elements in the TensorArray, concatenated along a new
/// axis (the new dimension 0).
class TensorArrayGather {
 public:
  /// Optional attribute setters for TensorArrayGather
  struct Attrs {
    /// The expected shape of an element, if known. Used to
    /// validate the shapes of TensorArray elements. If this shape is not
    /// fully specified, gathering zero-size TensorArrays is an error.
    ///
    /// Defaults to <unknown>
    TF_MUST_USE_RESULT Attrs ElementShape(PartialTensorShape x) {
      Attrs ret = *this;
      ret.element_shape_ = x;
      return ret;
    }

    PartialTensorShape element_shape_ = ::tensorflow::PartialTensorShape() /* unknown */;
  };
  TensorArrayGather(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                  ::tensorflow::Input indices, ::tensorflow::Input flow_in,
                  DataType dtype);
  TensorArrayGather(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                  ::tensorflow::Input indices, ::tensorflow::Input flow_in,
                  DataType dtype, const TensorArrayGather::Attrs& attrs);
  operator ::tensorflow::Output() const { return value; }
  operator ::tensorflow::Input() const { return value; }
  ::tensorflow::Node* node() const { return value.node(); }

  static Attrs ElementShape(PartialTensorShape x) {
    return Attrs().ElementShape(x);
  }

  Operation operation;
  ::tensorflow::Output value;
};

/// Creates a TensorArray for storing the gradients of values in the given handle.
///
/// If the given TensorArray gradient already exists, returns a reference to it.
///
/// Locks the size of the original TensorArray by disabling its dynamic size flag.
///
/// **A note about the input flow_in:**
///
/// The handle flow_in forces the execution of the gradient lookup to occur
/// only after certain other operations have occurred.  For example, when
/// the forward TensorArray is dynamically sized, writes to this TensorArray
/// may resize the object.  The gradient TensorArray is statically sized based
/// on the size of the forward TensorArray when this operation executes.
/// Furthermore, the size of the forward TensorArray is frozen by this call.
/// As a result, the flow is used to ensure that the call to generate the gradient
/// TensorArray only happens after all writes are executed.
///
/// In the case of dynamically sized TensorArrays, gradient computation should
/// only be performed on read operations that have themselves been chained via
/// flow to occur only after all writes have executed. That way the final size
/// of the forward TensorArray is known when this operation is called.
///
/// **A note about the source attribute:**
///
/// TensorArray gradient calls use an accumulator TensorArray object.  If
/// multiple gradients are calculated and run in the same session, the multiple
/// gradient nodes may accidentally flow through the same accumulator TensorArray.
/// This double counts and generally breaks the TensorArray gradient flow.
///
/// The solution is to identify which gradient call this particular
/// TensorArray gradient is being called in.  This is performed by identifying
/// a unique string (e.g. "gradients", "gradients_1", ...) from the input
/// gradient Tensor's name.  This string is used as a suffix when creating
/// the TensorArray gradient object here (the attribute `source`).
///
/// The attribute `source` is added as a suffix to the forward TensorArray's
/// name when performing the creation / lookup, so that each separate gradient
/// calculation gets its own TensorArray accumulator.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to the forward TensorArray.
/// * flow_in: A float scalar that enforces proper chaining of operations.
/// * source: The gradient source string, used to decide which gradient TensorArray
/// to return.
///
/// Returns:
/// * `Output` grad_handle
/// * `Output` flow_out
class TensorArrayGrad {
 public:
  TensorArrayGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                ::tensorflow::Input flow_in, StringPiece source);

  Operation operation;
  ::tensorflow::Output grad_handle;
  ::tensorflow::Output flow_out;
};

/// Creates a TensorArray for storing multiple gradients of values in the given handle.
///
/// Similar to TensorArrayGradV3. However it creates an accumulator with an
/// expanded shape compared to the input TensorArray whose gradient is being
/// computed. This enables multiple gradients for the same TensorArray to be
/// calculated using the same accumulator.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to the forward TensorArray.
/// * flow_in: A float scalar that enforces proper chaining of operations.
/// * shape_to_prepend: An int32 vector representing a shape. Elements in the gradient accumulator will
/// have shape which is this shape_to_prepend value concatenated with shape of the
/// elements in the TensorArray corresponding to the input handle.
/// * source: The gradient source string, used to decide which gradient TensorArray
/// to return.
///
/// Returns:
/// * `Output` grad_handle
/// * `Output` flow_out
class TensorArrayGradWithShape {
 public:
  TensorArrayGradWithShape(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         handle, ::tensorflow::Input flow_in,
                         ::tensorflow::Input shape_to_prepend, StringPiece
                         source);

  Operation operation;
  ::tensorflow::Output grad_handle;
  ::tensorflow::Output flow_out;
};

/// Read an element from the TensorArray into output `value`.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a TensorArray.
/// * flow_in: A float scalar that enforces proper chaining of operations.
/// * dtype: The type of the elem that is returned.
///
/// Returns:
/// * `Output`: The tensor that is read from the TensorArray.
class TensorArrayRead {
 public:
  TensorArrayRead(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                ::tensorflow::Input index, ::tensorflow::Input flow_in,
                DataType dtype);
  operator ::tensorflow::Output() const { return value; }
  operator ::tensorflow::Input() const { return value; }
  ::tensorflow::Node* node() const { return value.node(); }

  Operation operation;
  ::tensorflow::Output value;
};

/// Scatter the data from the input value into specific TensorArray elements.
///
/// `indices` must be a vector, its length must match the first dim of `value`.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a TensorArray.
/// * indices: The locations at which to write the tensor elements.
/// * value: The concatenated tensor to write to the TensorArray.
/// * flow_in: A float scalar that enforces proper chaining of operations.
///
/// Returns:
/// * `Output`: A float scalar that enforces proper chaining of operations.
class TensorArrayScatter {
 public:
  TensorArrayScatter(const ::tensorflow::Scope& scope, ::tensorflow::Input
                   handle, ::tensorflow::Input indices, ::tensorflow::Input
                   value, ::tensorflow::Input flow_in);
  operator ::tensorflow::Output() const { return flow_out; }
  operator ::tensorflow::Input() const { return flow_out; }
  ::tensorflow::Node* node() const { return flow_out.node(); }

  Operation operation;
  ::tensorflow::Output flow_out;
};

/// Get the current size of the TensorArray.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
/// * flow_in: A float scalar that enforces proper chaining of operations.
///
/// Returns:
/// * `Output`: The current size of the TensorArray.
class TensorArraySize {
 public:
  TensorArraySize(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                ::tensorflow::Input flow_in);
  operator ::tensorflow::Output() const { return size; }
  operator ::tensorflow::Input() const { return size; }
  ::tensorflow::Node* node() const { return size.node(); }

  Operation operation;
  ::tensorflow::Output size;
};

/// Split the data from the input value into TensorArray elements.
///
/// Assuming that `lengths` takes on values
///
///   ```
///   (n0, n1, ..., n(T-1))
///   ```
///
/// and that `value` has shape
///
///   ```
///   (n0 + n1 + ... + n(T-1) x d0 x d1 x ...),
///   ```
///
/// this splits values into a TensorArray with T tensors.
///
/// TensorArray index t will be the subtensor of values with starting position
///
///   ```
///   (n0 + n1 + ... + n(t-1), 0, 0, ...)
///   ```
///
/// and having size
///
///   ```
///   nt x d0 x d1 x ...
///   ```
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a TensorArray.
/// * value: The concatenated tensor to write to the TensorArray.
/// * lengths: The vector of lengths, how to split the rows of value into the
/// TensorArray.
/// * flow_in: A float scalar that enforces proper chaining of operations.
///
/// Returns:
/// * `Output`: A float scalar that enforces proper chaining of operations.
class TensorArraySplit {
 public:
  TensorArraySplit(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                 ::tensorflow::Input value, ::tensorflow::Input lengths,
                 ::tensorflow::Input flow_in);
  operator ::tensorflow::Output() const { return flow_out; }
  operator ::tensorflow::Input() const { return flow_out; }
  ::tensorflow::Node* node() const { return flow_out.node(); }

  Operation operation;
  ::tensorflow::Output flow_out;
};

/// An array of Tensors of given size.
///
/// Write data via Write and read via Read or Pack.
///
/// Args:
/// * scope: A Scope object
/// * size: The size of the array.
/// * dtype: The type of the elements on the tensor_array.
///
/// Optional attributes (see `Attrs`):
/// * element_shape: The expected shape of an element, if known. Used to
/// validate the shapes of TensorArray elements. If this shape is not
/// fully specified, gathering zero-size TensorArrays is an error.
/// * dynamic_size: A boolean that determines whether writes to the TensorArray
/// are allowed to grow the size.  By default, this is not allowed.
/// * clear_after_read: If true (default), Tensors in the TensorArray are cleared
/// after being read.  This disables multiple read semantics but allows early
/// release of memory.
/// * identical_element_shapes: If true (default is false), then all
/// elements in the TensorArray will be expected to have identical shapes.
/// This allows certain behaviors, like dynamically checking for
/// consistent shapes on write, and being able to fill in properly
/// shaped zero tensors on stack -- even if the element_shape attribute
/// is not fully defined.
/// * tensor_array_name: Overrides the name used for the temporary tensor_array
/// resource. Default value is the name of the 'TensorArray' op (which
/// is guaranteed unique).
///
/// Returns:
/// * `Output` handle: The handle to the TensorArray.
/// * `Output` flow: A scalar used to control gradient flow.
class TensorArray {
 public:
  /// Optional attribute setters for TensorArray
  struct Attrs {
    /// The expected shape of an element, if known. Used to
    /// validate the shapes of TensorArray elements. If this shape is not
    /// fully specified, gathering zero-size TensorArrays is an error.
    ///
    /// Defaults to <unknown>
    TF_MUST_USE_RESULT Attrs ElementShape(PartialTensorShape x) {
      Attrs ret = *this;
      ret.element_shape_ = x;
      return ret;
    }

    /// A boolean that determines whether writes to the TensorArray
    /// are allowed to grow the size.  By default, this is not allowed.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs DynamicSize(bool x) {
      Attrs ret = *this;
      ret.dynamic_size_ = x;
      return ret;
    }

    /// If true (default), Tensors in the TensorArray are cleared
    /// after being read.  This disables multiple read semantics but allows early
    /// release of memory.
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ClearAfterRead(bool x) {
      Attrs ret = *this;
      ret.clear_after_read_ = x;
      return ret;
    }

    /// If true (default is false), then all
    /// elements in the TensorArray will be expected to have identical shapes.
    /// This allows certain behaviors, like dynamically checking for
    /// consistent shapes on write, and being able to fill in properly
    /// shaped zero tensors on stack -- even if the element_shape attribute
    /// is not fully defined.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs IdenticalElementShapes(bool x) {
      Attrs ret = *this;
      ret.identical_element_shapes_ = x;
      return ret;
    }

    /// Overrides the name used for the temporary tensor_array
    /// resource. Default value is the name of the 'TensorArray' op (which
    /// is guaranteed unique).
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs TensorArrayName(StringPiece x) {
      Attrs ret = *this;
      ret.tensor_array_name_ = x;
      return ret;
    }

    PartialTensorShape element_shape_ = ::tensorflow::PartialTensorShape() /* unknown */;
    bool dynamic_size_ = false;
    bool clear_after_read_ = true;
    bool identical_element_shapes_ = false;
    StringPiece tensor_array_name_ = "";
  };
  TensorArray(const ::tensorflow::Scope& scope, ::tensorflow::Input size,
            DataType dtype);
  TensorArray(const ::tensorflow::Scope& scope, ::tensorflow::Input size,
            DataType dtype, const TensorArray::Attrs& attrs);

  static Attrs ElementShape(PartialTensorShape x) {
    return Attrs().ElementShape(x);
  }
  static Attrs DynamicSize(bool x) {
    return Attrs().DynamicSize(x);
  }
  static Attrs ClearAfterRead(bool x) {
    return Attrs().ClearAfterRead(x);
  }
  static Attrs IdenticalElementShapes(bool x) {
    return Attrs().IdenticalElementShapes(x);
  }
  static Attrs TensorArrayName(StringPiece x) {
    return Attrs().TensorArrayName(x);
  }

  Operation operation;
  ::tensorflow::Output handle;
  ::tensorflow::Output flow;
};

/// Push an element onto the tensor_array.
///
/// Args:
/// * scope: A Scope object
/// * handle: The handle to a TensorArray.
/// * index: The position to write to inside the TensorArray.
/// * value: The tensor to write to the TensorArray.
/// * flow_in: A float scalar that enforces proper chaining of operations.
///
/// Returns:
/// * `Output`: A float scalar that enforces proper chaining of operations.
class TensorArrayWrite {
 public:
  TensorArrayWrite(const ::tensorflow::Scope& scope, ::tensorflow::Input handle,
                 ::tensorflow::Input index, ::tensorflow::Input value,
                 ::tensorflow::Input flow_in);
  operator ::tensorflow::Output() const { return flow_out; }
  operator ::tensorflow::Input() const { return flow_out; }
  ::tensorflow::Node* node() const { return flow_out.node(); }

  Operation operation;
  ::tensorflow::Output flow_out;
};

/// Op is similar to a lightweight Dequeue.
///
/// The basic functionality is similar to dequeue with many fewer
/// capabilities and options.  This Op is optimized for performance.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The values tensor.
class Unstage {
 public:
  /// Optional attribute setters for Unstage
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Capacity(int64 x) {
      Attrs ret = *this;
      ret.capacity_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs MemoryLimit(int64 x) {
      Attrs ret = *this;
      ret.memory_limit_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Container(StringPiece x) {
      Attrs ret = *this;
      ret.container_ = x;
      return ret;
    }

    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs SharedName(StringPiece x) {
      Attrs ret = *this;
      ret.shared_name_ = x;
      return ret;
    }

    int64 capacity_ = 0;
    int64 memory_limit_ = 0;
    StringPiece container_ = "";
    StringPiece shared_name_ = "";
  };
  Unstage(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes);
  Unstage(const ::tensorflow::Scope& scope, const DataTypeSlice& dtypes, const
        Unstage::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return values[index]; }


  static Attrs Capacity(int64 x) {
    return Attrs().Capacity(x);
  }
  static Attrs MemoryLimit(int64 x) {
    return Attrs().MemoryLimit(x);
  }
  static Attrs Container(StringPiece x) {
    return Attrs().Container(x);
  }
  static Attrs SharedName(StringPiece x) {
    return Attrs().SharedName(x);
  }

  Operation operation;
  ::tensorflow::OutputList values;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_DATA_FLOW_OPS_H_
