/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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
*******************************************************************************/

#ifndef GRAPH_INTERFACE_PARTITION_IMPL_HPP
#define GRAPH_INTERFACE_PARTITION_IMPL_HPP

#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "common/engine.hpp"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/logical_tensor.hpp"
#include "graph/interface/op.hpp"

#include "graph/utils/id.hpp"
#include "graph/utils/utils.hpp"

#ifdef DNNL_WITH_SYCL
#include "graph/utils/sycl_check.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include <CL/cl.h>
#endif

namespace std {
template <>
struct hash<std::pair<size_t, size_t>> {
    size_t operator()(const std::pair<size_t, size_t> &v) const {
        size_t seed = 0;
        seed ^= std::hash<size_t> {}(v.first) + 0x9e3779b9 + (seed << 6)
                + (seed >> 2);
        seed ^= std::hash<size_t> {}(v.second) + 0x9e3779b9 + (seed << 6)
                + (seed >> 2);
        return seed;
    }
};
} // namespace std

namespace dnnl {
namespace impl {
namespace graph {

class backend_t;

class partition_impl_t : public std::enable_shared_from_this<partition_impl_t> {
public:
    explicit partition_impl_t(engine_kind_t engine_kind,
            fpmath_mode_t fpmath_mode, partition_kind_t pkind)
        : engine_kind_(engine_kind)
        , fpmath_mode_(fpmath_mode)
        , pkind_(pkind)
        , can_use_blocked_layout_(false) {}

    explicit partition_impl_t(engine_kind_t engine_kind,
            fpmath_mode_t fpmath_mode = fpmath_mode::strict)
        : engine_kind_(engine_kind)
        , fpmath_mode_(fpmath_mode)
        , pkind_(partition_kind_t::undef)
        , can_use_blocked_layout_(false) {}

    virtual ~partition_impl_t() = default;

    /// The getter for engine_kind_, which is used in C API
    engine_kind_t get_engine_kind() const { return engine_kind_; }

    /// The getter for fpmath_mode_
    fpmath_mode_t get_fpmath_mode() const { return fpmath_mode_; }

    /// The getter for partition kind
    partition_kind_t get_kind() const { return pkind_; }

    /// The getter for ops_, which is used in C API
    const std::vector<std::shared_ptr<op_t>> &get_ops() const { return ops_; }

    /// The getters for inputs_, which is used in C API
    const std::vector<logical_tensor_t> &get_inputs() const { return inputs_; }

    /// The getters for outputs_, which is used in C API
    const std::vector<logical_tensor_t> &get_outputs() const {
        return outputs_;
    }

    virtual bool is_initialized() const = 0;

    /// Deep copy a partition, and return the copied partition's smart pointer
    /// Derived class must clone the members in base class when implementing
    /// this method.
    virtual std::shared_ptr<partition_impl_t> clone() const = 0;

    /// Return the assigned backend of this partition
    virtual const backend_t *get_assigned_backend() const = 0;

    /// Infer the outputs shape according to the inputs shape and the ops in
    /// this partition.
    /// @param inputs The inputs logical tensors whose shapes are valid
    /// @param outputs The outputs logical tensors whose shapes are
    ///     invalid but will be filled by this function
    /// @return The status code
    /// @note
    ///     The order of the given in/outputs logical tensor may be not same
    ///     with the in/outputs_. Backend should do the reorder inside this
    ///     function's implementation.
    virtual status_t infer_shape(std::vector<const logical_tensor_t *> &inputs,
            std::vector<logical_tensor_t *> &outputs) const = 0;

    /// Compile the partition with specific inputs and outputs logical tensors
    /// and engine. A partition can be compiled multiple times with different
    /// inputs and outputs
    /// @param compiled_partition The pointer of an empty instance, whose
    ///     pimpl_ field should be filled with a smart pointer of internally
    ///     created compiled_partition_impl_t instance during the compile
    /// @param inputs The inputs logical tensors which are fully specified
    ///     id, data type, shape, and layout. For each logical tensor
    ///     in the inputs_, there must be one logical tensor in input whose
    ///     id is exactly same with it
    /// @param outputs The outputs logical tensors, which are fully specified
    ///     id, data type, shape, and layout. For each logical tensor
    ///     in the outputs_, there must be one logical tensor in output whose
    ///     id is exactly same with it
    /// @param aengine The engine that this partition is compiled for. Engine
    ///     contains the device target so the compilation knows what kind of
    ///     binary to generate. The device target doesn’t contain uArch
    ///     information. The backend should access the uArch information by
    ///     querying device runtime.
    /// @return The status code
    /// @note
    ///     1. The order of the given in/outputs logical tensor may be not same
    ///     with the in/outputs_. Backend should reorder the given logical
    ///     tensors by their id in most efficient way
    ///     2. The reordered in/outputs should be passed to the constructor when
    ///     creating the compiled_partition_impl_t instance, for representing
    ///     that the instance is specialized for such logical tensors.
    ///     3. It’s allowed to set outputs arguments’ layout to `any` and let
    ///     backend choose the optimal layouts by itself
    ///     4. The given in/outputs are const, so if backend need to set layout,
    ///     it should modify the copied ones in compiled_partition_impl_t
    ///     5. If the layout is row-major contiguous, the compilation must
    ///     succeed. If the layout is strided layout, it is implementation
    ///     dependent whether the compilation must succeed.
    ///     6. If certain dimension of shape has unknown value, indicated by
    ///     “-1” , it is implementation dependent whether the compilation
    ///     succeed. If it succeed, the compiled partition should be able to
    ///     handle any value for that dimension at the execution time.
    ///     7. If rank has unknown value, indicated by “-1” , it is
    ///     implementation dependent whether the compilation succeed.
    ///     If it succeed, the compiled partition should be able to handle any
    ///     value for input tensor with any rank at the execution time.
    virtual status_t compile(compiled_partition_t *compiled_partition,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs,
            const engine_t *aengine) const = 0;

    /// get partition_impl id
    size_t id() const { return id_; }

    /// set partition_impl id
    void set_id(const size_t id) { id_ = id; }

    /// Used to set the partition can use blocked layout
    virtual void set_use_blocked_layout(bool flag) {
        can_use_blocked_layout_ = flag;
    }

    /// Used to check if a partition can use blocked layout
    virtual bool get_use_blocked_layout() const {
        return can_use_blocked_layout_;
    }

protected:
    // Engine kind
    engine_kind_t engine_kind_;

    // floating-point math mode
    fpmath_mode_t fpmath_mode_;

    // Partition kind
    partition_kind_t pkind_;

    //////////////////////////////////////////////////////
    /// Q: What do the ops_/inputs_/outputs_ represent for?
    /// A: Take the following pattern as an example:
    ///    A    B
    ///     \  /
    ///      conv
    ///       |
    ///       C   D
    ///        \ /
    ///         add
    ///          |
    ///          E
    /// The ops_ are used to store the operators belong to
    /// this partition. They should be able to represent a
    /// subgraph.
    /// The inputs_ and outputs_ represent the edge in the
    /// graph, such as inputs_=[A,B,D] and outputs_=[E].
    /// The inputs_ and outputs_ are logical tensors, they
    /// must contain id/dtype and optionally shape/strides.
    /// Because they are inherited from the graph, and FWK
    /// may not have valid shape infos when creating the
    /// graph
    ///
    /// Q: How to populate the ops_/inputs_/outputs_?
    /// A: Take the above pattern as an example:
    /// The ops_, inputs_ and outputs_ should be added by
    /// each backend during the pattern matching. oneDNN
    /// Graph doesn’t care the order of addition at all.
    /// You can add the ops_ as [conv,add] or [add,conv],
    /// and add the inputs_ as [A,B,D] or [A,D,B] or others.
    /// It’s backend’s responsibility to record the exact
    /// mapping between in/outputs_ and the populated ops’
    /// in/outputs, if they want to know the semantics of
    /// logical tensor
    //////////////////////////////////////////////////////

    /// All the ops belong to this partition.
    std::vector<std::shared_ptr<op_t>> ops_;

    /// All the input logical tensors of a partition
    std::vector<logical_tensor_t> inputs_ {};

    /// All the output logical tensors of a partition
    std::vector<logical_tensor_t> outputs_ {};

    /// Partition_impl id
    size_t id_ = std::numeric_limits<size_t>::max();

    bool can_use_blocked_layout_;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(partition_impl_t);
};

class compiled_partition_impl_t {
public:
    /// The base constructor of compiled_partition_impl_t. The subclass
    /// instance should be created and set to compiled_partition's pimpl
    /// field during compilation
    ///
    /// @param engine The engine which this compiled_partition_impl_t
    ///     is specialized for. Should be equal to the engine that is
    ///     given when calling partition_impl_t::compile
    /// @param inputs The inputs logical tensors which this
    ///     compiled_partition_impl_t is specialized for. Should have
    ///     exact shape/dtype/layout information
    /// @param outputs The outputs logical tensors which this
    ///     compiled_partition_impl_t is specialized for. Should have
    ///     exact shape/dtype/layout information
    /// @param inplace_pairs The inplace pairs that used to indicate
    ///     which input and output tensor given on execute can share
    ///     same memory buffer
    compiled_partition_impl_t(const engine_t &engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs,
            const std::vector<inplace_pair_t> &inplace_pairs)
        : engine_(&engine)
        , inputs_(inputs)
        , outputs_(outputs)
        , inplace_pairs_(inplace_pairs) {};

    virtual ~compiled_partition_impl_t() = default;

    /// The getters for engine_, which is used in C API implementation
    const engine_t *get_engine() const { return engine_; }

    /// The getters for inputs_, which is used in verbose mode
    const std::vector<logical_tensor_t> &get_inputs() const { return inputs_; }

    /// The getters for outputs_, which is used in verbose mode
    const std::vector<logical_tensor_t> &get_outputs() const {
        return outputs_;
    }

    /// The getters for inplace_pairs_, which is used in C API
    const std::vector<inplace_pair_t> &get_inplace_pairs() const {
        return inplace_pairs_;
    }

    /// Query out a specific logical tensor by using an id. This function
    /// is used in C APIThe queried
    /// @param tid The id used to find the required logical tensor
    /// @param lt The address of buffer that is used to store the queried
    ///     logical tensor. Will be zero if not find the required one
    /// @return The status code. Will always be true
    /// @note If we don't find the logical tensor in compiled partition's
    ///     inputs_and outputs_, this means the logical tensor is not used by
    ///     this compiled partition. This will be a common situation if FWK
    ///     gives arbitrary connection, and shouldn't be regarded as an error
    status_t query_logical_tensor(size_t tid, logical_tensor_t *lt) const;

    /// The mutable getter for inputs and outputs
    /// @note After compile, backend may choose opaque layout for in/outputs.
    ///     The opaque layout is represented by the layout_id in logical tensor.
    ///     In order to tell frontend which backend generates this opaque
    ///     layout_id, we need to encode backend id into the layout_id in
    ///     oneDNN Graph. So, we need to get out the mutable reference.
    std::vector<logical_tensor_t> &get_mutable_inputs() { return inputs_; }

    std::vector<logical_tensor_t> &get_mutable_outputs() { return outputs_; }

    /// Execute a compiled_partition with given inputs/outputs tensors
    /// @param astream For different device target, stream represent
    ///     different runtime object, which can be used to execute the
    ///     compiled partition.
    /// @param inputs The inputs tensors, which contain metadata and buffer
    ///     For each logical tensor in the inputs_, there must be one tensor
    ///     in inputs arguments whose metadata is exactly same with it
    /// @param outputs The outputs tensors, which contain metadata and buffer
    ///     For each logical tensor in the outputs_, there must be one tensor
    ///     in output arguments whose metadata is exactly same with it.
    /// @return The status code
    /// @note
    ///     1. The given in/outputs tensors should have the same order with
    ///     the given in/outputs logical tensors on partition compilation.
    ///     However, the order of the given in/outputs tensors may be not same
    ///     with the in/outputs_ logical tensors. Backend should do the mapping
    ///     inside this function's implementation.
    ///     2. The stream is designed to interoperate with other custom op not
    ///     implemented using oneDNN graph. The custom op may want to execute
    ///     with the compiled partitions using same stream. For backend which
    ///     would like to manage device resource by its own, it may ignore
    ///     stream object. By doing this, it won’t support custom op
    ///     implemented outside oneDNN Graph.
    virtual status_t execute(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs)
            = 0;

#ifdef DNNL_WITH_SYCL
    virtual status_t execute_sycl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event)
            = 0;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    virtual status_t execute_ocl(const stream_t *astream,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const std::vector<cl_event> &ocl_deps, cl_event *ocl_event)
            = 0;
#endif

protected:
    /// The engine which this compiled_partition_impl_t is specialized
    /// for. Should directly store the engine that is given when calling
    /// partition_impl_t::compile
    const engine_t *engine_;

    /// The inputs logical tensors which this compiled_partition_impl_t
    /// is specialized for.Should have exact shape/dtype/layout and be
    /// in same order with inputs_ in partition_impl_t
    std::vector<logical_tensor_t> inputs_ {};

    /// The outputs logical tensors which this compiled_partition_impl_t
    /// is specialized for.Should have exact shape/dtype/layout and be
    /// in same order with outputs_ in partition_impl_t
    std::vector<logical_tensor_t> outputs_ {};

    /// The inplace_pair_t is used to indicate which input
    /// and output tensor given in execute can share same
    /// memory buffer.
    /// Take the following pattern as an example:
    ///    A    B
    ///     \  /
    ///      add
    ///       |
    ///       C
    /// Assume that A's id is 1, B's id is 2 and C's id is 3.
    /// If B and C can share same buffer, then the inplace_pairs_
    /// should be [{2, 3}]
    std::vector<inplace_pair_t> inplace_pairs_;
};

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
