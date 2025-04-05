/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef GRAPH_INTERFACE_VALUE_HPP
#define GRAPH_INTERFACE_VALUE_HPP

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/logical_tensor.hpp"

#include "graph/utils/debug.hpp"
#include "graph/utils/json.hpp"
#include "graph/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {

class value_t {
public:
    // A value connected to an op
    value_t(op_t &producer, size_t offset, const logical_tensor_t &lt,
            bool internal = false)
        : val_(lt)
        , producer_(&producer)
        , offset_(offset)
        , internal_(internal) {}

    // A value not associated with an op
    value_t(const logical_tensor_t &lt, bool internal = false)
        : val_(lt), internal_(internal) {}

    logical_tensor_t get_logical_tensor() const { return val_; }

    void set_logical_tensor(const logical_tensor_t &lt) {
        assertm(lt.id == val_.id, "logical tensor id conflict");
        val_ = lt;
    }

    void set_ndims(int32_t ndims) { val_.ndims = ndims; }

    void set_dims(const std::vector<dim_t> &dims) {
        val_.ndims = static_cast<int>(dims.size());
        for (size_t d = 0; d < dims.size(); ++d) {
            val_.dims[d] = dims[d];
        }
    }

    void set_layout_type(layout_type_t new_type) {
        val_.layout_type = new_type;
    }

    void set_layout_id(size_t layout_id) {
        val_.layout.layout_id = layout_id;
        val_.layout_type = layout_type::opaque;
    }

    void set_strides(const std::vector<dim_t> &strides) {
        for (size_t d = 0; d < strides.size(); ++d) {
            val_.layout.strides[d] = strides[d];
        }
        val_.layout_type = layout_type::strided;
    }

    void set_property(property_type_t ptype) { val_.property = ptype; }

    void set_data_type(data_type_t new_dtype) { val_.data_type = new_dtype; }

    op_t &get_producer() const {
        assertm(producer_ != nullptr, "Producer has not been set");
        return *producer_;
    }
    void set_producer(op_t &producer) { producer_ = &producer; }
    void reset_producer() { producer_ = nullptr; }
    bool has_producer() const { return producer_ != nullptr; }

    size_t get_offset() const { return offset_; }
    void set_offset(size_t offset) { offset_ = offset; }

    bool is_internal() const { return internal_; }

    bool operator==(const value_t &rhs) const {
        bool equal = logical_tensor_wrapper_t(this->val_)
                == logical_tensor_wrapper_t(rhs.val_);

        return equal && (this->producer_ == rhs.producer_)
                && (this->offset_ == rhs.offset_)
                && (this->consumers_ == rhs.consumers_)
                && (this->internal_ == rhs.internal_);
    }

    bool operator!=(const value_t &rhs) const { return !operator==(rhs); }

    // member class: A user of a value
    class consumer_t {
    public:
        consumer_t(op_t &op, size_t offset) : op_(&op), offset_(offset) {}

        bool operator==(const consumer_t &c) const {
            return op_ == c.op_ && offset_ == c.offset_;
        };

        op_t &get_op() const { return *op_; }

        size_t get_offset() const { return offset_; }

    private:
        op_t *op_ {nullptr};
        size_t offset_;
    };

    const std::vector<consumer_t> &get_consumers() const { return consumers_; }

    void add_consumer(op_t &op, size_t offset) {
        const consumer_t c {op, offset};
        if (std::find(consumers_.begin(), consumers_.end(), c)
                == consumers_.end()) {
            consumers_.push_back(c);
        }
    }

    /// Find whether a specific consumer exists in current op's consumers.
    ///
    /// @param start_index Search from [start_index]-th consumer. The
    /// consumers before [start_index] has already been matched in previous
    /// rounds of matching, so they cannot be searched again.
    /// @param kind Search for specific op_kind in the consumers
    /// @param expected_input_offset Check the consumer's input offset
    /// @param ignore_expected_input_offset: ignore the check of consumer's
    /// input offset, useful for binary ops.
    /// @returns an optional contained size_t, representing the offset of
    /// the found consumer
    graph::utils::optional_t<size_t> find_consumer(const size_t start_index,
            const op_kind_t kind, const size_t expected_input_offset,
            bool ignore_expected_input_offset = false);

    void swap_consumer(const size_t offset1, const size_t offset2) {
        std::swap(consumers_[offset1], consumers_[offset2]);
    }

    void remove_consumer(op_t &op, size_t offset) {
        const consumer_t c {op, offset};
        auto pos = std::find(consumers_.begin(), consumers_.end(), c);
        if (pos != consumers_.end()) {
            // Not all ops have been added through finalize
            consumers_.erase(pos);
        }
    }

    // Serialize
    status_t save(utils::json::json_writer_t *writer) const {
        writer->begin_object();
        auto lt = get_logical_tensor();
        auto ltw = logical_tensor_wrapper_t(lt);
        writer->write_keyvalue("id", ltw.id());
        writer->write_keyvalue(
                "dtype", std::string(utils::data_type2str(ltw.data_type())));
        // set {DNNL_GRAPH_UNKNOWN_DIM} when shape is not well inferred
        if (!ltw.is_shape_unknown()) {
            writer->write_keyvalue("shape", ltw.vdims());
        } else {
            const std::vector<dim_t> unknown {DNNL_GRAPH_UNKNOWN_DIM};
            writer->write_keyvalue("shape", unknown);
        }
        if (!ltw.is_stride_unknown()) {
            writer->write_keyvalue("stride", ltw.vstrides());
        } else {
            const std::vector<dim_t> unknown {DNNL_GRAPH_UNKNOWN_DIM};
            writer->write_keyvalue("stride", unknown);
        }
        writer->write_keyvalue("layout_type",
                std::string(utils::layout_type2str(ltw.layout_type())));
        writer->write_keyvalue("property_type",
                std::string(utils::property_type2str(ltw.property_type())));
        writer->end_object();
        return status::success;
    }

private:
    logical_tensor_t val_;

    op_t *producer_ {nullptr};
    size_t offset_ {std::numeric_limits<size_t>::max()};
    std::vector<consumer_t> consumers_;
    bool internal_ {false};
};

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
