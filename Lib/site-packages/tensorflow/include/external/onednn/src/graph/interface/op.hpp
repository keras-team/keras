/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GRAPH_INTERFACE_OP_HPP
#define GRAPH_INTERFACE_OP_HPP

#include <limits>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/logical_tensor.hpp"
#include "graph/interface/value.hpp"

#include "graph/utils/attribute_value.hpp"
#include "graph/utils/json.hpp"

namespace dnnl {
namespace impl {
namespace graph {
/// forward declaration
class op_schema_t;
class partition_impl_t;
} // namespace graph
} // namespace impl
} // namespace dnnl

/******************************************************************************
 * op functionalities:
 *  1. support frontend API
 *      - create op with id, kind, name string.
 *      - get id, kind, etc.
 *      - add input logical tensors.
 *      - add output logical tensors.
 *      - set/get attributes of the op.
 *  2. as a op on the graph
 *      - input logical tensor -> value -> one producer.
 *      - output logical tensor -> value -> multiple consumers.
 *      - set/get producers and consumers.
 *      - verify the op is legitimate, with op schema.
 *  3. as an internal (fused) op on the graph
 *      - create with id (not provided by users, how to generate?), kind, name string.
 *      - merge attributes from the source ops.
 *      - contain the ids of source ops.
 *      - fused op -> partition.
 *
 *****************************************************************************/

struct dnnl_graph_op : public std::enable_shared_from_this<dnnl_graph_op> {
public:
    using op_kind_t = dnnl::impl::graph::op_kind_t;
    using op_attr_t = dnnl::impl::graph::op_attr_t;
    using logical_tensor_t = dnnl::impl::graph::logical_tensor_t;
    using attribute_kind_t = dnnl::impl::graph::attribute_kind_t;
    using status_t = dnnl::impl::graph::status_t;
    using attribute_value_t = dnnl::impl::graph::utils::attribute_value_t;
    using value_t = dnnl::impl::graph::value_t;
    using pair_t = std::pair<size_t, size_t>; // <op_id, input/output offset>

    const static size_t DEFAULT_ID = std::numeric_limits<size_t>::max();

    // create dnnl_graph_op with explicit id, kind, and string
    dnnl_graph_op(
            size_t id, op_kind_t kind, std::string name, bool internal = false);

    // create dnnl_graph_op with default id, only for internal use.
    dnnl_graph_op(op_kind_t kind, std::string name)
        : dnnl_graph_op(DEFAULT_ID, kind, std::move(name), true) {}

    // convenient function to make tests happy
    dnnl_graph_op(op_kind_t kind)
        : dnnl_graph_op(DEFAULT_ID, kind, kind2str(kind), true) {}

    ~dnnl_graph_op() = default;

    // which op produced this input?
    dnnl_graph_op *get_input_op(size_t index) {
        return &(inputs_[index]->get_producer());
    }

    bool operator==(const dnnl_graph_op &other) const {
        return this->get_id() == other.get_id()
                && this->get_kind() == other.get_kind()
                && this->get_name() == other.get_name()
                && this->is_internal() == other.is_internal()
                && attributes_equal(other);
    }

    bool attributes_equal(const dnnl_graph_op &other) const {
        for (auto attr : this->attributes_) {
            // There is no need to check internal attributes.
            if (attr.first >= dnnl_graph_op_attr_t::dnnl_graph_op_attr_end)
                continue;
            if (other.attributes_.find(attr.first) == other.attributes_.end())
                return false;
            if (attr.second != other.attributes_.at(attr.first)) return false;
        }
        return true;
    }

    // some getters
    op_kind_t get_kind() const { return kind_; }
    size_t get_id() const { return id_; }
    const std::string &get_name() const { return name_; }
    bool is_internal() const { return internal_; }

    ///////////////////////////////////////////////////////////////////////////
    // input values
    size_t num_inputs() const { return inputs_.size(); }

    // add an input value to the op
    void add_input(const std::shared_ptr<value_t> &value) {
        // setup the input_tensor_map_
        const size_t offset = inputs_.size();
        input_tensor_map_[offset] = std::make_pair(id_, offset);

        inputs_.push_back(value);
    }

    // frontend API, add an input logical tensor to the op
    void add_input(const logical_tensor_t &lt) {
        add_input(std::make_shared<value_t>(lt));
    }

    std::shared_ptr<value_t> get_input_value(size_t offset) const {
        return inputs_.at(offset);
    }

    const std::vector<std::shared_ptr<value_t>> &get_input_values() const {
        return inputs_;
    }

    void fill_and_connect_input(
            size_t index, dnnl_graph_op &op, size_t offset) {
        while (op.num_outputs() <= offset) {
            op.add_output(dnnl::impl::graph::zero_logical_tensor());
        }
        connect_input(index, op.get_output_value(offset));
    }

    void connect_input(size_t index, dnnl_graph_op &op, size_t offset) {
        connect_input(index, op.get_output_value(offset));
    }

    void connect_input(size_t index, const std::shared_ptr<value_t> &output) {
        output->add_consumer(*this, index);
        if (inputs_.size() <= index) { inputs_.resize(index + 1); }
        inputs_[index] = output;
    }

    void swap_input_values(size_t offset1, size_t offset2) {
        std::shared_ptr<value_t> input1 = inputs_[offset1];
        input1->remove_consumer(*this, offset1);
        std::shared_ptr<value_t> input2 = inputs_[offset2];
        input2->remove_consumer(*this, offset2);
        std::swap(inputs_[offset1], inputs_[offset2]);
        input1->add_consumer(*this, offset2);
        input2->add_consumer(*this, offset1);
    }

    ///////////////////////////////////////////////////////////////////////////
    // output values
    size_t num_outputs() const { return outputs_.size(); }

    void add_output(const std::shared_ptr<value_t> &value) {
        const size_t offset = outputs_.size();
        output_tensor_map_[offset] = std::make_pair(id_, offset);

        value->set_producer(*this);
        value->set_offset(offset);
        outputs_.push_back(value);
    }

    // frontend API, add an output logical tensor to the op
    void add_output(const logical_tensor_t &lt) {
        add_output(std::make_shared<value_t>(lt));
    }

    void connect_output(size_t index, std::shared_ptr<value_t> &value) {
        value->set_producer(*this);
        value->set_offset(index);

        if (outputs_.size() <= index) { outputs_.resize(index + 1); }
        outputs_[index] = value;
    }

    const std::vector<std::shared_ptr<value_t>> &get_output_values() const {
        return outputs_;
    }

    std::shared_ptr<value_t> get_output_value(size_t offset) const {
        return outputs_.at(offset);
    }

    size_t num_output_consumers(size_t offset) const {
        return get_output_value(offset)->get_consumers().size();
    }

    ///////////////////////////////////////////////////////////////////////////
    // attributes handling
    template <typename Attr>
    dnnl_graph_op &set_attr(op_attr_t name, const Attr &a) {
        auto it = attributes_.find(name);
        if (it != attributes_.end()) {
            it->second = {a};
        } else {
            attributes_.insert({name, {a}});
        }
        return *this;
    }

    dnnl_graph_op &set_attr(op_attr_t name, const attribute_value_t &a) {
        auto it = attributes_.find(name);
        if (it != attributes_.end()) {
            it->second = a;
        } else {
            attributes_.insert({name, a});
        }
        return *this;
    }

    template <typename value_type>
    value_type get_attr(op_attr_t name) const {
        auto it = attributes_.find(name);
        assertm(it != attributes_.end(), "don't have such attribute");
        if (it == attributes_.end())
            return {};
        else
            return it->second.get<value_type>();
    }

    template <typename Attr>
    status_t get_attr(op_attr_t name, const Attr **attr) const {
        const auto &found = attributes_.find(name);
        if (found == attributes_.end()) {
            return dnnl::impl::graph::status::invalid_arguments;
        }

        Attr &val = found->second.get<Attr>();
        *attr = &val;
        return dnnl::impl::graph::status::success;
    }

    bool has_attr(op_attr_t name) const {
        return attributes_.find(name) != attributes_.end();
    }

    void remove_attr(op_attr_t name) { attributes_.erase(name); }

    const std::unordered_map<op_attr_t, attribute_value_t> &
    get_attributes() const {
        return attributes_;
    }

    size_t num_attributes() const { return attributes_.size(); }

    void merge_attributes(
            const std::unordered_map<op_attr_t, attribute_value_t> &attrs) {
        attributes_.insert(attrs.begin(), attrs.end());
    }

    bool is_same_attr_value(
            const dnnl_graph_op &op_b, op_attr_t attr_name) const {
        const auto &attr_a = get_attributes();
        const auto &attr_b = op_b.get_attributes();
        auto it_a = attr_a.find(attr_name);
        auto it_b = attr_b.find(attr_name);

        const bool same = (it_a == attr_a.end() || it_b == attr_b.end())
                ? false
                : (it_a->second == it_b->second);

        return same;
    }

    bool has_same_attr_values(const dnnl_graph_op &op_b,
            std::set<op_attr_t> excepted = {}) const {
        return std::all_of(attributes_.begin(), attributes_.end(),
                [&](const std::pair<op_attr_t, attribute_value_t> &attr) {
                    return excepted.count(attr.first)
                            ? true
                            : is_same_attr_value(op_b, attr.first);
                });
    }

    static std::string attr2str(op_attr_t attr) {
        using namespace dnnl::impl::graph;
#define CASE(a) \
    case (op_attr::a): return #a

        switch (attr) {
            CASE(alpha);
            CASE(beta);
            CASE(epsilon);
            CASE(max);
            CASE(min);
            CASE(momentum);
            CASE(scales);
            CASE(axis);
            CASE(begin_norm_axis);
            CASE(groups);
            CASE(axes);
            CASE(dilations);
            CASE(weights_shape);
            CASE(src_shape);
            CASE(kernel);
            CASE(order);
            CASE(output_padding);
            CASE(dst_shape);
            CASE(pads_begin);
            CASE(pads_end);
            CASE(shape);
            CASE(sizes);
            CASE(strides);
            CASE(zps);
            CASE(exclude_pad);
            CASE(keep_dims);
            CASE(keep_stats);
            CASE(per_channel_broadcast);
            CASE(special_zero);
            CASE(transpose_a);
            CASE(transpose_b);
            CASE(use_affine);
            CASE(use_dst);
            CASE(auto_broadcast);
            CASE(auto_pad);
            CASE(coordinate_transformation_mode);
            CASE(data_format);
            CASE(weights_format);
            CASE(mode);
            CASE(qtype);
            CASE(rounding_type);
            CASE(matched);
            CASE(backend);
            CASE(partition_id);
            CASE(op_depth);
            default: return "undefined_attr";
        }
#undef CASE
    }

    static std::string kind2str(op_kind_t kind) {
        using namespace dnnl::impl::graph::op_kind;
#define CASE(k) \
    case (k): return #k

        switch (kind) {
            CASE(Abs);
            CASE(AbsBackward);
            CASE(Add);
            CASE(AvgPool);
            CASE(AvgPoolBackward);
            CASE(BatchNormInference);
            CASE(BatchNormForwardTraining);
            CASE(BatchNormTrainingBackward);
            CASE(BiasAdd);
            CASE(BiasAddBackward);
            CASE(Clamp);
            CASE(ClampBackward);
            CASE(Concat);
            CASE(Convolution);
            CASE(ConvolutionBackwardData);
            CASE(ConvolutionBackwardWeights);
            CASE(ConvTranspose);
            CASE(ConvTransposeBackwardData);
            CASE(ConvTransposeBackwardWeights);
            CASE(Dequantize);
            CASE(Divide);
            CASE(DynamicDequantize);
            CASE(DynamicQuantize);
            CASE(Elu);
            CASE(EluBackward);
            CASE(End);
            CASE(Exp);
            CASE(GELU);
            CASE(GELUBackward);
            CASE(HardSigmoid);
            CASE(HardSigmoidBackward);
            CASE(HardSwish);
            CASE(HardSwishBackward);
            CASE(Interpolate);
            CASE(InterpolateBackward);
            CASE(LayerNorm);
            CASE(LayerNormBackward);
            CASE(LeakyReLU);
            CASE(Log);
            CASE(LogSoftmax);
            CASE(LogSoftmaxBackward);
            CASE(MatMul);
            CASE(Maximum);
            CASE(MaxPool);
            CASE(MaxPoolBackward);
            CASE(Minimum);
            CASE(Mish);
            CASE(MishBackward);
            CASE(Multiply);
            CASE(Pow);
            CASE(PReLU);
            CASE(PReLUBackward);
            CASE(Quantize);
            CASE(Reciprocal);
            CASE(ReduceL1);
            CASE(ReduceL2);
            CASE(ReduceMax);
            CASE(ReduceMean);
            CASE(ReduceMin);
            CASE(ReduceProd);
            CASE(ReduceSum);
            CASE(ReLU);
            CASE(ReLUBackward);
            CASE(Reorder);
            CASE(Round);
            CASE(Select);
            CASE(Sigmoid);
            CASE(SigmoidBackward);
            CASE(SoftMax);
            CASE(SoftMaxBackward);
            CASE(SoftPlus);
            CASE(SoftPlusBackward);
            CASE(Sqrt);
            CASE(SqrtBackward);
            CASE(Square);
            CASE(SquaredDifference);
            CASE(StaticReshape);
            CASE(StaticTranspose);
            CASE(Subtract);
            CASE(Tanh);
            CASE(TanhBackward);
            CASE(TypeCast);
            CASE(Wildcard);
            CASE(LastSymbol);
            default: return "internal_op";
        }
#undef CASE
    }

    ///////////////////////////////////////////////////////////////////////////
    // partition handling
    void set_partition(dnnl::impl::graph::partition_impl_t *part) {
        partition_ = part;
    }

    dnnl::impl::graph::partition_impl_t *get_partition() const {
        return partition_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // As a fused op
    bool is_fused() const { return !op_ids_.empty(); }

    void add_op_ids(size_t id) { op_ids_.push_back(id); }

    void add_op_ids(const std::vector<size_t> &ids) {
        for (auto id : ids)
            op_ids_.push_back(id);
    }

    const std::vector<size_t> &get_op_ids() const { return op_ids_; }

    const std::unordered_map<size_t, pair_t> &get_input_tensor_map() const {
        return input_tensor_map_;
    }

    const std::unordered_map<size_t, pair_t> &get_output_tensor_map() const {
        return output_tensor_map_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Serialize
    status_t save(dnnl::impl::graph::utils::json::json_writer_t *writer) const {
        writer->begin_object();
        writer->write_keyvalue("id", get_id());
        writer->write_keyvalue("name", get_name());
        writer->write_keyvalue("kind", kind2str(get_kind()));
        auto attrs = get_attributes();
        std::unordered_map<std::string, attribute_value_t> copied_attrs;
        std::for_each(attrs.begin(), attrs.end(),
                [&copied_attrs](
                        const std::pair<op_attr_t, attribute_value_t> &v) {
                    copied_attrs.emplace(
                            std::make_pair(attr2str(v.first), v.second));
                });

        copied_attrs.erase("op_depth");
        copied_attrs.erase("matched");
        writer->write_keyvalue("attrs", copied_attrs);
        writer->write_keyvalue("inputs", get_input_values());
        writer->write_keyvalue("outputs", get_output_values());
        writer->end_object();
        return dnnl::impl::graph::status::success;
    }

private:
    size_t id_ {};
    op_kind_t kind_ {};
    std::string name_ {};
    std::vector<std::shared_ptr<value_t>> inputs_ {};
    std::vector<std::shared_ptr<value_t>> outputs_ {};
    std::unordered_map<op_attr_t, attribute_value_t> attributes_;

    dnnl::impl::graph::partition_impl_t *partition_ {nullptr};
    bool internal_ {false};

    // fused op: we still need to represent a fused op
    // possibly we can remove these once the new backend API and new pattern
    // matcher is done.
    std::vector<size_t> op_ids_ {};
    // Map from the fused op input index -> (original op id, op input offset)
    std::unordered_map<size_t, pair_t> input_tensor_map_;
    // Map from the fused op output index -> (original op id, op output offset)
    std::unordered_map<size_t, pair_t> output_tensor_map_;
};

namespace dnnl {
namespace impl {
namespace graph {
template <typename FUN>
status_t topo_order_visit(const std::vector<op_t *> &root_ops, const FUN &f) {
    std::stack<op_t *> todo;
    std::unordered_set<op_t *> visited;
    for (auto &op : root_ops) {
        todo.push(op);
    }

    while (!todo.empty()) {
        op_t *top = todo.top();
        if (visited.find(top) != visited.end()) {
            todo.pop();
            continue;
        }
        bool ready = true;
        auto &inputs = top->get_input_values();
        // Need to iterate backwards because some examples are broken and depend
        // on the partition order
        for (auto it = inputs.rbegin(); it != inputs.rend(); ++it) {
            if ((*it)->has_producer()) {
                op_t &producer = (*it)->get_producer();
                if (visited.find(&producer) == visited.end()) {
                    // Need to visit first
                    todo.push(&producer);
                    ready = false;
                }
            }
        }
        if (ready) {
            todo.pop();
            status_t ret = f(top);
            if (ret != status::success) return ret;
            visited.insert(top);
        }
    }
    return status::success;
}
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
