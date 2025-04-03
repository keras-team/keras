// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/pytorch/visibility.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

using DecoderRTInfo = std::unordered_map<std::string, ov::Any>;

/// Plays a role of node, block and module decoder (kind of temporary fat API)
class PYTORCH_FRONTEND_API TorchDecoder : public IDecoder {
public:
    ~TorchDecoder() override;

    // Do not search for input in tensor map; try to access it as a constant of specified type T and return its value
    // Using Any here is an easy way to avoid template definition, returned object is supposed to be of one of the
    // fundamental types like int, float etc.
    virtual Any const_input(size_t index) const = 0;

    // Using size_t for input/output unique ids are in sync with torch code, see def in
    // torch/include/torch/csrc/jit/ir/ir.h, Value::unique_

    // TODO: set of input and output methods are not aligned; also they are not aligned with the rest of FEs

    virtual const std::vector<size_t>& inputs() const = 0;

    // ------------------------------
    // TODO: physically inputs and outputs refer to PT Values so shape/type is not a property of input/output
    // Do we need a separate Decoder for Tensor to request properties of it instead of having an impression
    // that inputs/outputs have types and shapes?

    // Return debug name of the input tensor
    virtual const std::string& get_input_debug_name(size_t index) const = 0;

    // Return signature name of the input tensor
    virtual const std::string& get_input_signature_name(size_t index) const = 0;

    // Return shape if inputs has torch::Tensor type in the original model, otherwise returns the shape [] of a scalar
    virtual PartialShape get_input_shape(size_t index) const = 0;

    // Return strides if inputs has torch::Tensor type in original model, otherwise return [].
    virtual const std::vector<size_t>& get_input_strides(size_t index) const = 0;

    // Return element::Type when it the original type can be represented, otherwise returns PT-specific data type object
    // (see custom_type.hpp)
    virtual Any get_input_type(size_t index) const = 0;

    // Return debug name of the input tensor
    virtual const std::string& get_output_debug_name(size_t index) const = 0;

    // Return shape if inputs has torch::Tensor type in the original model, otherwise returns the shape [] of a scalar
    virtual PartialShape get_output_shape(size_t index) const = 0;

    // Return element::Type when it the original type can be represented, otherwise returns PT-specific data type object
    // (see custom_type.hpp)
    virtual Any get_output_type(size_t index) const = 0;
    // ------------------------------

    // TODO: required? can be implemented in the context of a single node?
    virtual bool input_is_none(size_t index) const = 0;

    virtual OutputVector try_decode_get_attr() const = 0;

    // Work for natural constant nodes, e.g. for prim::Constant; don't know other nodes kinds that fit
    // TODO: why OutputVector instead of just single output?
    virtual OutputVector as_constant() const = 0;

    // Get string from constant. Work for natural constant nodes, e.g. for prim::Constant; don't know other nodes kinds
    // that fit
    virtual const std::string& as_string() const = 0;

    // Returns PT node kind as a string mnemonics for native type uint32_t Symbol in Torch
    // Decide whether we need an equivalent member for integer representation (in this case a map is required to
    // understand what it means)
    virtual const std::string& get_op_type() const = 0;

    // Returns PT node schema as a string
    virtual const std::string& get_schema() const = 0;

    // TODO: use canonical name output_size
    virtual size_t num_of_outputs() const = 0;

    // If the node output is a list of getitem nodes, returns the size of the list
    // If the node output is not a list of getitem nodes, returns 0
    virtual size_t output_list_size() const = 0;

    // Return a vector of output IDs
    virtual const std::vector<size_t>& outputs() const = 0;

    // Return a vector of output IDs
    virtual size_t output(size_t index) const = 0;

    // Embed mapping to/from the original node representation from/to node passed as a parameter
    // the representation of this mapping is specific for particular decorated type and may be NOP
    // returns the same node as syntactically convenient way to make nested sentences in code
    virtual std::shared_ptr<Node> mark_node(std::shared_ptr<Node> ov_node) const = 0;

    /// \brief Returns the number of sub-graphs that can be enumerated with get_subgraph
    virtual size_t get_subgraph_size() const = 0;

    /// \brief Returns subgraph converted on demand by the first access
    /// If there is no query for specific sub-graph it shouldn't be converted
    // node_visitor is a function that will be fed by nodes in subgraph for all nodes in graph
    virtual void visit_subgraph(std::function<void(std::shared_ptr<TorchDecoder>)> node_visitor) const = 0;

    /// Probably this together with immediate nodes visitor is a replacement for visit_subgraphs with an index
    virtual std::shared_ptr<TorchDecoder> get_subgraph_decoder(size_t index) const = 0;

    /// \brief Returns if output may contain alias of input in AliasDB
    virtual bool may_produce_alias(size_t in_index, size_t out_index) const = 0;

    /// Returns if input is inlined
    // Used in Torch.FX decoder
    virtual bool is_input_inlined(size_t index) const = 0;

    /// Return decoder for inlined input
    virtual std::shared_ptr<TorchDecoder> get_inlined_input_decoder(size_t index) const = 0;

    /// Returns named attribute as Any. For example kwargs input for FX graph
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    /// Returns index of named input. For example kwargs input for FX graph
    virtual size_t get_named_input(const std::string& name) const = 0;

    /// Returns the id of the decoder type ("fx": TorchFX, "ts": TorchScript)
    virtual const std::string& decoder_type_name() const = 0;

    /// \brief Returns the rt_info for the element
    virtual DecoderRTInfo get_rt_info() const = 0;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
