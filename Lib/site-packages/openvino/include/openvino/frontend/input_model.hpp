// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/graph_iterator.hpp"
#include "openvino/frontend/place.hpp"
#include "openvino/frontend/visibility.hpp"

namespace ov {
namespace frontend {
class FrontEnd;
/// \brief InputModel class represents an original, not yet converted model graph in a
/// framework format given services to find places of interest in a graph or specialize/edit
/// the model before conversion.
///
/// \note Class methods are divided into several groups: searching for places, naming and
/// annotation, topology editing, setting tensor properties.
///
/// Editing requests may affect ability to convert the original model to OV Model.
/// Aim to provide these editing capabilities is to unlock conversion for models that
/// are not natively supported "as-is" because of undefined shapes, types or operations.
///
/// Specific front-end implementation is supposed to have a lazy implementation for
/// all methods, not doing a complete load of a model without an explicit method call.
/// For example, the list of all inputs are not pre-fetched by InputModel derived
/// class instance creation, but only when get_inputs method is called. But it is not
/// an obligation, the most convenient way should be chosen depending on the framework
/// model representation.
///
/// All editing requests affect the model representation that is held behind the scene
/// successive method calls observe a new graph structure.
class FRONTEND_API InputModel {
    std::shared_ptr<void> m_shared_object;
    std::shared_ptr<InputModel> m_actual;
    friend class ::ov::frontend::FrontEnd;

public:
    using Ptr = std::shared_ptr<InputModel>;

    InputModel() = default;
    InputModel(const InputModel&) = delete;
    InputModel(InputModel&&) = delete;
    InputModel& operator=(const InputModel&) = delete;
    InputModel& operator=(InputModel&&) = delete;

    virtual ~InputModel();

    /////  Searching for places  /////

    /// \brief Returns all inputs for a model
    /// An input is a place in a graph where data is supposed to flow inside graph from
    /// outside. It can be a tensor, port, operation; which kind of place can be an output
    /// is FW dependent. Usually framework models have a dedicated artifact to code model
    /// input, it can be a tensor without producer, that writes to it in ONNX, or a special
    /// operation like Placeholder in TensorFlow.
    ///
    /// \return A vector of input place references
    virtual std::vector<Place::Ptr> get_inputs() const;

    /// \brief Returns all output for a model
    /// An output is a terminal place in a graph where data escapes the flow. It can be a
    /// tensor, port, operation; which kind of place can be an output is FW dependent. In
    /// comparison to a graph input, the output is less formally defined thing and
    /// determination of initial list of outputs may include some conventions defined by a
    /// frontend itself, not a framework. For example, all output ports without consumers
    /// may be considered as outputs.
    ///
    /// \return A vector of output place references
    virtual std::vector<Place::Ptr> get_outputs() const;

    /// \brief Returns a tensor place by a tensor name following framework conventions, or
    /// nullptr if a tensor with this name doesn't exist.
    /// \param tensor_name Name of tensor
    /// \return Tensor place corresponding to specified tensor name or nullptr if not exists
    virtual Place::Ptr get_place_by_tensor_name(const std::string& tensor_name) const;

    /// \brief Returns a tensor place by an input index.
    /// \param input_idx Index of model input
    /// \return Tensor place corresponding to specified input index or nullptr
    virtual Place::Ptr get_place_by_input_index(size_t input_idx) const;

    /// \brief Returns an operation place by an operation name following framework
    /// conventions, or nullptr if an operation with this name doesn't exist.
    /// \param operation_name Name of operation
    /// \return Place representing operation or nullptr if not exists
    virtual Place::Ptr get_place_by_operation_name(const std::string& operation_name) const;

    /// \brief Returns an input port place by operation name and appropriate port index
    /// \param operation_name Name of operation
    /// \param input_port_index Index of input port for this operation
    /// \return Place representing input port of operation or nullptr if not exists
    virtual Place::Ptr get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                  int input_port_index);

    /// \brief Returns an output port place by operation name and appropriate port index
    /// \param operation_name Name of operation
    /// \param output_port_index Index of output port for this operation
    /// \return Place representing output port of operation or nullptr if not exists
    virtual Place::Ptr get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                                   int output_port_index);

    ///// Naming and annotation  /////

    /// \brief Sets name for tensor. Overwrites existing names of this place
    /// \param tensor Tensor place
    /// \param new_name New name for this tensor
    virtual void set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name);

    /// \brief Adds new name for tensor
    /// \param tensor Tensor place
    /// \param new_name New name to be added to this place
    virtual void add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name);

    /// \brief Sets name for operation. Overwrites existing names of this place
    /// \param operation Operation place
    /// \param new_name New name for this operation
    virtual void set_name_for_operation(const Place::Ptr& operation, const std::string& new_name);

    /// \brief Unassign specified name from tensor place(s)
    /// \param name Name of tensor
    virtual void free_name_for_tensor(const std::string& name);

    /// \brief Unassign specified name from operation place(s)
    /// \param name Name of operation
    virtual void free_name_for_operation(const std::string& name);

    /// \brief Set name for a particular dimension of a place (e.g. batch dimension)
    /// \param place Model's place
    /// \param shape_dim_index Dimension index
    /// \param dim_name Name to assign on this dimension
    virtual void set_name_for_dimension(const Place::Ptr& place, size_t shape_dim_index, const std::string& dim_name);

    ///// Topology Editing  /////

    /// \brief Cut immediately before this place and assign this place as new input; prune
    /// all nodes that don't contribute to any output.
    /// \param place New place to be assigned as input
    /// \param new_name_optional Optional new name assigned to this input place
    virtual void cut_and_add_new_input(const Place::Ptr& place, const std::string& new_name_optional = "");

    /// \brief Cut immediately after this place and assign this place as new output; prune
    /// all nodes that don't contribute to any output.
    /// \param place New place to be assigned as output
    /// \param new_name_optional Optional new name assigned to this output place
    virtual void cut_and_add_new_output(const Place::Ptr& place, const std::string& new_name_optional = "");

    /// \brief Assign this place as new output or add necessary nodes to represent a new
    /// output.
    ///
    /// \param place Anchor point to add an output
    /// \return new output place, may be the same as a given place
    virtual Place::Ptr add_output(const Place::Ptr& place);

    /// \brief Removes any sinks directly attached to this place with all inbound data flow
    /// if it is not required by any other output.
    /// \param place Model place
    virtual void remove_output(const Place::Ptr& place);

    /// \brief Replaces all existing outputs with new ones removing all data flow that is
    /// not required for new outputs.
    ///
    /// \param outputs Vector with places that will become new outputs; may intersect
    /// existing outputs.
    /// \param outputs Array of new output places
    virtual void override_all_outputs(const std::vector<Place::Ptr>& outputs);

    /// \brief Modifies the graph to use new inputs instead of existing ones. New inputs
    /// should completely satisfy all existing outputs.
    /// \param inputs Array of new input places
    virtual void override_all_inputs(const std::vector<Place::Ptr>& inputs);

    /// \brief Leaves only subgraph that are defined by new inputs and new outputs.
    /// \param inputs Array of new input places
    /// \param outputs Array of new output places
    virtual void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs);

    ///// Setting tensor properties  /////

    /// \brief Defines all possible shape that may be used for this place; place should be
    /// uniquely refer to some data. This partial shape will be converted to corresponding
    /// shape of results OV nodes and will define shape inference when the model is
    /// converted to OV.
    /// \param place Model place
    /// \param shape Partial shape for this place
    virtual void set_partial_shape(const Place::Ptr& place, const ov::PartialShape& shape);

    /// \brief Returns current partial shape used for this place
    /// \param place Model place
    /// \return Partial shape for this place
    virtual ov::PartialShape get_partial_shape(const Place::Ptr& place) const;

    /// \brief Sets new element type for a place
    /// \param place Model place
    /// \param type New element type
    virtual void set_element_type(const Place::Ptr& place, const ov::element::Type& type);

    /// \brief Returns current element type used for this place
    /// \param place Model place
    /// \return Element type for this place
    virtual ov::element::Type get_element_type(const Place::Ptr& place) const;

    /// \brief Freezes a tensor with statically defined value or replace existing value for
    /// already constant node or tensor
    /// \param place Tensor place
    /// \param value Value for tensor place representing a memory buffer
    virtual void set_tensor_value(const Place::Ptr& place, const void* value);

    /// \brief Defines partial value (lower bound and upper bound) for a tensor place
    /// TODO: more details for min_value and max_value format; who defines shape?
    /// \param place Tensor place
    /// \param min_value Lower bound of partial value for tensor place
    /// \param max_value Upper bound of partial value for tensor place
    virtual void set_tensor_partial_value(const Place::Ptr& place, const void* min_value, const void* max_value);
};

}  // namespace frontend
}  // namespace ov
