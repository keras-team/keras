// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/frontend/visibility.hpp"

namespace ov {
namespace frontend {
/// \brief An interface for identifying a place in a graph and iterate over it; can refer to
/// an operation node, tensor, port etc.
///
/// \note Each front end implementation provides specialization of this interface to
/// represent a place in a model graph. Various methods in the front end classes accept and
/// retrieve instances of Place to point to particular node part which should be modified or
/// satisfies some criteria. For example, this class is used to report model inputs
/// and outputs, for searching operations and tensors by name, for setting shape etc.
///
/// Place can refer to Tensor, Input Edge, Input Port, Operation, Output Port, Output Edge
///
///                [Tensor A]
///                    |
///                    | [Input Edge]
///                    |
///                    V
///           -------------------
///           [  [Input Port 0] ]
///           [                 ]
///           [   Operation A   ]
///           [                 ]
///           [ [Output Port 0] ]
///           -------------------
///                    |
///                    | [Output Edge]
///                    |
///                    V
///                [Tensor B]
///                    |
///                    | [Input Edge]
///                    |
///                    V
///           -------------------
///           [  [Input Port 0] ]
///           [                 ]
///           [   Operation B   ]
///           [                 ]
///           [ [Output Port 0] ]
///           -------------------
///                    |
///                    | [Output Edge]
///                    |
///                    V
///                [Tensor C]
///
class FRONTEND_API Place {
public:
    typedef std::shared_ptr<Place> Ptr;

    virtual ~Place();

    /// \brief All associated names (synonyms) that identify this place in the graph in a
    /// framework specific way
    ///
    /// \return A vector of strings each representing a name that identifies this place in
    /// the graph. Can be empty if there are no names associated with this place or name
    /// cannot be attached.
    virtual std::vector<std::string> get_names() const;

    /// \brief Returns references to all operation nodes that consume data from this place
    /// \note It can be called for any kind of graph place searching for the first consuming
    /// operations. It is optional if place has only one output port
    ///
    /// \return A vector with all operation node references that consumes data from this
    /// place
    virtual std::vector<Ptr> get_consuming_operations() const;

    /// \brief Returns references to all operation nodes that consume data from this place
    /// for specified output port
    ///
    /// \note It can be called for any kind of graph place searching for the first consuming
    /// operations.
    ///
    /// \param output_port_index If place is an operational node it specifies which output
    /// port should be considered.
    ///
    /// \return A vector with all operation node references that consumes data from this
    /// place
    virtual std::vector<Ptr> get_consuming_operations(int output_port_index) const;

    /// \brief Returns references to all operation nodes that consume data from this place
    /// for specified output port
    ///
    /// \note It can be called for any kind of graph place searching for the first consuming
    /// operations.
    ///
    /// \param outputName If a given place is itself an operation node, this specifies name
    /// of output port group
    ///
    /// \return A vector with all operation node references that consumes data from this
    /// place
    virtual std::vector<Ptr> get_consuming_operations(const std::string& outputName) const;

    /// \brief Returns references to all operation nodes that consume data from this place
    /// for specified output port
    ///
    /// \note It can be called for any kind of graph place searching for the first consuming
    /// operations.
    ///
    /// \param outputName If a given place is itself an operation node, this specifies name
    /// of output port group, each group can have multiple ports
    ///
    /// \param outputPortIndex If place is an operational node it specifies which output
    /// port should be considered.
    ///
    /// \return A vector with all operation node references that consumes data from this
    /// place
    virtual std::vector<Ptr> get_consuming_operations(const std::string& outputName, int outputPortIndex) const;

    /// \brief Returns a tensor place that gets data from this place; applicable for
    /// operations, output ports and output edges which have only one output port
    ///
    /// \return A tensor place which hold the resulting value for this place
    virtual Ptr get_target_tensor() const;

    /// \brief Returns a tensor place that gets data from this place; applicable for operations
    ///
    /// \param outputName Name of output port group
    ///
    /// \return A tensor place which hold the resulting value for this place
    virtual Ptr get_target_tensor(const std::string& outputName) const;

    /// \brief Returns a tensor place that gets data from this place; applicable for operations
    ///
    /// \param outputName Name of output port group, each group can have multiple ports
    ///
    /// \param outputPortIndex Output port index if the current place is an operation node
    /// and has multiple output ports
    ///
    /// \return A tensor place which hold the resulting value for this place
    virtual Ptr get_target_tensor(const std::string& outputName, int outputPortIndex) const;

    /// \brief Returns a tensor place that gets data from this place; applicable for operations
    ///
    /// \param output_port_index Output port index if the current place is an operation node
    /// and has multiple output ports
    ///
    /// \return A tensor place which hold the resulting value for this place
    virtual Ptr get_target_tensor(int output_port_index) const;

    /// \brief Returns a tensor place that supplies data for this place; applicable for
    /// operations, input ports and input edges which have only one input port
    ///
    /// \return A tensor place which supplies data for this place
    virtual Ptr get_source_tensor() const;

    /// \brief Returns a tensor place that supplies data for this place; applicable for operations
    ///
    /// \param input_port_index Input port index for operational nodes.
    ///
    /// \return A tensor place which supplies data for this place
    virtual Ptr get_source_tensor(int input_port_index) const;

    /// \brief Returns a tensor place that supplies data for this place; applicable for operations
    ///
    /// \param inputName Name of input port group
    ///
    /// \return A tensor place which supplies data for this place
    virtual Ptr get_source_tensor(const std::string& inputName) const;

    /// \brief Returns a tensor place that supplies data for this place; applicable for operations
    ///
    /// \param inputName If a given place is itself an operation node, this specifies name
    /// of output port group, each group can have multiple ports
    ///
    /// \param inputPortIndex Input port index for operational nodes.
    ///
    /// \return A tensor place which supplies data for this place
    virtual Ptr get_source_tensor(const std::string& inputName, int inputPortIndex) const;

    /// \brief Get an operation node place that immediately produces data for this place;
    /// applicable if place has only one input port
    ///
    /// \return An operation place that produces data for this place
    virtual Ptr get_producing_operation() const;

    /// \brief Get an operation node place that immediately produces data for this place
    ///
    /// \param input_port_index If a given place is itself an operation node, this specifies
    /// a port index
    ///
    /// \return An operation place that produces data for this place
    virtual Ptr get_producing_operation(int input_port_index) const;

    /// \brief Get an operation node place that immediately produces data for this place
    ///
    /// \param inputName If a given place is itself an operation node, this specifies name
    /// of output port group
    ///
    /// \return An operation place that produces data for this place
    virtual Ptr get_producing_operation(const std::string& inputName) const;

    /// \brief Get an operation node place that immediately produces data for this place
    ///
    /// \param inputName If a given place is itself an operation node, this specifies name
    /// of output port group, each group can have multiple ports
    ///
    /// \param inputPortIndex If a given place is itself an operation node, this specifies a
    /// port index
    ///
    /// \return An operation place that produces data for this place
    virtual Ptr get_producing_operation(const std::string& inputName, int inputPortIndex) const;

    /// \brief Returns a port that produces data for this place
    virtual Ptr get_producing_port() const;

    /// \brief For operation node returns reference to an input port; applicable if
    /// operation node has only one input port
    ///
    /// \return Input port place or nullptr if not exists
    virtual Ptr get_input_port() const;

    /// \brief For operation node returns reference to an input port with specified index
    ///
    /// \param input_port_index Input port index
    ///
    /// \return Appropriate input port place or nullptr if not exists
    virtual Ptr get_input_port(int input_port_index) const;

    /// \brief For operation node returns reference to an input port with specified name;
    /// applicable if port group has only one input port
    ///
    /// \param input_name Name of port group
    ///
    /// \return Appropriate input port place or nullptr if not exists
    virtual Ptr get_input_port(const std::string& input_name) const;

    /// \brief For operation node returns reference to an input port with specified name and
    /// index
    ///
    /// \param input_name Name of port group, each group can have multiple ports
    ///
    /// \param input_port_index Input port index in a group
    ///
    /// \return Appropriate input port place or nullptr if not exists
    virtual Ptr get_input_port(const std::string& input_name, int input_port_index) const;

    /// \brief For operation node returns reference to an output port; applicable for
    /// operations with only one output port
    ///
    /// \return Appropriate output port place or nullptr if not exists
    virtual Ptr get_output_port() const;

    /// \brief For operation node returns reference to an output port with specified index
    ///
    /// \param output_port_index Output port index
    ///
    /// \return Appropriate output port place or nullptr if not exists
    virtual Ptr get_output_port(int output_port_index) const;

    /// \brief For operation node returns reference to an output port with specified name;
    /// applicable if port group has only one output port
    ///
    /// \param output_name Name of output port group
    ///
    /// \return Appropriate output port place or nullptr if not exists
    virtual Ptr get_output_port(const std::string& output_name) const;

    /// \brief For operation node returns reference to an output port with specified name
    /// and index
    ///
    /// \param output_name Name of output port group, each group can have multiple ports
    ///
    /// \param output_port_index Output port index
    ///
    /// \return Appropriate output port place or nullptr if not exists
    virtual Ptr get_output_port(const std::string& output_name, int output_port_index) const;

    /// \brief Returns all input ports that consume data flows through this place
    virtual std::vector<Place::Ptr> get_consuming_ports() const;

    /// \brief Returns true if this place is input for a model.
    virtual bool is_input() const;

    /// \brief Returns true if this place is output for a model.
    virtual bool is_output() const;

    /// \brief Returns true if another place is the same as this place.
    ///
    /// \param another Another place object
    virtual bool is_equal(const Ptr& another) const;

    /// \brief Returns true if another place points to the same data.
    ///
    /// \note The same data means all places on path: output port -> output edge -> tensor
    /// -> input edge -> input port.
    ///
    /// \param another Another place object
    virtual bool is_equal_data(const Ptr& another) const;
};
}  // namespace frontend
}  // namespace ov
