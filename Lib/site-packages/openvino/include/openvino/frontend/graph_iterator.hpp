// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/any.hpp"
#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/visibility.hpp"

namespace ov {
namespace frontend {

/// Abstract representation for an input model graph that gives nodes in topologically sorted order
class FRONTEND_API GraphIterator : ::ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("Variant::GraphIterator", "0", RuntimeAttribute);

    using Ptr = std::shared_ptr<GraphIterator>;

    /// \brief Get a number of operation nodes in the graph
    virtual size_t size() const = 0;

    /// \brief Set iterator to the start position
    virtual void reset() = 0;

    /// \brief Move to the next node in the graph
    virtual void next() = 0;

    /// \brief Returns true if iterator goes out of the range of available nodes
    virtual bool is_end() const = 0;

    /// \brief Return a pointer to a decoder of the current node
    virtual std::shared_ptr<DecoderBase> get_decoder() const = 0;

    /// \brief Checks if the main model graph contains a function of the requested name in the library
    /// Returns GraphIterator to this function and nullptr, if it does not exist
    virtual std::shared_ptr<GraphIterator> get_body_graph_iterator(const std::string& func_name) const = 0;

    /// \brief Returns a vector of input names in the original order
    virtual std::vector<std::string> get_input_names() const = 0;

    /// \brief Returns a vector of output names in the original order
    virtual std::vector<std::string> get_output_names() const = 0;

    /// \brief Returns a map from internal tensor name to (user-defined) external name for inputs
    virtual std::map<std::string, std::string> get_input_names_map() const;

    /// \brief Returns a map from internal tensor name to (user-defined) external name for outputs
    virtual std::map<std::string, std::string> get_output_names_map() const;
};

}  // namespace frontend
}  // namespace ov
