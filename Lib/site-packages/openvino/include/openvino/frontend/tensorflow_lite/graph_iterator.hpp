// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/runtime_attribute.hpp"
#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/visibility.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {

/// Abstract representation for an input model graph that gives nodes in topologically sorted order
/// It returns decoders for model inputs and outputs (DecoderBaseTensor objects) and for operation nodes
/// (DecoderBaseOperation objects) DecoderBaseOperation objects for operation nodes must be sorted in topological order
/// from producing nodes to consumer nodes when `get_decoder()` is called. DecoderBaseTensor objects for inputs and
/// outputs must be returned first by `get_decoder()` method. Order of DecoderBaseTensor objects for inputs and outputs
/// defines their order in the original model, i.e. model input index and model output index.
/// For example, calling `get_decoder()` during iterating GraphIterator returns
/// DecoderBaseTensor (for input 0), ..., DecoderBaseTensor (for input n-1),
/// DecoderBaseTensor (for output 0), ..., DecoderBaseTensor (for output m-1),
/// DecoderBaseOperation (for op 1), ..., DecoderBaseOperation (for op k),
/// where n - number of inputs in the model, m - number of outputs in the model k - number of operation nodes.
/// NOTE: constants are ignored and no decoder object is returned for constant.
class TENSORFLOW_LITE_FRONTEND_API GraphIterator : ::ov::RuntimeAttribute {
public:
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

    /// \brief Returns the number of sub-graphs that can be enumerated with get_subgraph
    virtual size_t get_subgraph_size() const = 0;

    /// \brief Returns iterator for a subgraph created on demand
    /// If there is no query for specific sub-graph iterator shouldn't be created
    /// idx should be in range 0..get_subgraph_size()-1
    virtual std::shared_ptr<GraphIterator> get_subgraph(size_t idx) const = 0;

    /// \brief Destructor
    virtual ~GraphIterator();
};

}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
