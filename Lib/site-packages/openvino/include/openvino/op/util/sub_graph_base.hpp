// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/parameter.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Abstract base class for sub-graph based ops, i.e ops that have only one
/// sub-graph
///
class OPENVINO_API SubGraphOp : public MultiSubGraphOp {
public:
    OPENVINO_OP("SubGraphOp", "util", op::util::MultiSubGraphOp);

    virtual const std::shared_ptr<Model>& get_function() const {
        return m_bodies[0];
    };
    virtual void set_function(const std::shared_ptr<Model>& func) {
        m_bodies[0] = func;
    };
    /// \return a reference to the input descriptions.
    const std::vector<std::shared_ptr<InputDescription>>& get_input_descriptions() const {
        return m_input_descriptions[0];
    }
    /// \return a reference to the input descriptions. Can add input descriptions
    /// before
    /// validation.
    std::vector<std::shared_ptr<InputDescription>>& get_input_descriptions() {
        return m_input_descriptions[0];
    }
    /// \return a reference to the output descriptions.
    const std::vector<std::shared_ptr<OutputDescription>>& get_output_descriptions() const {
        return m_output_descriptions[0];
    }
    /// \return a reference to the output descriptions. Can add output descriptions
    /// before
    /// validation.
    std::vector<std::shared_ptr<OutputDescription>>& get_output_descriptions() {
        return m_output_descriptions[0];
    }

    ///
    /// \brief      Indicate that a body parameter comes from slices of a value
    ///
    /// \param      parameter  The parameter to receive the slices
    /// \param      value      The value to be sliced. This will be added as an input to
    ///                        SubGraphOp.
    /// \param      start      First index on axis of the slicing
    /// \param      stride     Stepping of the slice
    /// \param      part_size  Size of the slice on axis
    /// \param      end        The last index on axis of the slicing
    /// \param      axis       The axis to slice along
    ///
    virtual void set_sliced_input(const std::shared_ptr<ov::op::v0::Parameter>& parameter,
                                  const Output<Node>& value,
                                  int64_t start,
                                  int64_t stride,
                                  int64_t part_size,
                                  int64_t end,
                                  int64_t axis);
    ///
    /// \brief      Indicates that a body parameter has an initial value in the first
    ///             iteration and computed value thereafter
    ///
    /// \param[in]  body_parameter    The body parameter
    /// \param      initial_value     Value for the parameter in first iteration. This
    ///                               will be added as an input to Loop.
    /// \param      successive_value  Value for the parameter in successive iterations.
    ///                               The value is what is active in the most recent
    ///                               completed iteration.
    ///
    virtual void set_merged_input(const std::shared_ptr<ov::op::v0::Parameter>& body_parameter,
                                  const Output<Node>& initial_value,
                                  const Output<Node>& successive_value);
    ///
    /// \brief      Indicates that a body parameter has an invariant value during
    ///             iteration that may depend on values computed outside of the
    ///             iteration.
    ///
    /// \param      body_parameter  The body parameter
    /// \param      value           The value supplied as an input to the block
    ///
    virtual void set_invariant_input(const std::shared_ptr<ov::op::v0::Parameter>& body_parameter,
                                     const Output<Node>& value);
    ///
    /// \brief      Gets a value for a particular iteration point
    ///
    /// \param      body_value  The value
    /// \param      iteration   The iteration that supplies the value. Negative values
    ///                         are from the last iteration.
    ///                         Default value -1 (the last iteration).
    ///
    /// \return     The iterator value.
    ///
    virtual Output<Node> get_iter_value(const Output<Node>& body_value, int64_t iteration = -1);
    ///
    /// \brief      Concatenates slices from all iterations
    ///
    /// \param      value      The value supplying slice values from each iteration.
    /// \param      start      First index on axis of the slicing
    /// \param      stride     Stepping of the slice
    /// \param      part_size  Size of the slice on axis
    /// \param      end        The last index on axis of the slicing
    /// \param      axis       The axis to slice along
    ///
    /// \return     The concatenated slices.
    ///
    virtual Output<Node> get_concatenated_slices(const Output<Node>& value,
                                                 int64_t start,
                                                 int64_t stride,
                                                 int64_t part_size,
                                                 int64_t end,
                                                 int64_t axis);

    SubGraphOp(const SubGraphOp&) = delete;
    SubGraphOp(SubGraphOp&&) = default;

    SubGraphOp& operator=(const SubGraphOp&) = delete;
    SubGraphOp& operator=(SubGraphOp&&) = default;

    int64_t get_num_iterations() const {
        return m_num_iterations;
    }

protected:
    int64_t m_num_iterations = -1;  // -1 means infinity for Loop op, inconsistent for TensorIterator

    // Find an input corresponding to value, adding one if necessary.
    Input<Node> input_for_value(const Output<Node>& value);

    SubGraphOp();
    explicit SubGraphOp(const OutputVector& args);

private:
    using MultiSubGraphOp::get_function;

    using MultiSubGraphOp::set_function;
};
using InputDescriptionVector = std::vector<util::SubGraphOp::InputDescription::Ptr>;
using OutputDescriptionVector = std::vector<util::SubGraphOp::OutputDescription::Ptr>;
}  // namespace util
}  // namespace op

}  // namespace ov
