// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v13 {
/// \brief Multinomial operation creates a sequence of indices of classes sampled from the multinomial distribution.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Multinomial : public Op {
public:
    OPENVINO_OP("Multinomial", "opset13");
    Multinomial() = default;
    /**
     * @brief Multinomial operation creates a sequence of indices of classes sampled from the multinomial distribution.
     *
     * @param probs Input tensor containing at each index poisition probability/log probability of sampling a given
     * class. Any floating-point precision values are allowed.
     * @param num_samples Scalar or 1D tensor with a single value that determines the number of samples to generate per
     * batch. Values should be of an integer type.
     * @param convert_type Data type to which to convert the output class indices. Allowed values: i32/i64
     * @param with_replacement Boolean that determines whether a sampled class can appear more than once in the output.
     * @param log_probs Boolean that determines whether to treat input probabilities as log probabilities.
     * @param global_seed First seed value (key) of Philox random number generation algorithm. (See RandomUniform for
     * details)
     * @param op_seed Second seed value (counter) of Philox random number generation algorithm. (See RandomUniform for
     * details)
     */
    Multinomial(const Output<Node>& input,
                const Output<Node>& num_samples,
                const ov::element::Type_t convert_type,
                const bool with_replacement,
                const bool log_probs,
                const uint64_t global_seed = 0,
                const uint64_t op_seed = 0);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    ov::element::Type_t get_convert_type() const;
    bool get_with_replacement() const;
    bool get_log_probs() const;
    uint64_t get_global_seed() const;
    uint64_t get_op_seed() const;

    void set_convert_type(const ov::element::Type_t convert_type);
    void set_with_replacement(const bool with_replacement);
    void set_log_probs(const bool log_probs);
    void set_global_seed(const uint64_t global_seed);
    void set_op_seed(const uint64_t op_seed);

private:
    ov::element::Type_t m_convert_type;
    bool m_with_replacement;
    bool m_log_probs;
    uint64_t m_global_seed;
    uint64_t m_op_seed;
};
}  // namespace v13
}  // namespace op
}  // namespace ov
