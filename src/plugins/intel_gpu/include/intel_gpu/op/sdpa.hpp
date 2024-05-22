// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class SDPA : public ov::op::v13::ScaledDotProductAttention {
public:
    OPENVINO_OP("SDPA", "gpu_opset");

    SDPA() = default;

    SDPA(const ov::Output<Node>& Q,
         const ov::Output<Node>& K,
         const ov::Output<Node>& V,
         const std::vector<int64_t>& order_q,
         const std::vector<int64_t>& order_k,
         const std::vector<int64_t>& order_v,
         const std::vector<int64_t>& order_out,
         const bool is_causal,
         const ov::element::Type output_type = ov::element::undefined);

    SDPA(const ov::Output<Node>& Q,
         const ov::Output<Node>& K,
         const ov::Output<Node>& V,
         const ov::Output<Node>& attn_mask,
         const std::vector<int64_t>& order_q,
         const std::vector<int64_t>& order_k,
         const std::vector<int64_t>& order_v,
         const std::vector<int64_t>& order_out,
         const bool is_causal,
         const ov::element::Type output_type = ov::element::undefined);

    SDPA(const ov::Output<Node>& Q,
         const ov::Output<Node>& K,
         const ov::Output<Node>& V,
         const ov::Output<Node>& attn_mask,
         const ov::Output<Node>& scale,
         const std::vector<int64_t>& order_q,
         const std::vector<int64_t>& order_k,
         const std::vector<int64_t>& order_v,
         const std::vector<int64_t>& order_out,
         const bool is_causal,
         const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool get_causal() const { return m_is_causal; }

    std::vector<int64_t> get_input0_transpose_order() const { return m_order_q; }
    std::vector<int64_t> get_input1_transpose_order() const { return m_order_k; }
    std::vector<int64_t> get_input2_transpose_order() const { return m_order_v; }
    std::vector<int64_t> get_output_transpose_order() const { return m_order_out; }
    ov::element::Type get_output_type() const { return m_output_type; }

    static std::vector<int64_t> default_order(size_t rank) {
        std::vector<int64_t> order(rank);
        std::iota(order.begin(), order.end(), 0);
        return order;
    }

protected:
    std::vector<int64_t> m_order_q;
    std::vector<int64_t> m_order_k;
    std::vector<int64_t> m_order_v;
    std::vector<int64_t> m_order_out;
    bool m_is_causal;
    ov::element::Type m_output_type;
};

std::vector<ov::PartialShape> shape_infer(const SDPA* op,
                                          std::vector<ov::PartialShape> input_shapes,
                                          const std::vector<int64_t>& order_q,
                                          const std::vector<int64_t>& order_k,
                                          const std::vector<int64_t>& order_v,
                                          const std::vector<int64_t>& order_out);


}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
