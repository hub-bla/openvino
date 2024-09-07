// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

template <typename T>
OutputVector translate_direct_reduce_op(const NodeContext& node) {
    default_op_checks(node,
                      2,
                      {"Any",
                       "All",
                       "EuclideanNorm",
                       "Max",
                       "Mean",
                       "Min",
//                       "Prod",
                       "Sum",
                       "MEAN",
                       "REDUCE_ALL",
                       "REDUCE_ANY",
                       "REDUCE_MAX",
                       "REDUCE_MIN",
//                       "REDUCE_PROD",
                       "SUM"});
    auto input = node.get_input(0);
    auto axis = node.get_input(1);
    auto keep_dims = node.get_attribute<bool>("keep_dims", false);
    auto reduce_op = make_shared<T>(input, axis, keep_dims);
    set_node_name(node.get_name(), reduce_op);
    return {reduce_op};
}

OutputVector translate_prod_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Prod", "REDUCE_PROD"}, true);
    auto input = node.get_input(0);
    auto axis = node.get_input(1);
    auto keep_dims = node.get_attribute<bool>("keep_dims", false);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());

    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        input = complex_type_mark->input_value(0);


        auto gather_index_real = make_shared<v0::Constant>(element::i64, Shape{}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(element::i64, Shape{}, 1);
        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);


        auto real_part = make_shared<v8::Gather>(input, gather_index_real, minus_one);
        auto imag_part = make_shared<v8::Gather>(input, gather_index_imag, minus_one);

        auto const_two = create_same_type_const_scalar<float>(real_part, 2.0f);

        // r = sqrt(a^2 + b^2)
        // theta = arctan(b/a)
        // exponential_form = r * e^(theta * j)

        // conversion

        auto sum_sq = std::make_shared<v1::Add>(std::make_shared<v1::Power>(real_part, const_two),
                                                std::make_shared<v1::Power>(imag_part, const_two));
        auto r = std::make_shared<v0::Sqrt>(sum_sq);

        auto theta = make_shared<v0::Atan>(make_shared<v1::Divide>(imag_part, real_part));



        // formula = e^( j * k ) * (r_0 * r_1 * ... * r_n)
        // k = theta_0 + theta_1 + ... + theta_n

        auto new_theta = make_shared<v1::ReduceSum>(theta, axis, keep_dims);

        auto new_r = make_shared<v1::ReduceProd>(r, axis, keep_dims);

        // convert back to the rectangular form
        auto sin = make_shared<v0::Sin>(new_theta);
        auto cos = make_shared<v0::Cos>(new_theta);

        auto new_real = make_shared<v1::Multiply>(cos, r);
        auto new_imag = make_shared<v1::Multiply>(sin, r);

        auto real_unsqueeze = make_shared<v0::Unsqueeze>(new_real, minus_one);
        auto imag_unsqueeze = make_shared<v0::Unsqueeze>(new_imag, minus_one);

        auto concat_result = make_shared<v0::Concat>(OutputVector{real_unsqueeze, imag_unsqueeze}, -1);
        set_node_name(node.get_name(), concat_result);

        auto complex_result = make_shared<ComplexTypeMark>(concat_result, complex_part_type);
        return {complex_result};
    }

    auto result = make_shared<v1::ReduceProd>(input, axis, keep_dims);
    return {result};
}

template OutputVector translate_direct_reduce_op<v1::ReduceLogicalOr>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceLogicalAnd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMax>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMean>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceMin>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceProd>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v1::ReduceSum>(const NodeContext& node);
template OutputVector translate_direct_reduce_op<v4::ReduceL2>(const NodeContext& node);
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
