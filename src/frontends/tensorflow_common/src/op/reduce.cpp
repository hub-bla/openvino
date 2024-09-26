// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tensor_iterator.hpp"
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
                       "Sum",
                       "MEAN",
                       "REDUCE_ALL",
                       "REDUCE_ANY",
                       "REDUCE_MAX",
                       "REDUCE_MIN",
                       "REDUCE_PROD",
                       "SUM"});
    auto input = node.get_input(0);
    auto axis = node.get_input(1);
    auto keep_dims = node.get_attribute<bool>("keep_dims", false);
    auto reduce_op = make_shared<T>(input, axis, keep_dims);
    set_node_name(node.get_name(), reduce_op);
    return {reduce_op};
}

OutputVector translate_prod_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Prod"}, true);
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


        auto const_one = make_shared<v0::Constant>(element::i64, Shape{}, 1);
        auto const_zero = make_shared<v0::Constant>(element::i64, Shape{}, 0);
        vector<int64_t> dims;
        get_const_input(node, 1, &dims);

        Output<Node> real_res, imag_res;

        for (int64_t ind = 0; ind < static_cast<int64_t>(dims.size()); ++ind){
            auto temp_axis = make_shared<v0::Constant>(element::i64, Shape{}, dims[ind]);
            auto temp_reduce = make_shared<v1::ReduceProd>(real_part, temp_axis);

            auto temp_reduce_shape = make_shared<v0::ShapeOf>(temp_reduce);

            //c
            auto real_param = make_shared<v0::Parameter>(real_part->get_element_type(), temp_reduce->get_shape());
            //d
            auto imag_param = make_shared<v0::Parameter>(real_part->get_element_type(), temp_reduce->get_shape());

            //a
            auto result_real_param = make_shared<v0::Parameter>(real_part->get_element_type(), temp_reduce->get_shape());
            //b
            auto result_imag_param = make_shared<v0::Parameter>(real_part->get_element_type(), temp_reduce->get_shape());

            auto result_real_init = make_shared<v3::Broadcast>(const_one, temp_reduce_shape);
            auto result_imag_init = make_shared<v3::Broadcast>(const_zero, temp_reduce_shape);

            // (ac - bd) + i(ad + bc)

            auto ac = make_shared<v1::Multiply>(result_real_param, real_param);

            auto bd = make_shared<v1::Multiply>(result_imag_param, imag_param);

            auto ad = make_shared<v1::Multiply>(result_real_param, imag_param);

            auto bc = make_shared<v1::Multiply>(result_imag_param, real_param);

            auto new_real = make_shared<v1::Subtract>(ac, bd);
            auto new_imag = make_shared<v1::Add>(ad, bc);

            auto iterator = make_shared<v0::TensorIterator>();

            iterator->set_sliced_input(real_param, real_part, 0, 1, -1, -1, dims[ind]);
            iterator->set_sliced_input(imag_param, imag_part, 0, 1, -1, -1, dims[ind]);
            iterator->set_merged_input(result_real_param, result_real_init, new_real);
            iterator->set_merged_input(result_imag_param, result_imag_init, new_imag);

            auto body =
                    std::make_shared<Model>(OutputVector{new_real, new_imag}, ParameterVector{real_param, imag_param, result_real_param, result_imag_param});

            iterator->set_function(body);

            real_res = iterator->get_iter_value(new_real, -1);
            imag_res = iterator->get_iter_value(new_imag, -1);
        }

        auto real_unsqueeze = make_shared<v0::Unsqueeze>(real_res, minus_one);
        auto imag_unsqueeze = make_shared<v0::Unsqueeze>(imag_res, minus_one);

        auto concat_result = make_shared<v0::Concat>(OutputVector{real_unsqueeze, imag_unsqueeze}, -1);
        auto complex_result = make_shared<ComplexTypeMark>(concat_result, complex_part_type);
        return {complex_result};
    }

    auto prod_result = make_shared<v1::ReduceProd>(input, axis, keep_dims);
    set_node_name(node.get_name(), prod_result);
    return {prod_result};
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
