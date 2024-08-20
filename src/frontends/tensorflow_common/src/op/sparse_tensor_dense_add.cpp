// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_sparse_tensor_dense_add_op(const NodeContext& node) {
    default_op_checks(node, 4, {"SparseTensorDenseAdd"});
    auto a_indices = node.get_input(0);
    auto a_values = node.get_input(1);
    auto a_shape = node.get_input(2);
    auto b = node.get_input(3);

    // create dense tensor
    auto zero_const = create_same_type_const_scalar<int32_t>(a_values, 0);
    ov::Output<ov::Node> a = make_shared<v3::Broadcast>(zero_const, a_shape);
    a = make_shared<v15::ScatterNDUpdate>(a, a_indices, a_values);
    auto res = make_shared<v1::Add>(a, b);
    set_node_name(node.get_name(), res);
    return {res};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
