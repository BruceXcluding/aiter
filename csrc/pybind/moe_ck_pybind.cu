// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_ck.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ck_moe", &ck_moe,
          py::arg("hidden_states"), py::arg("w1"), py::arg("w2"),
          py::arg("topk_weights"), py::arg("topk_ids"),
          py::arg("w1_scale") = std::nullopt, py::arg("w2_scale") = std::nullopt,
          py::arg("a1_scale") = std::nullopt, py::arg("a2_scale") = std::nullopt,
          py::arg("block_m") = 32);

    m.def("ck_moe_stage1", &ck_moe_stage1,
          py::arg("hidden_states"),
          py::arg("w1"),
          py::arg("w2"),
          py::arg("sorted_token_ids"),
          py::arg("sorted_expert_ids"),
          py::arg("num_valid_ids"),
          py::arg("out"),
          py::arg("w1_scale") = std::nullopt,
          py::arg("a1_scale") = std::nullopt);
}
