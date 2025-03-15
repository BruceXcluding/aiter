/* SPDX-License-Identifier: MIT
   Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
*/
#include "rocm_ops.hpp"
#include "moe_op.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      MOE_OP_PYBIND;
}