// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _OPENVINO_OP_REG
#    warning "_OPENVINO_OP_REG not defined"
#    define _OPENVINO_OP_REG(x, y)
#endif

// Previous opsets operators
// TODO (ticket: 156877): Add remaining operators from opset15 at the end of opset16 development
_OPENVINO_OP_REG(Parameter, ov::op::v0)
_OPENVINO_OP_REG(Convert, ov::op::v0)
_OPENVINO_OP_REG(ShapeOf, ov::op::v3)

// New operations added in opset16
_OPENVINO_OP_REG(Identity, ov::op::v16)
