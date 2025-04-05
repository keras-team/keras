// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "onnx/common/constants.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {

// ONNX (model-local) function identifiers are a tuple (domain, op, overload).
// The pair (domain, op) represents a specification of a function, while
// overload is used to disambiguate between multiple (specialized) implementations of
// the same specification. Overload is optional and can be empty.
// Multiple overloads may be used to distinguish implementations specialized
// for a specific type or rank of input tensors or for specific attribute values.

// A single string representation of (domain, op)
using FunctionSpecId = std::string;

// A single string representation of (domain, op, overload)
using FunctionImplId = std::string;

FunctionImplId GetFunctionImplId(const std::string& domain, const std::string& op, const std::string& overload) {
  if (overload.empty())
    return NormalizeDomain(domain) + "::" + op;
  return NormalizeDomain(domain) + "::" + op + "::" + overload;
}

FunctionImplId GetFunctionImplId(const FunctionProto& function) {
  return GetFunctionImplId(function.domain(), function.name(), function.overload());
}

FunctionImplId GetCalleeId(const NodeProto& node) {
  return GetFunctionImplId(node.domain(), node.op_type(), node.overload());
}

} // namespace ONNX_NAMESPACE
