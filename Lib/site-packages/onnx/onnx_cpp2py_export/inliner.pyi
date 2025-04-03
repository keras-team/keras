# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

def inline_local_functions(model: bytes, convert_version: bool) -> bytes:
    """Inlines calls to model-local function in input model and returns it.
    Both input and output are serialized ModelProtos.
    """

def inline_selected_functions(model: bytes, function_ids: list[tuple[str,str]], exclude: bool) -> bytes:
    """Inlines calls to selected model-local functions in input model and returns it.
    Inlines all functions specified in function_ids, unless exclude is true, in which
    case it inlines all functions except those specified in function_ids.
    Both input and output are serialized ModelProtos.
    """
