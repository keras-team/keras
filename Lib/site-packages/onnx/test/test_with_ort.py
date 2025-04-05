# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# This file is for testing ONNX with ONNX Runtime
# Create a general scenario to use ONNX Runtime with ONNX
from __future__ import annotations

import unittest


class TestONNXRuntime(unittest.TestCase):
    def test_with_ort_example(self) -> None:
        try:
            import onnxruntime

            del onnxruntime
        except ImportError:
            raise unittest.SkipTest("onnxruntime not installed") from None

        from numpy import float32, random
        from onnxruntime import InferenceSession
        from onnxruntime.datasets import get_example

        from onnx import checker, load, shape_inference, version_converter

        # get certain example model from ORT using opset 9
        example1 = get_example("sigmoid.onnx")

        # test ONNX functions
        model = load(example1)
        checker.check_model(model)
        checker.check_model(model, full_check=True)
        inferred_model = shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True, data_prop=True
        )
        converted_model = version_converter.convert_version(inferred_model, 10)

        # test ONNX Runtime functions
        sess = InferenceSession(
            converted_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        x = random.random((3, 4, 5))
        x = x.astype(float32)

        sess.run([output_name], {input_name: x})


if __name__ == "__main__":
    unittest.main()
