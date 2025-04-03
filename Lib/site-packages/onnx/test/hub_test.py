# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import glob
import os
import unittest
from os.path import join

import pytest

from onnx import ModelProto, hub


@pytest.mark.skipif(
    "TEST_HUB" not in os.environ or not os.environ["TEST_HUB"],
    reason="Conserving Git LFS quota",
)
class TestModelHub(unittest.TestCase):
    def setUp(self) -> None:
        self.name = "MNIST"
        self.repo = "onnx/models:main"
        self.opset = 7

    def test_force_reload(self) -> None:
        model = hub.load(self.name, self.repo, force_reload=True)
        self.assertIsInstance(model, ModelProto)

        cached_files = list(
            glob.glob(join(hub.get_dir(), "**", "*.onnx"), recursive=True)
        )
        self.assertGreaterEqual(len(cached_files), 1)

    def test_listing_models(self) -> None:
        model_info_list_1 = hub.list_models(self.repo, model="mnist", tags=["vision"])
        model_info_list_2 = hub.list_models(self.repo, tags=["vision"])
        model_info_list_3 = hub.list_models(self.repo)

        self.assertGreater(len(model_info_list_1), 1)
        self.assertGreater(len(model_info_list_2), len(model_info_list_1))
        self.assertGreater(len(model_info_list_3), len(model_info_list_2))

    def test_basic_usage(self) -> None:
        model = hub.load(self.name, self.repo)
        self.assertIsInstance(model, ModelProto)

        cached_files = list(
            glob.glob(join(hub.get_dir(), "**", "*.onnx"), recursive=True)
        )
        self.assertGreaterEqual(len(cached_files), 1)

    def test_custom_cache(self) -> None:
        old_cache = hub.get_dir()
        new_cache = join(old_cache, "custom")
        hub.set_dir(new_cache)

        model = hub.load(self.name, self.repo)
        self.assertIsInstance(model, ModelProto)

        cached_files = list(glob.glob(join(new_cache, "**", "*.onnx"), recursive=True))
        self.assertGreaterEqual(len(cached_files), 1)

        hub.set_dir(old_cache)

    def test_download_with_opset(self) -> None:
        model = hub.load(self.name, self.repo, opset=8)
        self.assertIsInstance(model, ModelProto)

    def test_opset_error(self) -> None:
        self.assertRaises(
            AssertionError, lambda: hub.load(self.name, self.repo, opset=-1)
        )

    def test_manifest_not_found(self) -> None:
        self.assertRaises(
            AssertionError,
            lambda: hub.load(self.name, "onnx/models:unknown", silent=True),
        )

    def test_verify_repo_ref(self) -> None:
        # Not trusted repo:
        verified = hub._verify_repo_ref("mhamilton723/models")
        self.assertFalse(verified)

        # Not trusted repo:
        verified = hub._verify_repo_ref("onnx/models:unknown")
        self.assertFalse(verified)

        # Trusted repo:
        verified = hub._verify_repo_ref(self.repo)
        self.assertTrue(verified)

    def test_get_model_info(self) -> None:
        hub.get_model_info("mnist", self.repo, opset=8)
        hub.get_model_info("mnist", self.repo)
        self.assertRaises(
            AssertionError, lambda: hub.get_model_info("mnist", self.repo, opset=-1)
        )

    def test_download_model_with_test_data(self) -> None:
        directory = hub.download_model_with_test_data("mnist")
        files = os.listdir(directory)
        self.assertIsInstance(directory, str)
        self.assertIn(member="model.onnx", container=files, msg="Onnx model not found")
        self.assertIn(
            member="test_data_set_0", container=files, msg="Test data not found"
        )

    def test_model_with_preprocessing(self) -> None:
        model = hub.load_composite_model(
            "ResNet50-fp32", preprocessing_model="ResNet-preproc"
        )
        self.assertIsInstance(model, ModelProto)


if __name__ == "__main__":
    unittest.main()
