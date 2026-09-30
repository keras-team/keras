import importlib
import importlib.machinery
import os
import sys
import types
import uuid
from unittest import mock

from keras.src.testing import test_case
from keras.src.utils import backend_utils
from keras.src.utils import module_utils
from keras.src.utils import rng_utils


class LazyModuleTest(test_case.TestCase):
    def _make_package(self, source):
        tmp = self.get_temp_dir()
        pkg_name = f"keras_pkg_{uuid.uuid4().hex[:10]}"
        os.makedirs(os.path.join(tmp, pkg_name), exist_ok=True)
        with open(os.path.join(tmp, pkg_name, "__init__.py"), "w") as f:
            f.write(source)
        self.enterContext(mock.patch.dict(sys.modules))
        self.enterContext(mock.patch.dict(sys.path_importer_cache))
        self.enterContext(mock.patch.object(sys, "path", [tmp, *sys.path]))
        importlib.invalidate_caches()
        return pkg_name

    def test_missing_required_attr_reports_unavailable_and_raises(self):
        pkg_name = self._make_package("a = 1\n")
        lm = module_utils.LazyModule(pkg_name, required_attr="missing_attr")
        self.assertFalse(lm.available)
        with self.assertRaisesRegex(ImportError, pkg_name):
            _ = lm.some_attr

    def test_present_required_attr_succeeds(self):
        pkg_name = self._make_package("present_attr = 1\n")
        lm = module_utils.LazyModule(pkg_name, required_attr="present_attr")
        self.assertTrue(lm.available)
        self.assertEqual(lm.present_attr, 1)

    def test_available_false_caches_but_allows_retry(self):
        lm = module_utils.LazyModule("keras_retry_pkg")
        self.assertFalse(lm.available)

        fake_mod = types.ModuleType("keras_retry_pkg")
        fake_mod.some_attr = 42
        self.enterContext(
            mock.patch.dict(sys.modules, {"keras_retry_pkg": fake_mod})
        )

        self.assertEqual(lm.some_attr, 42)
        self.assertFalse(lm.available)

    def test_tensorflow_namespace_package_reports_unavailable(self):
        pkg_name = "tensorflow"
        fake_mod = types.ModuleType(pkg_name)
        fake_mod.__spec__ = importlib.machinery.ModuleSpec(
            pkg_name, loader=None, origin=None
        )
        self.enterContext(mock.patch.dict(sys.modules, {pkg_name: fake_mod}))

        self.enterContext(
            mock.patch.object(module_utils.tensorflow, "_available", None)
        )
        self.enterContext(
            mock.patch.object(module_utils.tensorflow, "module", None)
        )

        self.assertFalse(module_utils.tensorflow.available)
        self.assertFalse(backend_utils.in_tf_graph())
        rng_utils.set_random_seed(0)

    def test_configured_required_attrs_are_present_on_installed_modules(self):
        instances = [
            v
            for v in vars(module_utils).values()
            if isinstance(v, module_utils.LazyModule)
        ]
        for inst in instances:
            # Importing torch_xla outside a torch/XLA job is not safe.
            if inst.name == "torch_xla":
                continue
            try:
                if isinstance(inst, module_utils.OrbaxLazyModule):
                    mod = importlib.import_module("orbax.checkpoint").v1
                else:
                    mod = importlib.import_module(inst.name)
            except (ImportError, AttributeError):
                continue
            self.assertTrue(hasattr(mod, inst.required_attr), inst.name)

    def test_in_tf_graph_true_for_live_tf_with_executing_eagerly(self):
        mock_tf = types.ModuleType("tensorflow")
        mock_tf.executing_eagerly = lambda: False
        self.enterContext(mock.patch.dict(sys.modules, {"tensorflow": mock_tf}))
        self.assertTrue(backend_utils.in_tf_graph())

    def _orbax(self, parent_mod, required_attr=None):
        modules = {"orbax.checkpoint": parent_mod}
        if hasattr(parent_mod, "v1"):
            modules["orbax.checkpoint.v1"] = parent_mod.v1
        self.enterContext(mock.patch.dict(sys.modules, modules))
        return module_utils.OrbaxLazyModule(
            "orbax.checkpoint.v1",
            pip_name="orbax-checkpoint",
            required_attr=required_attr,
        )

    def _assert_orbax_unavailable(self, ocp):
        self.assertFalse(ocp.available)
        self.assertIsNone(ocp.module)
        with self.assertRaisesRegex(ImportError, "orbax-checkpoint"):
            _ = ocp.AnyAttr

    def test_orbax_unavailable_if_v1_lacks_training(self):
        parent_mod = types.ModuleType("orbax.checkpoint")
        parent_mod.v1 = types.ModuleType("orbax.checkpoint.v1")
        self._assert_orbax_unavailable(
            self._orbax(parent_mod, required_attr="training")
        )

    def test_orbax_unavailable_if_parent_lacks_v1(self):
        parent_mod = types.ModuleType("orbax.checkpoint")
        parent_mod.other_symbol = True
        self._assert_orbax_unavailable(self._orbax(parent_mod))

    def test_orbax_lazy_module_regular_v1_and_multihost(self):
        parent_mod = types.ModuleType("orbax.checkpoint")
        v1_mod = types.ModuleType("orbax.checkpoint.v1")
        v1_mod.checkpointer = "v1_ckpt"
        multihost_mod = types.ModuleType("orbax.checkpoint.multihost")
        parent_mod.v1 = v1_mod
        parent_mod.multihost = multihost_mod
        ocp = self._orbax(parent_mod)
        self.assertTrue(ocp.available)
        self.assertEqual(ocp.checkpointer, "v1_ckpt")
        self.assertIs(ocp.multihost, multihost_mod)
