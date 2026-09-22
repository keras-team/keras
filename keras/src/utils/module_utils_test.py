import importlib
import importlib.machinery
import os
import sys
import threading
import types
import uuid
from unittest import mock

from absl.testing import parameterized

from keras.src.testing import test_case
from keras.src.utils import backend_utils
from keras.src.utils import module_utils


# Every test below constructs fresh `LazyModule` / `OrbaxLazyModule` instances
# rather than mutating the module-level singletons in `module_utils`, so cached
# `_available` state cannot leak across tests in the same process.
class LazyModuleTest(test_case.TestCase):
    def _make_namespace_dir(self, subdirs=()):
        tmp = self.get_temp_dir()
        pkg_name = f"keras_ns_{uuid.uuid4().hex[:10]}"
        target = os.path.join(tmp, pkg_name, *subdirs)
        os.makedirs(target, exist_ok=True)
        self.enterContext(mock.patch.dict(sys.modules))
        self.enterContext(mock.patch.dict(sys.path_importer_cache))
        if tmp not in sys.path:
            self.enterContext(mock.patch.object(sys, "path", [tmp, *sys.path]))
        importlib.invalidate_caches()
        return tmp, pkg_name

    def test_namespace_package_not_cached_on_module_attr(self):
        _, pkg_name = self._make_namespace_dir()
        lm = module_utils.LazyModule(pkg_name, pip_name="some-pkg")
        self.assertFalse(lm.available)
        self.assertIsNone(lm.module)
        with mock.patch.object(
            importlib, "import_module", wraps=importlib.import_module
        ) as mocked_import:
            with self.assertRaisesRegex(ImportError, "pip install some-pkg"):
                _ = lm.some_attr
            with self.assertRaisesRegex(ImportError, "pip install some-pkg"):
                lm.initialize()
            mocked_import.assert_not_called()

    def test_namespace_package_does_not_pollute_sys_modules(self):
        _, pkg_name = self._make_namespace_dir()
        lm = module_utils.LazyModule(pkg_name)
        self.assertFalse(lm.available)
        self.assertNotIn(pkg_name, sys.modules)

        _, dotted_pkg = self._make_namespace_dir(subdirs=("io", "gfile"))
        dotted_lm = module_utils.LazyModule(f"{dotted_pkg}.io.gfile")
        self.assertFalse(dotted_lm.available)
        for prefix in (
            dotted_pkg,
            f"{dotted_pkg}.io",
            f"{dotted_pkg}.io.gfile",
        ):
            self.assertNotIn(prefix, sys.modules)

    def test_module_spec_without_search_locations_or_public_attrs_rejected(
        self,
    ):
        pkg_name = f"keras_spec_none_{uuid.uuid4().hex[:10]}"
        fake_mod = types.ModuleType(pkg_name)
        fake_mod.__spec__ = importlib.machinery.ModuleSpec(
            pkg_name, loader=None, origin=None
        )
        self.enterContext(mock.patch.dict(sys.modules, {pkg_name: fake_mod}))
        lm = module_utils.LazyModule(pkg_name, pip_name="some-pkg")
        self.assertFalse(lm.available)
        with self.assertRaisesRegex(ImportError, "pip install some-pkg"):
            _ = lm.random

    def test_namespace_module_with_public_attributes_is_spared(self):
        pkg_name = f"keras_ns_pub_{uuid.uuid4().hex[:10]}"
        fake_mod = types.ModuleType(pkg_name)
        fake_mod.__spec__ = importlib.machinery.ModuleSpec(
            pkg_name, loader=None, origin=None, is_package=True
        )
        fake_mod.public_attr = "usable"
        self.enterContext(mock.patch.dict(sys.modules, {pkg_name: fake_mod}))
        lm = module_utils.LazyModule(pkg_name)
        self.assertTrue(lm.available)
        self.assertEqual(lm.public_attr, "usable")

    def test_non_modulespec_dunder_spec_is_spared(self):
        pkg_name = f"keras_obj_spec_{uuid.uuid4().hex[:10]}"
        fake_mod = types.ModuleType(pkg_name)
        fake_mod.__spec__ = object()
        fake_mod.sentinel = 7
        self.enterContext(mock.patch.dict(sys.modules, {pkg_name: fake_mod}))
        lm = module_utils.LazyModule(pkg_name)
        self.assertTrue(lm.available)
        self.assertEqual(lm.sentinel, 7)

    def test_failed_import_under_namespace_parent_keeps_live_sibling(self):
        tmp, pkg_name = self._make_namespace_dir(subdirs=("other", "empty_sub"))
        init_py = os.path.join(tmp, pkg_name, "other", "__init__.py")
        with open(init_py, "w") as f:
            f.write("VALUE = 123\n")
        importlib.invalidate_caches()

        sibling = importlib.import_module(f"{pkg_name}.other")
        parent_before = sys.modules[pkg_name]
        lm = module_utils.LazyModule(f"{pkg_name}.empty_sub")
        self.assertFalse(lm.available)
        self.assertIs(sys.modules.get(pkg_name), parent_before)
        self.assertIs(getattr(parent_before, "other", None), sibling)
        self.assertIs(importlib.reload(sibling), sibling)

    def test_preexisting_parent_in_sys_modules_stays_same_object(self):
        _, pkg_name = self._make_namespace_dir(subdirs=("missing_child",))
        parent_before = importlib.import_module(pkg_name)
        lm = module_utils.LazyModule(f"{pkg_name}.missing_child")
        self.assertFalse(lm.available)
        self.assertIs(sys.modules.get(pkg_name), parent_before)

    def test_concurrent_import_under_namespace_root_completes(self):
        tmp, pkg_name = self._make_namespace_dir(subdirs=("other", "empty_sub"))
        init_py = os.path.join(tmp, pkg_name, "other", "__init__.py")
        with open(init_py, "w") as f:
            f.write("VALUE = 456\n")
        importlib.invalidate_caches()

        barrier = threading.Barrier(2)
        errors = []

        def _worker():
            try:
                barrier.wait(timeout=5)
                mod = importlib.import_module(f"{pkg_name}.other")
                if getattr(mod, "VALUE", None) != 456:
                    errors.append(AssertionError("Unexpected VALUE"))
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=_worker)
        thread.start()
        barrier.wait(timeout=5)
        lm = module_utils.LazyModule(f"{pkg_name}.empty_sub")
        self.assertFalse(lm.available)
        thread.join(timeout=5)
        self.assertEqual(errors, [])

    def test_in_tf_graph_returns_false_when_preimported_tf_is_namespace(self):
        _, pkg_name = self._make_namespace_dir()
        ns_tf = importlib.import_module(pkg_name)
        fresh_tf_lazy = module_utils.LazyModule("tensorflow")
        self.enterContext(mock.patch.dict(sys.modules, {"tensorflow": ns_tf}))
        self.enterContext(
            mock.patch.object(module_utils, "tensorflow", fresh_tf_lazy)
        )
        self.assertFalse(backend_utils.in_tf_graph())

    @parameterized.named_parameters(
        ("stdlib_package", "json"),
        ("builtin_module", "sys"),
    )
    def test_regular_and_builtin_modules_report_available(self, mod_name):
        lm = module_utils.LazyModule(mod_name)
        self.assertTrue(lm.available)

    def test_missing_module_reports_unavailable_and_raises_import_error(self):
        pkg_name = f"keras_missing_{uuid.uuid4().hex[:10]}"
        lm = module_utils.LazyModule(pkg_name, pip_name="missing-pkg")
        self.assertFalse(lm.available)
        with self.assertRaisesRegex(ImportError, "pip install missing-pkg"):
            _ = lm.some_attr

    def test_bare_module_type_mock_is_spared(self):
        pkg_name = f"keras_mock_mod_{uuid.uuid4().hex[:10]}"
        fake_mod = types.ModuleType(pkg_name)
        fake_mod.sentinel = 42
        self.enterContext(mock.patch.dict(sys.modules, {pkg_name: fake_mod}))
        lm = module_utils.LazyModule(pkg_name)
        self.assertTrue(lm.available)
        self.assertEqual(lm.sentinel, 42)

    def test_magic_mock_is_spared(self):
        pkg_name = f"keras_magic_mock_{uuid.uuid4().hex[:10]}"
        fake_mod = mock.MagicMock()
        fake_mod.sentinel = 99
        self.enterContext(mock.patch.dict(sys.modules, {pkg_name: fake_mod}))
        lm = module_utils.LazyModule(pkg_name)
        self.assertTrue(lm.available)
        self.assertEqual(lm.sentinel, 99)

    def test_orbax_lazy_module_namespace_v1_reports_unavailable(self):
        _, pkg_name = self._make_namespace_dir()
        ns_v1 = importlib.import_module(pkg_name)
        parent_mod = types.ModuleType("orbax.checkpoint")
        parent_mod.v1 = ns_v1
        self.enterContext(
            mock.patch.dict(
                sys.modules,
                {"orbax.checkpoint": parent_mod, "orbax.checkpoint.v1": ns_v1},
            )
        )
        ocp = module_utils.OrbaxLazyModule(
            "orbax.checkpoint.v1",
            pip_name="orbax-checkpoint",
        )
        self.assertFalse(ocp.available)
        self.assertIsNone(ocp.module)
        self.assertNotIn("orbax.checkpoint.v1", sys.modules)
        with self.assertRaisesRegex(
            ImportError, "pip install orbax-checkpoint"
        ):
            _ = ocp.AnyAttr
        with self.assertRaisesRegex(
            ImportError, "pip install orbax-checkpoint"
        ):
            ocp.initialize()

    def test_orbax_lazy_module_missing_v1_attribute_reports_unavailable(self):
        parent_mod = types.ModuleType("orbax.checkpoint")
        parent_mod.other_symbol = True
        self.enterContext(
            mock.patch.dict(sys.modules, {"orbax.checkpoint": parent_mod})
        )
        ocp = module_utils.OrbaxLazyModule(
            "orbax.checkpoint.v1",
            pip_name="orbax-checkpoint",
        )
        self.assertFalse(ocp.available)
        self.assertIsNone(ocp.module)
        with self.assertRaisesRegex(
            ImportError, "pip install orbax-checkpoint"
        ):
            _ = ocp.AnyAttr

    def test_orbax_lazy_module_regular_v1_and_multihost(self):
        parent_mod = types.ModuleType("orbax.checkpoint")
        v1_mod = types.ModuleType("orbax.checkpoint.v1")
        v1_mod.checkpointer = "v1_ckpt"
        multihost_mod = types.ModuleType("orbax.checkpoint.multihost")
        parent_mod.v1 = v1_mod
        parent_mod.multihost = multihost_mod
        self.enterContext(
            mock.patch.dict(
                sys.modules,
                {
                    "orbax.checkpoint": parent_mod,
                    "orbax.checkpoint.v1": v1_mod,
                },
            )
        )
        ocp = module_utils.OrbaxLazyModule(
            "orbax.checkpoint.v1",
            pip_name="orbax-checkpoint",
        )
        self.assertTrue(ocp.available)
        self.assertEqual(ocp.checkpointer, "v1_ckpt")
        self.assertIs(ocp.multihost, multihost_mod)

    def test_dotted_submodule_missing_under_namespace_parent(self):
        _, empty_ns = self._make_namespace_dir()
        missing_child = module_utils.LazyModule(f"{empty_ns}.io.gfile")
        self.assertFalse(missing_child.available)
        self.assertNotIn(empty_ns, sys.modules)

    def test_dotted_submodule_nested_empty_namespace_dirs(self):
        _, nested_empty_ns = self._make_namespace_dir(subdirs=("io", "gfile"))
        nested_ns_child = module_utils.LazyModule(f"{nested_empty_ns}.io.gfile")
        self.assertFalse(nested_ns_child.available)
        self.assertNotIn(nested_empty_ns, sys.modules)

    def test_dotted_submodule_regular_package_under_namespace_parent(self):
        tmp, parent_ns = self._make_namespace_dir(subdirs=("io", "gfile"))
        for rel in (("io",), ("io", "gfile")):
            init_py = os.path.join(tmp, parent_ns, *rel, "__init__.py")
            with open(init_py, "w") as f:
                f.write("MARKER = 1\n")
        importlib.invalidate_caches()
        present_child = module_utils.LazyModule(f"{parent_ns}.io.gfile")
        self.assertTrue(present_child.available)
        self.assertEqual(present_child.MARKER, 1)
