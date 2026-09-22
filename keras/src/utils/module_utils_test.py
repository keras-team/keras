import importlib
import os
import sys
import types
import uuid
from unittest import mock

from absl.testing import parameterized

from keras.src.testing import test_case
from keras.src.utils import module_utils

_MISSING = object()


# Every test below constructs fresh `LazyModule` / `OrbaxLazyModule` instances
# rather than mutating the module-level singletons in `module_utils`, so cached
# `_available` state cannot leak across tests in the same process.
class LazyModuleTest(test_case.TestCase):
    def _make_namespace_dir(self, subdirs=()):
        tmp = self.get_temp_dir()
        pkg_name = f"keras_ns_{uuid.uuid4().hex[:10]}"
        target = os.path.join(tmp, pkg_name, *subdirs)
        os.makedirs(target, exist_ok=True)
        if tmp not in sys.path:
            sys.path.insert(0, tmp)
        importlib.invalidate_caches()

        def _cleanup():
            if tmp in sys.path:
                sys.path.remove(tmp)
            sys.path_importer_cache.pop(tmp, None)
            prefix = f"{pkg_name}."
            for key in list(sys.modules):
                if key == pkg_name or key.startswith(prefix):
                    sys.modules.pop(key, None)
            importlib.invalidate_caches()

        self.addCleanup(_cleanup)
        return tmp, pkg_name

    def _plant_module(self, name, obj):
        prev = sys.modules.get(name, _MISSING)
        sys.modules[name] = obj

        def _restore():
            if prev is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev

        self.addCleanup(_restore)

    def test_namespace_package_reports_unavailable(self):
        _, pkg_name = self._make_namespace_dir()
        lm = module_utils.LazyModule(pkg_name)
        self.assertFalse(lm.available)

    def test_namespace_package_getattr_raises_import_error(self):
        _, pkg_name = self._make_namespace_dir()
        lm = module_utils.LazyModule(pkg_name, pip_name="some-pkg")
        with self.assertRaisesRegex(ImportError, "pip install some-pkg"):
            _ = lm.some_attr

    def test_namespace_package_not_cached_on_module_attr(self):
        _, pkg_name = self._make_namespace_dir()
        lm = module_utils.LazyModule(pkg_name, pip_name="some-pkg")
        self.assertFalse(lm.available)
        self.assertIsNone(lm.module)
        with self.assertRaisesRegex(ImportError, "pip install some-pkg"):
            _ = lm.some_attr

    def test_namespace_package_does_not_pollute_sys_modules(self):
        _, pkg_name = self._make_namespace_dir()
        lm = module_utils.LazyModule(pkg_name)
        self.assertFalse(lm.available)
        self.assertNotIn(pkg_name, sys.modules)

        dotted_lm = module_utils.LazyModule(f"{pkg_name}.io.gfile")
        self.assertFalse(dotted_lm.available)
        self.assertNotIn(pkg_name, sys.modules)
        self.assertNotIn(f"{pkg_name}.io", sys.modules)

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
        self._plant_module(pkg_name, fake_mod)
        lm = module_utils.LazyModule(pkg_name)
        self.assertTrue(lm.available)
        self.assertEqual(lm.sentinel, 42)

    def test_magic_mock_is_spared(self):
        pkg_name = f"keras_magic_mock_{uuid.uuid4().hex[:10]}"
        fake_mod = mock.MagicMock()
        fake_mod.sentinel = 99
        self._plant_module(pkg_name, fake_mod)
        lm = module_utils.LazyModule(pkg_name)
        self.assertTrue(lm.available)
        self.assertEqual(lm.sentinel, 99)

    def test_orbax_lazy_module_namespace_package_reports_unavailable(self):
        _, pkg_name = self._make_namespace_dir()
        real_ns_mod = importlib.import_module(pkg_name)
        self._plant_module("orbax.checkpoint", real_ns_mod)
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

    def test_dotted_submodule_under_namespace_parent_characterization(self):
        # Characterization test (no mutation): pins the CPython import behavior
        # that (a) a missing child or nested empty namespace child under a
        # namespace parent reports available is False (e.g. tensorflow.io.gfile
        # under an empty tensorflow/ dir), while (b) a real regular package
        # under a namespace parent resolves with a real spec.origin ->
        # available is True (e.g. healthy orbax.checkpoint under orbax/).
        _, empty_ns = self._make_namespace_dir()
        missing_child = module_utils.LazyModule(f"{empty_ns}.io.gfile")
        self.assertFalse(missing_child.available)

        _, nested_empty_ns = self._make_namespace_dir(subdirs=("io", "gfile"))
        nested_ns_child = module_utils.LazyModule(f"{nested_empty_ns}.io.gfile")
        self.assertFalse(nested_ns_child.available)
        self.assertNotIn(nested_empty_ns, sys.modules)

        tmp, parent_ns = self._make_namespace_dir(subdirs=("io", "gfile"))
        for rel in (("io",), ("io", "gfile")):
            init_py = os.path.join(tmp, parent_ns, *rel, "__init__.py")
            with open(init_py, "w") as f:
                f.write("MARKER = 1\n")
        importlib.invalidate_caches()
        present_child = module_utils.LazyModule(f"{parent_ns}.io.gfile")
        self.assertTrue(present_child.available)
        self.assertEqual(present_child.MARKER, 1)
