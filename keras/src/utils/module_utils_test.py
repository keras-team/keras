import importlib
import os
import re
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src.backend import config
from keras.src.testing import test_case
from keras.src.utils import backend_utils
from keras.src.utils import module_utils

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
CONFTEST_PATH = os.path.join(REPO_ROOT, "conftest.py")
ALL_BACKENDS = sorted(
    list(config.BUILT_IN_BACKENDS) + list(config.PLUGGABLE_BACKENDS)
)


def _exact(message):
    """Returns a regex matching `message` as the whole exception text."""
    return f"^{re.escape(message)}$"


class LazyModuleTest(test_case.TestCase):
    def test_missing_required_attr(self):
        lm = module_utils.LazyModule("numpy", required_attr="dummy_attribute")
        self.assertFalse(lm.available)
        with self.assertRaisesRegex(ImportError, "numpy"):
            _ = lm.ones

    def test_default_required_attr(self):
        lm = module_utils.LazyModule("numpy")
        self.assertEqual(lm.required_attr, "__version__")
        self.assertTrue(lm.available)
        self.assertIs(lm.ones, np.ones)

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


class UnofficialBackendTest(test_case.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = self.get_temp_dir()

        self.dummy_pkg_dir = os.path.join(self.temp_dir, "keras_dummy")
        os.makedirs(self.dummy_pkg_dir)
        with open(os.path.join(self.dummy_pkg_dir, "__init__.py"), "w"):
            pass

        self.dummy_src_dir = os.path.join(self.dummy_pkg_dir, "src")
        os.makedirs(self.dummy_src_dir)
        with open(os.path.join(self.dummy_src_dir, "__init__.py"), "w") as f:
            f.write(
                "SENTINEL = 'dummy_backend'\n"
                "class Variable: pass\n"
                "class name_scope:\n"
                "    def __init__(self, *args, **kwargs): pass\n"
                "    def __enter__(self): return self\n"
                "    def __exit__(self, *args, **kwargs): pass\n"
                "class device_scope:\n"
                "    def __init__(self, *args, **kwargs): pass\n"
                "    def __enter__(self): return self\n"
                "    def __exit__(self, *args, **kwargs): pass\n"
                "class _DummyAttr:\n"
                "    def __getattr__(self, name): return _DummyAttr()\n"
                "    def __call__(self, *args, **kwargs): return _DummyAttr()\n"
                "random = _DummyAttr()\n"
                "distribution_lib = _DummyAttr()\n"
                "ops = _DummyAttr()\n"
                "numpy = _DummyAttr()\n"
                "nn = _DummyAttr()\n"
                "SUPPORTS_GRADIENT = True\n"
                "SUPPORTS_COMPLEX_DTYPES = True\n"
                "SUPPORTS_RAGGED_TENSORS = True\n"
                "SUPPORTS_SPARSE_TENSORS = True\n"
                "IS_THREAD_SAFE = False\n"
                "__all__ = [\n"
                "    'SENTINEL', 'Variable', 'name_scope', 'device_scope',\n"
                "    'random', 'distribution_lib', 'ops', 'numpy', 'nn',\n"
                "    'SUPPORTS_GRADIENT', 'SUPPORTS_COMPLEX_DTYPES',\n"
                "    'SUPPORTS_RAGGED_TENSORS', 'SUPPORTS_SPARSE_TENSORS',\n"
                "    'IS_THREAD_SAFE',\n"
                "]\n"
            )

        with open(os.path.join(self.dummy_src_dir, "trainer.py"), "w") as f:
            f.write("class Trainer: pass\n")

        with open(os.path.join(self.temp_dir, "excluded_tests.txt"), "w") as f:
            f.write("dummy_test.py::test_dummy\n")

    def _set_allow_unofficial(self, value):
        previous = os.environ.get("KERAS_ALLOW_UNOFFICIAL_BACKEND")

        def restore():
            if previous is None:
                os.environ.pop("KERAS_ALLOW_UNOFFICIAL_BACKEND", None)
            else:
                os.environ["KERAS_ALLOW_UNOFFICIAL_BACKEND"] = previous

        self.addCleanup(restore)
        if value is None:
            os.environ.pop("KERAS_ALLOW_UNOFFICIAL_BACKEND", None)
        else:
            os.environ["KERAS_ALLOW_UNOFFICIAL_BACKEND"] = value

    def _run_subprocess(self, code, **extra_env):
        env = os.environ.copy()
        env.pop("KERAS_ALLOW_UNOFFICIAL_BACKEND", None)
        env.update(extra_env)
        pythonpath = f"{self.temp_dir}{os.pathsep}{REPO_ROOT}"
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{pythonpath}{os.pathsep}{existing}" if existing else pythonpath
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"{proc.stdout}\n{proc.stderr}",
        )

    @parameterized.named_parameters(
        ("unset", None),
        ("empty", ""),
        ("zero", "0"),
        ("false", "false"),
        ("no", "no"),
    )
    def test_unofficial_backend_flag_off(self, val):
        self._set_allow_unofficial(val)
        # Messages are unchanged for identifiers and non-identifiers alike.
        for name in ("dummy", "foo-bar"):
            with self.assertRaisesRegex(
                ValueError, _exact(f"Unsupported backend : {name}")
            ):
                module_utils.get_pluggable_backend_module(backend=name)
            with self.assertRaisesRegex(
                ValueError,
                _exact(
                    f"Available backends are ({ALL_BACKENDS}). "
                    f"Received: backend={name}"
                ),
            ):
                backend_utils.DynamicBackend().set_backend(name)

    @parameterized.named_parameters(
        ("true", "true"),
        ("one", "1"),
        ("yes_upper", "YES"),
    )
    def test_unofficial_backend_flag_on(self, val):
        code = (
            "import keras; "
            "assert keras.config.backend() == 'dummy'\n"
            "from keras.src.utils.module_utils import "
            "get_pluggable_backend_module; "
            "assert get_pluggable_backend_module('src').SENTINEL "
            "== 'dummy_backend'\n"
            "from keras.src.utils.backend_utils import DynamicBackend; "
            "db = DynamicBackend('numpy'); "
            "db.set_backend('dummy'); "
            "assert db.SENTINEL == 'dummy_backend'\n"
        )
        self._run_subprocess(
            code, KERAS_ALLOW_UNOFFICIAL_BACKEND=val, KERAS_BACKEND="dummy"
        )

    @parameterized.named_parameters(
        ("dotted", "foo.bar"),
        ("empty", ""),
        ("leading_digit", "123"),
        ("hyphen", "foo-bar"),
    )
    def test_unofficial_backend_invalid_identifier(self, invalid_name):
        self._set_allow_unofficial("true")
        self.assertFalse(config.backend_is_pluggable(invalid_name))

        hint = (
            ". When KERAS_ALLOW_UNOFFICIAL_BACKEND is set, the backend "
            "name must be a valid Python identifier."
        )
        # An empty name means "use the current backend" in
        # `get_pluggable_backend_module`, so it is only checked in
        # `set_backend`.
        if invalid_name:
            with self.assertRaisesRegex(
                ValueError,
                _exact(f"Unsupported backend : {invalid_name}{hint}"),
            ):
                module_utils.get_pluggable_backend_module(backend=invalid_name)
        with self.assertRaisesRegex(
            ValueError,
            _exact(
                f"Available backends are ({ALL_BACKENDS}). "
                f"Received: backend={invalid_name}{hint}"
            ),
        ):
            backend_utils.DynamicBackend().set_backend(invalid_name)

    def test_unofficial_backend_missing_package(self):
        self._set_allow_unofficial("true")
        with self.assertRaisesRegex(
            ImportError,
            _exact("The keras_dummy_missing package could not be imported."),
        ):
            module_utils.get_pluggable_backend_module(backend="dummy_missing")
        self.assertIsNone(
            module_utils.get_pluggable_backend_module(
                backend="dummy_missing", allow_missing=True
            )
        )

    def test_builtin_backend_not_treated_as_pluggable_when_flag_on(self):
        self._set_allow_unofficial("true")
        self.assertFalse(config.backend_is_pluggable("jax"))
        self.assertFalse(config.backend_is_pluggable("tensorflow"))
        with self.assertRaisesRegex(
            ValueError, _exact("Unsupported backend : jax")
        ):
            module_utils.get_pluggable_backend_module(backend="jax")

    def test_official_pluggable_backend_returns_true(self):
        self.assertTrue(config.backend_is_pluggable("mlx"))

    def test_official_pluggable_backend_missing(self):
        # Simulate a missing `keras_mlx` whether or not it is installed.
        with mock.patch.object(
            importlib, "import_module", side_effect=ImportError
        ):
            with self.assertRaisesRegex(
                ImportError,
                _exact(
                    "The mlx backend is not installed. "
                    "You can install it via `pip install keras-mlx`"
                ),
            ):
                module_utils.get_pluggable_backend_module(backend="mlx")

    def test_backend_is_pluggable_unhashable_type(self):
        self.assertFalse(config.backend_is_pluggable(["x"]))

    @pytest.mark.skipif(
        not os.path.exists(CONFTEST_PATH),
        reason="Requires source checkout conftest.py.",
    )
    def test_conftest_pytest_collection_modifyitems(self):
        code = (
            "import importlib.util\n"
            "import os\n"
            "repo_root = os.environ['TEST_REPO_ROOT']\n"
            "conftest_path = os.path.join(repo_root, 'conftest.py')\n"
            "spec = importlib.util.spec_from_file_location("
            "'conftest', conftest_path)\n"
            "conftest = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(conftest)\n"
            "class _Item:\n"
            "    def __init__(self):\n"
            "        self.keywords = {}\n"
            "        self.nodeid = 'dummy_test.py::test_dummy'\n"
            "        self.markers = []\n"
            "        self.add_marker = self.markers.append\n"
            "item = _Item()\n"
            "conftest.pytest_collection_modifyitems(None, [item])\n"
            "assert len(item.markers) == 1\n"
        )
        self._run_subprocess(
            code,
            KERAS_ALLOW_UNOFFICIAL_BACKEND="true",
            KERAS_BACKEND="dummy",
            TEST_REPO_ROOT=REPO_ROOT,
        )
