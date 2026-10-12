import importlib

import numpy as np

from keras.src.testing import test_case
from keras.src.utils import module_utils


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
