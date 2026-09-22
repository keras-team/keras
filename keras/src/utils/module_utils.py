import importlib
import sys


def _is_namespace_package(module):
    spec = getattr(module, "__spec__", None)
    return getattr(spec, "origin", False) is None and not any(
        not attr.startswith("_") for attr in dir(module)
    )


def _has_live_descendant(name):
    prefix = f"{name}."
    locks = getattr(importlib._bootstrap, "_module_locks", ())
    return any(
        key.startswith(prefix) for key in (*list(sys.modules), *list(locks))
    )


def _cleanup_namespace_modules(names):
    for name in reversed(names):
        mod = sys.modules.get(name)
        if _is_namespace_package(mod) and not _has_live_descendant(name):
            sys.modules.pop(name, None)
            parent_name, _, attr = name.rpartition(".")
            if parent_name:
                parent_mod = sys.modules.get(parent_name)
                if getattr(parent_mod, attr, None) is mod:
                    delattr(parent_mod, attr)


class LazyModule:
    """Lazily imports a module on first check or attribute access.

    `available` is `True` only when the import succeeds and the module is not
    an attribute-less namespace package. Direct attribute access on an
    unavailable or leftover namespace install raises `ImportError`.
    """

    def __init__(self, name, pip_name=None, import_error_msg=None):
        self.name = name
        self.pip_name = pip_name or name
        self.import_error_msg = import_error_msg or (
            f"This requires the {self.name} module. "
            f"You can install it via `pip install {self.pip_name}`"
        )
        self.module = None
        self._available = None

    @property
    def available(self):
        if self._available is None:
            try:
                self.initialize()
            except ImportError:
                self._available = False
        return self._available

    def _reject_if_namespace(self, module, newly_loaded=()):
        if _is_namespace_package(module):
            _cleanup_namespace_modules(newly_loaded)
            self._available = False
            raise ImportError(self.import_error_msg)
        return module

    def _import_module(self, name):
        parts = name.split(".")
        prefixes = [".".join(parts[: i + 1]) for i in range(len(parts))]
        newly_loaded = [p for p in prefixes if p not in sys.modules]
        try:
            module = importlib.import_module(name)
        except ImportError:
            _cleanup_namespace_modules(newly_loaded)
            self._available = False
            raise ImportError(self.import_error_msg)
        return self._reject_if_namespace(module, newly_loaded)

    def initialize(self):
        if self._available is False:
            raise ImportError(self.import_error_msg)
        self.module = self._import_module(self.name)
        self._available = True

    def __getattr__(self, name):
        if name == "_api_export_path":
            raise AttributeError
        if self.module is None:
            if not self.available:
                raise ImportError(self.import_error_msg)
        return getattr(self.module, name)

    def __repr__(self):
        return f"LazyModule({self.name})"


class OrbaxLazyModule(LazyModule):
    def _newly_loaded_orbax_modules(self, pre_existing):
        return sorted(
            (
                k
                for k in list(sys.modules)
                if (k == "orbax" or k.startswith("orbax."))
                and k not in pre_existing
            ),
            key=lambda s: s.count("."),
        )

    def initialize(self):
        if self._available is False:
            raise ImportError(self.import_error_msg)
        pre_existing = {
            k
            for k in list(sys.modules)
            if k == "orbax" or k.startswith("orbax.")
        }
        parent_module = self._import_module("orbax.checkpoint")
        try:
            v1_module = parent_module.v1
        except (ImportError, AttributeError):
            _cleanup_namespace_modules(
                self._newly_loaded_orbax_modules(pre_existing)
            )
            self._available = False
            raise ImportError(self.import_error_msg)
        cleanup_targets = [
            *self._newly_loaded_orbax_modules(pre_existing),
            "orbax.checkpoint.v1",
        ]
        self.module = self._reject_if_namespace(v1_module, cleanup_targets)
        self.parent_module = parent_module
        self._available = True

    def __getattr__(self, name):
        if name == "_api_export_path":
            raise AttributeError
        if self.module is None:
            if not self.available:
                raise ImportError(self.import_error_msg)
        if name == "multihost":
            return self.parent_module.multihost
        return getattr(self.module, name)


tensorflow = LazyModule("tensorflow")
gfile = LazyModule("tensorflow.io.gfile", pip_name="tensorflow")
tensorflow_io = LazyModule("tensorflow_io")
scipy = LazyModule("scipy")
jax = LazyModule("jax")
h5py = LazyModule("h5py")
torch_xla = LazyModule(
    "torch_xla",
    import_error_msg=(
        "This requires the torch_xla module. You can install it via "
        "`pip install torch-xla`. Additionally, you may need to update "
        "LD_LIBRARY_PATH if necessary. Torch XLA builds a shared library, "
        "_XLAC.so, which needs to link to the version of Python it was built "
        "with. Use the following command to update LD_LIBRARY_PATH: "
        "`export LD_LIBRARY_PATH=<path to Python>/lib:$LD_LIBRARY_PATH`"
    ),
)
optree = LazyModule("optree")
dmtree = LazyModule("tree")
tf2onnx = LazyModule("tf2onnx")
jax2onnx = LazyModule("jax2onnx")
grain = LazyModule("grain")
litert = LazyModule("ai_edge_litert")
ocp = OrbaxLazyModule(
    "orbax.checkpoint.v1",
    pip_name="orbax-checkpoint",
    import_error_msg=(
        "OrbaxCheckpoint requires the 'orbax-checkpoint' package. "
        "You can install it via pip install orbax-checkpoint"
    ),
)
