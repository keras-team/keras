import importlib
import sys


def _is_namespace_package(module):
    spec = getattr(module, "__spec__", None)
    return (
        spec is not None
        and spec.origin is None
        and spec.submodule_search_locations is not None
    )


def _cleanup_namespace_modules(names):
    for name in names:
        if _is_namespace_package(sys.modules.get(name)):
            sys.modules.pop(name, None)


class LazyModule:
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
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _import_module(self, name):
        parts = name.split(".")
        prefixes = [".".join(parts[: i + 1]) for i in range(len(parts))]
        newly_loaded = [p for p in prefixes if p not in sys.modules]
        try:
            module = importlib.import_module(name)
        except ImportError:
            _cleanup_namespace_modules(newly_loaded)
            raise ImportError(self.import_error_msg)
        if _is_namespace_package(module):
            _cleanup_namespace_modules(newly_loaded)
            raise ImportError(self.import_error_msg)
        return module

    def initialize(self):
        self.module = self._import_module(self.name)

    def __getattr__(self, name):
        if name == "_api_export_path":
            raise AttributeError
        if self.module is None:
            self.initialize()
        return getattr(self.module, name)

    def __repr__(self):
        return f"LazyModule({self.name})"


class OrbaxLazyModule(LazyModule):
    def initialize(self):
        parent_module = self._import_module("orbax.checkpoint")
        try:
            v1_module = parent_module.v1
        except ImportError:
            raise ImportError(self.import_error_msg)
        if _is_namespace_package(v1_module):
            raise ImportError(self.import_error_msg)
        self.module = v1_module
        self.parent_module = parent_module

    def __getattr__(self, name):
        if name == "_api_export_path":
            raise AttributeError
        if self.module is None:
            self.initialize()
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
