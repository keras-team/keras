import importlib


class LazyModule:
    def __init__(
        self, name, pip_name=None, import_error_msg=None, required_attr=None
    ):
        self.name = name
        self.pip_name = pip_name or name
        self.import_error_msg = import_error_msg or (
            f"This requires the {self.name} module. "
            f"You can install it via `pip install {self.pip_name}`"
        )
        self.required_attr = required_attr
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

    def _import(self, name):
        try:
            return importlib.import_module(name)
        except ImportError:
            raise ImportError(self.import_error_msg)

    def _require(self, module, attr):
        if not hasattr(module, attr):
            raise ImportError(self.import_error_msg)

    def initialize(self):
        module = self._import(self.name)
        if self.required_attr is not None:
            self._require(module, self.required_attr)
        self.module = module

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
        parent_module = self._import("orbax.checkpoint")
        self._require(parent_module, "v1")
        v1_module = parent_module.v1

        if self.required_attr is not None:
            self._require(v1_module, self.required_attr)

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


tensorflow = LazyModule("tensorflow", required_attr="__version__")
gfile = LazyModule(
    "tensorflow.io.gfile", pip_name="tensorflow", required_attr="GFile"
)
tensorflow_io = LazyModule("tensorflow_io", required_attr="__version__")
scipy = LazyModule("scipy", required_attr="__version__")
jax = LazyModule("jax", required_attr="__version__")
h5py = LazyModule("h5py", required_attr="__version__")
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
    required_attr="__version__",
)
optree = LazyModule("optree", required_attr="__version__")
dmtree = LazyModule("tree", required_attr="__version__")
tf2onnx = LazyModule("tf2onnx", required_attr="__version__")
jax2onnx = LazyModule("jax2onnx", required_attr="to_onnx")
grain = LazyModule("grain", required_attr="__version__")
litert = LazyModule("ai_edge_litert", required_attr="__version__")
ocp = OrbaxLazyModule(
    "orbax.checkpoint.v1",
    pip_name="orbax-checkpoint",
    import_error_msg=(
        "OrbaxCheckpoint requires the 'orbax-checkpoint' package. "
        "You can install it via pip install orbax-checkpoint"
    ),
    required_attr="training",
)
