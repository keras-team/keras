import importlib


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

    def initialize(self):
        try:
            self.module = importlib.import_module(self.name)
        except ImportError:
            raise ImportError(self.import_error_msg)

    def __getattr__(self, name):
        if name == "_api_export_path":
            raise AttributeError
        if self.module is None:
            self.initialize()
        return getattr(self.module, name)

    def __repr__(self):
        return f"LazyModule({self.name})"


tensorflow = LazyModule("tensorflow")
gfile = LazyModule("tensorflow.io.gfile", pip_name="tensorflow")
tensorflow_io = LazyModule("tensorflow_io")
scipy = LazyModule("scipy")
jax = LazyModule("jax")
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
grain = LazyModule("grain")
