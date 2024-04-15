import importlib


class LazyModule:
    def __init__(self, name, pip_name=None):
        self.name = name
        pip_name = pip_name or name
        self.pip_name = pip_name
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
            raise ImportError(
                f"This requires the {self.name} module. "
                f"You can install it via `pip install {self.pip_name}`"
            )

    def __getattr__(self, name):
        if name == "_api_export_path":
            raise AttributeError
        if self.module is None:
            self.initialize()
        return getattr(self.module, name)


tensorflow = LazyModule("tensorflow")
gfile = LazyModule("tensorflow.io.gfile", pip_name="tensorflow")
tensorflow_io = LazyModule("tensorflow_io")
scipy = LazyModule("scipy")
jax = LazyModule("jax")
optree = LazyModule("optree")
dmtree = LazyModule("tree")
