from flax import nnx


class JaxLayer:
    pass


class NnxLayer(nnx.Module):
    def __init_subclass__(cls):
        super().__init_subclass__()
