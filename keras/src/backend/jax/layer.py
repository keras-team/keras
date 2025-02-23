from flax import nnx

class JaxLayer(nnx.Object):
    def __init_subclass__(cls):
        super().__init_subclass__()
