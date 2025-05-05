from flax import nnx


class JaxLayer(nnx.Module):
    def __new__(cls, *args, **kwargs):
        """Overrides __new__ to save constructor arguments for potential
        serialization/config.
        """
        instance = super(JaxLayer, cls).__new__(cls)
        vars(instance)['_object__state'] = nnx.object.ObjectState()
        instance.__init_args = args
        instance.__init_kwargs = kwargs
        return instance
