import io


class KerasSaveable:
    # Note: renaming this function will cause old pickles to be broken.
    # This is probably not a huge deal, as pickle should not be a recommended
    # saving format -- it should only be supported for use with distributed
    # computing frameworks.

    def _obj_type(self):
        raise NotImplementedError(
            "KerasSaveable subclases must provide an "
            "implementation for `obj_type()`"
        )

    @classmethod
    def _unpickle_model(cls, bytesio):
        import keras.src.saving.saving_lib as saving_lib

        # pickle is not safe regardless of what you do.
        return saving_lib._load_model_from_fileobj(
            bytesio, custom_objects=None, compile=True, safe_mode=False
        )

    def __reduce__(self):
        """__reduce__ is used to customize the behavior of `pickle.pickle()`.

        The method returns a tuple of two elements: a function, and a list of
        arguments to pass to that function.  In this case we just leverage the
        keras saving library."""
        import keras.src.saving.saving_lib as saving_lib

        buf = io.BytesIO()
        saving_lib._save_model_to_fileobj(self, buf, "h5")
        return (
            self._unpickle_model,
            (buf,),
        )
