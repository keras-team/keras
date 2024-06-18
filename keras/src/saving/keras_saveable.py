import io

from keras.src.saving.object_registration import get_custom_objects


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
    def _unpickle_model(cls, data):
        import keras.src.saving.saving_lib as saving_lib

        # pickle is not safe regardless of what you do.

        if "custom_objects_buf" in data.keys():
            import pickle

            custom_objects = pickle.load(data["custom_objects_buf"])
        else:
            custom_objects = None

        return saving_lib._load_model_from_fileobj(
            data["model_buf"],
            custom_objects=custom_objects,
            compile=True,
            safe_mode=False,
        )

    def __reduce__(self):
        """__reduce__ is used to customize the behavior of `pickle.pickle()`.

        The method returns a tuple of two elements: a function, and a list of
        arguments to pass to that function.  In this case we just leverage the
        keras saving library."""
        import keras.src.saving.saving_lib as saving_lib

        data = {}

        model_buf = io.BytesIO()
        saving_lib._save_model_to_fileobj(self, model_buf, "h5")
        data["model_buf"] = model_buf

        try:
            import cloudpickle

            custom_objects_buf = io.BytesIO()
            cloudpickle.dump(get_custom_objects(), custom_objects_buf)
            custom_objects_buf.seek(0)
            data["custom_objects_buf"] = custom_objects_buf
        except ImportError:
            pass

        return (
            self._unpickle_model,
            (data,),
        )
