class MlxExportArchive:
    def track(self, resource):
        raise NotImplementedError(
            "`track` is not implemented in the mlx backend."
        )

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        raise NotImplementedError(
            "`add_endpoint` is not implemented in the mlx backend."
        )

    def track_and_add_endpoint(
        self, name, resource, input_signature, **kwargs
    ):
        raise NotImplementedError(
            "`export_saved_model` only currently supports the "
            "tensorflow, jax and torch backends."
        )
