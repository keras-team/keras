class NumpyExportArchive:
    def track(self, resource):
        raise NotImplementedError(
            "`track` is not implemented in the numpy backend."
        )

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        raise NotImplementedError(
            "`add_endpoint` is not implemented in the numpy backend."
        )
