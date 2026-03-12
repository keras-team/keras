"""Base class for NeptuneModel export archive."""


class NeptuneModelExportArchive:
    def __init__(self):
        raise NotImplementedError(
            "NeptuneExportArchive is an abstract class. "
            "Use a subclass such as OrbaxSavedModelExportArchive."
        )

    def track(self, resource):
        raise NotImplementedError()

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        raise NotImplementedError()

    def track_and_add_endpoint(self, name, resource, input_signature, **kwargs):
        raise NotImplementedError()

    def add_variable_collection(self, name, variables):
        raise NotImplementedError()

    def write_out(self, filepath, options=None, verbose=True):
        raise NotImplementedError()
