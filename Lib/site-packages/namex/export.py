class export:
    """Decorator to export a public API in a given package.

    Example usage:

    ```python
    @export(package="keras_tuner", path="keras_tuner.applications.HyperResNet")
    class HyperResNet:
        ...
    ```

    You can also pass a list of paths as `path`, to make
    the same symbol visible under various aliases:

    ```python
    @export(
        package="keras_tuner",
        path=[
            "keras_tuner.applications.HyperResNet",
            "keras_tuner.applications.resnet.HyperResNet",
        ])
    class HyperResNet:
        ...
    ```

    **Note:** All export packages must start with the package name.
    Yes, that is redundant, but that is a helpful sanity check.
    The expectation is that each package will customize
    `export_api` to provide a default value for `package`,
    which will serve to validate all `path` values
    and avoid users inadvertendly ending up with non-exported
    symbols due to a bad path (e.g. `path="applications.HyperResNet"`
    instead of `path="keras_tuner.applications.HyperResNet"`).
    """

    def __init__(self, package, path):
        if isinstance(path, str):
            export_paths = [path]
        elif isinstance(path, list):
            export_paths = path
        else:
            raise ValueError(
                f"Invalid type for `path` argument: "
                f"Received '{path}' "
                f"of type {type(path)}"
            )

        for p in export_paths:
            if not p.startswith(package + "."):
                raise ValueError(
                    f"All `export_path` values should start with '{package}.'. "
                    f"Received: path={path}"
                )

        self.package = package
        self.path = path

    def __call__(self, symbol):
        if hasattr(symbol, "_api_export_path") and symbol._api_export_symbol_id == id(
            symbol
        ):
            raise ValueError(
                f"Symbol {symbol} is already exported as '{symbol._api_export_path}'. "
                f"Cannot also export it to '{self.path}'."
            )
        symbol._api_export_path = self.path
        symbol._api_export_symbol_id = id(symbol)
        return symbol
