# Originally imported via:
#   stubgen {...} -m mlir._mlir_libs._mlir.passmanager
# Local modifications:
#   * Relative imports for cross-module references.
#   * Add __all__


from . import ir as _ir

__all__ = [
    "PassManager",
]

class PassManager:
    def __init__(self, context: _ir.Context | None = None) -> None: ...
    def _CAPICreate(self) -> object: ...
    def _testing_release(self) -> None: ...
    def enable_ir_printing(
        self,
        print_before_all: bool = False,
        print_after_all: bool = True,
        print_module_scope: bool = False,
        print_after_change: bool = False,
        print_after_failure: bool = False,
        large_elements_limit: int | None = None,
        enable_debug_info: bool = False,
        print_generic_op_form: bool = False,
        tree_printing_dir_path: str | None = None,
    ) -> None: ...
    def enable_verifier(self, enable: bool) -> None: ...
    @staticmethod
    def parse(pipeline: str, context: _ir.Context | None = None) -> PassManager: ...
    def run(self, module: _ir._OperationBase) -> None: ...
    @property
    def _CAPIPtr(self) -> object: ...
