class CheckerContext:
    ir_version: int = ...
    opset_imports: dict[str, int] = ...


class LexicalScopeContext:
    ir_version: int = ...
    opset_imports: dict[str, int] = ...


class ValidationError(Exception): ...

def check_value_info(bytes: bytes, checker_context: CheckerContext) -> None: ...  # noqa: A002
def check_tensor(bytes: bytes, checker_context: CheckerContext) -> None: ...  # noqa: A002
def check_sparse_tensor(bytes: bytes, checker_context: CheckerContext) -> None: ...  # noqa: A002
def check_attribute(bytes: bytes, checker_context: CheckerContext, lexical_scope_context: LexicalScopeContext) -> None: ...  # noqa: A002
def check_node(bytes: bytes, checker_context: CheckerContext, lexical_scope_context: LexicalScopeContext) -> None: ...  # noqa: A002
def check_function(bytes: bytes, checker_context: CheckerContext, lexical_scope_context: LexicalScopeContext) -> None: ...  # noqa: A002
def check_graph(bytes: bytes, checker_context: CheckerContext, lexical_scope_context: LexicalScopeContext) -> None: ...  # noqa: A002
def check_model(bytes: bytes, full_check: bool, skip_opset_compatibility_check: bool, check_custom_domain: bool) -> None: ...  # noqa: A002
def check_model_path(path: str, full_check: bool, skip_opset_compatibility_check: bool, check_custom_domain: bool) -> None: ...
