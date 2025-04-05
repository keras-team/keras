
def parse_model(model: str) -> tuple[bool, bytes, bytes]:
    """Returns (success-flag, error-message, serialized-proto).

    If success-flag is true, then serialized-proto contains the parsed ModelProto.
    Otherwise, error-message contains a string describing the parse error.
    """

def parse_graph(graph: str) -> tuple[bool, bytes, bytes]:
    """Returns (success-flag, error-message, serialized-proto).

    If success-flag is true, then serialized-proto contains the parsed GraphProto.
    Otherwise, error-message contains a string describing the parse error.
    """

def parse_function(function: str) -> tuple[bool, bytes, bytes]:
    """Returns (success-flag, error-message, serialized-proto).

    If success-flag is true, then serialized-proto contains the parsed FunctionProto.
    Otherwise, error-message contains a string describing the parse error.
    """

def parse_node(node: str) -> tuple[bool, bytes, bytes]:
    """Returns (success-flag, error-message, serialized-proto).

    If success-flag is true, then serialized-proto contains the parsed NodeProto.
    Otherwise, error-message contains a string describing the parse error.
    """
