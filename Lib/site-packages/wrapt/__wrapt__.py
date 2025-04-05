import os

from .wrappers import (ObjectProxy, CallableObjectProxy,
        PartialCallableObjectProxy, FunctionWrapper,
        BoundFunctionWrapper, _FunctionWrapperBase)

try:
    if not os.environ.get('WRAPT_DISABLE_EXTENSIONS'):
        from ._wrappers import (ObjectProxy, CallableObjectProxy,
            PartialCallableObjectProxy, FunctionWrapper,
            BoundFunctionWrapper, _FunctionWrapperBase)

except ImportError:
    pass
