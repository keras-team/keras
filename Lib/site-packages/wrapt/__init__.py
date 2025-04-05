__version_info__ = ('1', '17', '2')
__version__ = '.'.join(__version_info__)

from .__wrapt__ import (ObjectProxy, CallableObjectProxy, FunctionWrapper,
        BoundFunctionWrapper, PartialCallableObjectProxy)

from .patches import (resolve_path, apply_patch, wrap_object, wrap_object_attribute,
        function_wrapper, wrap_function_wrapper, patch_function_wrapper,
        transient_function_wrapper)

from .weakrefs import WeakFunctionProxy

from .decorators import (adapter_factory, AdapterFactory, decorator,
        synchronized)

from .importer import (register_post_import_hook, when_imported,
        notify_module_loaded, discover_post_import_hooks)

# Import of inspect.getcallargs() included for backward compatibility. An
# implementation of this was previously bundled and made available here for
# Python <2.7. Avoid using this in future.

from inspect import getcallargs

# Variant of inspect.formatargspec() included here for forward compatibility.
# This is being done because Python 3.11 dropped inspect.formatargspec() but
# code for handling signature changing decorators relied on it. Exposing the
# bundled implementation here in case any user of wrapt was also needing it.

from .arguments import formatargspec
