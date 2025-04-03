"""This module implements a post import hook mechanism styled after what is
described in PEP-369. Note that it doesn't cope with modules being reloaded.

"""

import sys
import threading

PY2 = sys.version_info[0] == 2

if PY2:
    string_types = basestring,
    find_spec = None
else:
    string_types = str,
    from importlib.util import find_spec

from .__wrapt__ import ObjectProxy

# The dictionary registering any post import hooks to be triggered once
# the target module has been imported. Once a module has been imported
# and the hooks fired, the list of hooks recorded against the target
# module will be truncated but the list left in the dictionary. This
# acts as a flag to indicate that the module had already been imported.

_post_import_hooks = {}
_post_import_hooks_init = False
_post_import_hooks_lock = threading.RLock()

# Register a new post import hook for the target module name. This
# differs from the PEP-369 implementation in that it also allows the
# hook function to be specified as a string consisting of the name of
# the callback in the form 'module:function'. This will result in a
# proxy callback being registered which will defer loading of the
# specified module containing the callback function until required.

def _create_import_hook_from_string(name):
    def import_hook(module):
        module_name, function = name.split(':')
        attrs = function.split('.')
        __import__(module_name)
        callback = sys.modules[module_name]
        for attr in attrs:
            callback = getattr(callback, attr)
        return callback(module)
    return import_hook

def register_post_import_hook(hook, name):
    # Create a deferred import hook if hook is a string name rather than
    # a callable function.

    if isinstance(hook, string_types):
        hook = _create_import_hook_from_string(hook)

    with _post_import_hooks_lock:
        # Automatically install the import hook finder if it has not already
        # been installed.

        global _post_import_hooks_init

        if not _post_import_hooks_init:
            _post_import_hooks_init = True
            sys.meta_path.insert(0, ImportHookFinder())

        # Check if the module is already imported. If not, register the hook
        # to be called after import.

        module = sys.modules.get(name, None)

        if module is None:
            _post_import_hooks.setdefault(name, []).append(hook)

    # If the module is already imported, we fire the hook right away. Note that
    # the hook is called outside of the lock to avoid deadlocks if code run as a
    # consequence of calling the module import hook in turn triggers a separate
    # thread which tries to register an import hook.

    if module is not None:
        hook(module)

# Register post import hooks defined as package entry points.

def _create_import_hook_from_entrypoint(entrypoint):
    def import_hook(module):
        __import__(entrypoint.module_name)
        callback = sys.modules[entrypoint.module_name]
        for attr in entrypoint.attrs:
            callback = getattr(callback, attr)
        return callback(module)
    return import_hook

def discover_post_import_hooks(group):
    try:
        import pkg_resources
    except ImportError:
        return

    for entrypoint in pkg_resources.iter_entry_points(group=group):
        callback = _create_import_hook_from_entrypoint(entrypoint)
        register_post_import_hook(callback, entrypoint.name)

# Indicate that a module has been loaded. Any post import hooks which
# were registered against the target module will be invoked. If an
# exception is raised in any of the post import hooks, that will cause
# the import of the target module to fail.

def notify_module_loaded(module):
    name = getattr(module, '__name__', None)

    with _post_import_hooks_lock:
        hooks = _post_import_hooks.pop(name, ())

    # Note that the hook is called outside of the lock to avoid deadlocks if
    # code run as a consequence of calling the module import hook in turn
    # triggers a separate thread which tries to register an import hook.

    for hook in hooks:
        hook(module)

# A custom module import finder. This intercepts attempts to import
# modules and watches out for attempts to import target modules of
# interest. When a module of interest is imported, then any post import
# hooks which are registered will be invoked.

class _ImportHookLoader:

    def load_module(self, fullname):
        module = sys.modules[fullname]
        notify_module_loaded(module)

        return module

class _ImportHookChainedLoader(ObjectProxy):

    def __init__(self, loader):
        super(_ImportHookChainedLoader, self).__init__(loader)

        if hasattr(loader, "load_module"):
          self.__self_setattr__('load_module', self._self_load_module)
        if hasattr(loader, "create_module"):
          self.__self_setattr__('create_module', self._self_create_module)
        if hasattr(loader, "exec_module"):
          self.__self_setattr__('exec_module', self._self_exec_module)

    def _self_set_loader(self, module):
        # Set module's loader to self.__wrapped__ unless it's already set to
        # something else. Import machinery will set it to spec.loader if it is
        # None, so handle None as well. The module may not support attribute
        # assignment, in which case we simply skip it. Note that we also deal
        # with __loader__ not existing at all. This is to future proof things
        # due to proposal to remove the attribue as described in the GitHub
        # issue at https://github.com/python/cpython/issues/77458. Also prior
        # to Python 3.3, the __loader__ attribute was only set if a custom
        # module loader was used. It isn't clear whether the attribute still
        # existed in that case or was set to None.

        class UNDEFINED: pass

        if getattr(module, "__loader__", UNDEFINED) in (None, self):
            try:
                module.__loader__ = self.__wrapped__
            except AttributeError:
                pass

        if (getattr(module, "__spec__", None) is not None
                and getattr(module.__spec__, "loader", None) is self):
            module.__spec__.loader = self.__wrapped__

    def _self_load_module(self, fullname):
        module = self.__wrapped__.load_module(fullname)
        self._self_set_loader(module)
        notify_module_loaded(module)

        return module

    # Python 3.4 introduced create_module() and exec_module() instead of
    # load_module() alone. Splitting the two steps.

    def _self_create_module(self, spec):
        return self.__wrapped__.create_module(spec)

    def _self_exec_module(self, module):
        self._self_set_loader(module)
        self.__wrapped__.exec_module(module)
        notify_module_loaded(module)

class ImportHookFinder:

    def __init__(self):
        self.in_progress = {}

    def find_module(self, fullname, path=None):
        # If the module being imported is not one we have registered
        # post import hooks for, we can return immediately. We will
        # take no further part in the importing of this module.

        with _post_import_hooks_lock:
            if fullname not in _post_import_hooks:
                return None

        # When we are interested in a specific module, we will call back
        # into the import system a second time to defer to the import
        # finder that is supposed to handle the importing of the module.
        # We set an in progress flag for the target module so that on
        # the second time through we don't trigger another call back
        # into the import system and cause a infinite loop.

        if fullname in self.in_progress:
            return None

        self.in_progress[fullname] = True

        # Now call back into the import system again.

        try:
            if not find_spec:
                # For Python 2 we don't have much choice but to
                # call back in to __import__(). This will
                # actually cause the module to be imported. If no
                # module could be found then ImportError will be
                # raised. Otherwise we return a loader which
                # returns the already loaded module and invokes
                # the post import hooks.

                __import__(fullname)

                return _ImportHookLoader()

            else:
                # For Python 3 we need to use find_spec().loader
                # from the importlib.util module. It doesn't actually
                # import the target module and only finds the
                # loader. If a loader is found, we need to return
                # our own loader which will then in turn call the
                # real loader to import the module and invoke the
                # post import hooks.

                loader = getattr(find_spec(fullname), "loader", None)

                if loader and not isinstance(loader, _ImportHookChainedLoader):
                    return _ImportHookChainedLoader(loader)

        finally:
            del self.in_progress[fullname]

    def find_spec(self, fullname, path=None, target=None):
        # Since Python 3.4, you are meant to implement find_spec() method
        # instead of find_module() and since Python 3.10 you get deprecation
        # warnings if you don't define find_spec().

        # If the module being imported is not one we have registered
        # post import hooks for, we can return immediately. We will
        # take no further part in the importing of this module.

        with _post_import_hooks_lock:
            if fullname not in _post_import_hooks:
                return None

        # When we are interested in a specific module, we will call back
        # into the import system a second time to defer to the import
        # finder that is supposed to handle the importing of the module.
        # We set an in progress flag for the target module so that on
        # the second time through we don't trigger another call back
        # into the import system and cause a infinite loop.

        if fullname in self.in_progress:
            return None

        self.in_progress[fullname] = True

        # Now call back into the import system again.

        try:
            # This should only be Python 3 so find_spec() should always
            # exist so don't need to check.

            spec = find_spec(fullname)
            loader = getattr(spec, "loader", None)

            if loader and not isinstance(loader, _ImportHookChainedLoader):
                spec.loader = _ImportHookChainedLoader(loader)

            return spec

        finally:
            del self.in_progress[fullname]

# Decorator for marking that a function should be called as a post
# import hook when the target module is imported.

def when_imported(name):
    def register(hook):
        register_post_import_hook(hook, name)
        return hook
    return register
