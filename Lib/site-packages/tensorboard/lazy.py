# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorBoard is a webapp for understanding TensorFlow runs and graphs."""


import functools
import threading
import types


def lazy_load(name):
    """Decorator to define a function that lazily loads the module 'name'.

    This can be used to defer importing troublesome dependencies - e.g. ones that
    are large and infrequently used, or that cause a dependency cycle -
    until they are actually used.

    Args:
      name: the fully-qualified name of the module; typically the last segment
        of 'name' matches the name of the decorated function

    Returns:
      Decorator function that produces a lazy-loading module 'name' backed by the
      underlying decorated function.
    """

    def wrapper(load_fn):
        # Wrap load_fn to call it exactly once and update __dict__ afterwards to
        # make future lookups efficient (only failed lookups call __getattr__).
        @_memoize
        def load_once(self):
            if load_once.loading:
                raise ImportError(
                    "Circular import when resolving LazyModule %r" % name
                )
            load_once.loading = True
            try:
                module = load_fn()
            finally:
                load_once.loading = False
            self.__dict__.update(module.__dict__)
            load_once.loaded = True
            return module

        load_once.loading = False
        load_once.loaded = False

        # Define a module that proxies getattr() and dir() to the result of calling
        # load_once() the first time it's needed. The class is nested so we can close
        # over load_once() and avoid polluting the module's attrs with our own state.
        class LazyModule(types.ModuleType):
            def __getattr__(self, attr_name):
                return getattr(load_once(self), attr_name)

            def __dir__(self):
                return dir(load_once(self))

            def __repr__(self):
                if load_once.loaded:
                    return "<%r via LazyModule (loaded)>" % load_once(self)
                return (
                    "<module %r via LazyModule (not yet loaded)>"
                    % self.__name__
                )

        return LazyModule(name)

    return wrapper


def _memoize(f):
    """Memoizing decorator for f, which must have exactly 1 hashable
    argument."""
    nothing = object()  # Unique "no value" sentinel object.
    cache = {}
    # Use a reentrant lock so that if f references the resulting wrapper we die
    # with recursion depth exceeded instead of deadlocking.
    lock = threading.RLock()

    @functools.wraps(f)
    def wrapper(arg):
        if cache.get(arg, nothing) is nothing:
            with lock:
                if cache.get(arg, nothing) is nothing:
                    cache[arg] = f(arg)
        return cache[arg]

    return wrapper
