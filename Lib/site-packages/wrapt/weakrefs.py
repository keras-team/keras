import functools
import weakref

from .__wrapt__ import ObjectProxy, _FunctionWrapperBase

# A weak function proxy. This will work on instance methods, class
# methods, static methods and regular functions. Special treatment is
# needed for the method types because the bound method is effectively a
# transient object and applying a weak reference to one will immediately
# result in it being destroyed and the weakref callback called. The weak
# reference is therefore applied to the instance the method is bound to
# and the original function. The function is then rebound at the point
# of a call via the weak function proxy.

def _weak_function_proxy_callback(ref, proxy, callback):
    if proxy._self_expired:
        return

    proxy._self_expired = True

    # This could raise an exception. We let it propagate back and let
    # the weakref.proxy() deal with it, at which point it generally
    # prints out a short error message direct to stderr and keeps going.

    if callback is not None:
        callback(proxy)

class WeakFunctionProxy(ObjectProxy):

    __slots__ = ('_self_expired', '_self_instance')

    def __init__(self, wrapped, callback=None):
        # We need to determine if the wrapped function is actually a
        # bound method. In the case of a bound method, we need to keep a
        # reference to the original unbound function and the instance.
        # This is necessary because if we hold a reference to the bound
        # function, it will be the only reference and given it is a
        # temporary object, it will almost immediately expire and
        # the weakref callback triggered. So what is done is that we
        # hold a reference to the instance and unbound function and
        # when called bind the function to the instance once again and
        # then call it. Note that we avoid using a nested function for
        # the callback here so as not to cause any odd reference cycles.

        _callback = callback and functools.partial(
                _weak_function_proxy_callback, proxy=self,
                callback=callback)

        self._self_expired = False

        if isinstance(wrapped, _FunctionWrapperBase):
            self._self_instance = weakref.ref(wrapped._self_instance,
                    _callback)

            if wrapped._self_parent is not None:
                super(WeakFunctionProxy, self).__init__(
                        weakref.proxy(wrapped._self_parent, _callback))

            else:
                super(WeakFunctionProxy, self).__init__(
                        weakref.proxy(wrapped, _callback))

            return

        try:
            self._self_instance = weakref.ref(wrapped.__self__, _callback)

            super(WeakFunctionProxy, self).__init__(
                    weakref.proxy(wrapped.__func__, _callback))

        except AttributeError:
            self._self_instance = None

            super(WeakFunctionProxy, self).__init__(
                    weakref.proxy(wrapped, _callback))

    def __call__(*args, **kwargs):
        def _unpack_self(self, *args):
            return self, args

        self, args = _unpack_self(*args)

        # We perform a boolean check here on the instance and wrapped
        # function as that will trigger the reference error prior to
        # calling if the reference had expired.

        instance = self._self_instance and self._self_instance()
        function = self.__wrapped__ and self.__wrapped__

        # If the wrapped function was originally a bound function, for
        # which we retained a reference to the instance and the unbound
        # function we need to rebind the function and then call it. If
        # not just called the wrapped function.

        if instance is None:
            return self.__wrapped__(*args, **kwargs)

        return function.__get__(instance, type(instance))(*args, **kwargs)
