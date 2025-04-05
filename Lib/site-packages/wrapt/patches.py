import inspect
import sys

PY2 = sys.version_info[0] == 2

if PY2:
    string_types = basestring,
else:
    string_types = str,

from .__wrapt__ import FunctionWrapper

# Helper functions for applying wrappers to existing functions.

def resolve_path(module, name):
    if isinstance(module, string_types):
        __import__(module)
        module = sys.modules[module]

    parent = module

    path = name.split('.')
    attribute = path[0]

    # We can't just always use getattr() because in doing
    # that on a class it will cause binding to occur which
    # will complicate things later and cause some things not
    # to work. For the case of a class we therefore access
    # the __dict__ directly. To cope though with the wrong
    # class being given to us, or a method being moved into
    # a base class, we need to walk the class hierarchy to
    # work out exactly which __dict__ the method was defined
    # in, as accessing it from __dict__ will fail if it was
    # not actually on the class given. Fallback to using
    # getattr() if we can't find it. If it truly doesn't
    # exist, then that will fail.

    def lookup_attribute(parent, attribute):
        if inspect.isclass(parent):
            for cls in inspect.getmro(parent):
                if attribute in vars(cls):
                    return vars(cls)[attribute]
            else:
                return getattr(parent, attribute)
        else:
            return getattr(parent, attribute)

    original = lookup_attribute(parent, attribute)

    for attribute in path[1:]:
        parent = original
        original = lookup_attribute(parent, attribute)

    return (parent, attribute, original)

def apply_patch(parent, attribute, replacement):
    setattr(parent, attribute, replacement)

def wrap_object(module, name, factory, args=(), kwargs={}):
    (parent, attribute, original) = resolve_path(module, name)
    wrapper = factory(original, *args, **kwargs)
    apply_patch(parent, attribute, wrapper)
    return wrapper

# Function for applying a proxy object to an attribute of a class
# instance. The wrapper works by defining an attribute of the same name
# on the class which is a descriptor and which intercepts access to the
# instance attribute. Note that this cannot be used on attributes which
# are themselves defined by a property object.

class AttributeWrapper(object):

    def __init__(self, attribute, factory, args, kwargs):
        self.attribute = attribute
        self.factory = factory
        self.args = args
        self.kwargs = kwargs

    def __get__(self, instance, owner):
        value = instance.__dict__[self.attribute]
        return self.factory(value, *self.args, **self.kwargs)

    def __set__(self, instance, value):
        instance.__dict__[self.attribute] = value

    def __delete__(self, instance):
        del instance.__dict__[self.attribute]

def wrap_object_attribute(module, name, factory, args=(), kwargs={}):
    path, attribute = name.rsplit('.', 1)
    parent = resolve_path(module, path)[2]
    wrapper = AttributeWrapper(attribute, factory, args, kwargs)
    apply_patch(parent, attribute, wrapper)
    return wrapper

# Functions for creating a simple decorator using a FunctionWrapper,
# plus short cut functions for applying wrappers to functions. These are
# for use when doing monkey patching. For a more featured way of
# creating decorators see the decorator decorator instead.

def function_wrapper(wrapper):
    def _wrapper(wrapped, instance, args, kwargs):
        target_wrapped = args[0]
        if instance is None:
            target_wrapper = wrapper
        elif inspect.isclass(instance):
            target_wrapper = wrapper.__get__(None, instance)
        else:
            target_wrapper = wrapper.__get__(instance, type(instance))
        return FunctionWrapper(target_wrapped, target_wrapper)
    return FunctionWrapper(wrapper, _wrapper)

def wrap_function_wrapper(module, name, wrapper):
    return wrap_object(module, name, FunctionWrapper, (wrapper,))

def patch_function_wrapper(module, name, enabled=None):
    def _wrapper(wrapper):
        return wrap_object(module, name, FunctionWrapper, (wrapper, enabled))
    return _wrapper

def transient_function_wrapper(module, name):
    def _decorator(wrapper):
        def _wrapper(wrapped, instance, args, kwargs):
            target_wrapped = args[0]
            if instance is None:
                target_wrapper = wrapper
            elif inspect.isclass(instance):
                target_wrapper = wrapper.__get__(None, instance)
            else:
                target_wrapper = wrapper.__get__(instance, type(instance))
            def _execute(wrapped, instance, args, kwargs):
                (parent, attribute, original) = resolve_path(module, name)
                replacement = FunctionWrapper(original, target_wrapper)
                setattr(parent, attribute, replacement)
                try:
                    return wrapped(*args, **kwargs)
                finally:
                    setattr(parent, attribute, original)
            return FunctionWrapper(target_wrapped, _execute)
        return FunctionWrapper(wrapper, _wrapper)
    return _decorator
