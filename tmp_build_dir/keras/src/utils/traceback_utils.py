import inspect
import os
import traceback
import types
from functools import wraps

from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state

_EXCLUDED_PATHS = (
    os.path.abspath(os.path.join(__file__, "..", "..")),
    os.path.join("tensorflow", "python"),
)


@keras_export("keras.config.enable_traceback_filtering")
def enable_traceback_filtering():
    """Turn on traceback filtering.

    Raw Keras tracebacks (also known as stack traces)
    involve many internal frames, which can be
    challenging to read through, while not being actionable for end users.
    By default, Keras filters internal frames in most exceptions that it
    raises, to keep traceback short, readable, and focused on what's
    actionable for you (your own code).

    See also `keras.config.disable_traceback_filtering()` and
    `keras.config.is_traceback_filtering_enabled()`.

    If you have previously disabled traceback filtering via
    `keras.config.disable_traceback_filtering()`, you can re-enable it via
    `keras.config.enable_traceback_filtering()`.
    """
    global_state.set_global_attribute("traceback_filtering", True)


@keras_export("keras.config.disable_traceback_filtering")
def disable_traceback_filtering():
    """Turn off traceback filtering.

    Raw Keras tracebacks (also known as stack traces)
    involve many internal frames, which can be
    challenging to read through, while not being actionable for end users.
    By default, Keras filters internal frames in most exceptions that it
    raises, to keep traceback short, readable, and focused on what's
    actionable for you (your own code).

    See also `keras.config.enable_traceback_filtering()` and
    `keras.config.is_traceback_filtering_enabled()`.

    If you have previously disabled traceback filtering via
    `keras.config.disable_traceback_filtering()`, you can re-enable it via
    `keras.config.enable_traceback_filtering()`.
    """
    global_state.set_global_attribute("traceback_filtering", False)


@keras_export("keras.config.is_traceback_filtering_enabled")
def is_traceback_filtering_enabled():
    """Check if traceback filtering is enabled.

    Raw Keras tracebacks (also known as stack traces)
    involve many internal frames, which can be
    challenging to read through, while not being actionable for end users.
    By default, Keras filters internal frames in most exceptions that it
    raises, to keep traceback short, readable, and focused on what's
    actionable for you (your own code).

    See also `keras.config.enable_traceback_filtering()` and
    `keras.config.disable_traceback_filtering()`.

    If you have previously disabled traceback filtering via
    `keras.config.disable_traceback_filtering()`, you can re-enable it via
    `keras.config.enable_traceback_filtering()`.

    Returns:
        Boolean, `True` if traceback filtering is enabled,
        and `False` otherwise.
    """
    return global_state.get_global_attribute("traceback_filtering", True)


def include_frame(fname):
    for exclusion in _EXCLUDED_PATHS:
        if exclusion in fname:
            return False
    return True


def _process_traceback_frames(tb):
    """Iterate through traceback frames and return a new, filtered traceback."""
    last_tb = None
    tb_list = list(traceback.walk_tb(tb))
    for f, line_no in reversed(tb_list):
        if include_frame(f.f_code.co_filename):
            last_tb = types.TracebackType(last_tb, f, f.f_lasti, line_no)
    if last_tb is None and tb_list:
        # If no frames were kept during filtering, create a new traceback
        # from the outermost function.
        f, line_no = tb_list[-1]
        last_tb = types.TracebackType(last_tb, f, f.f_lasti, line_no)
    return last_tb


def filter_traceback(fn):
    """Filter out Keras-internal traceback frames in exceptions raised by fn."""

    @wraps(fn)
    def error_handler(*args, **kwargs):
        if not is_traceback_filtering_enabled():
            return fn(*args, **kwargs)

        filtered_tb = None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            filtered_tb = _process_traceback_frames(e.__traceback__)
            # To get the full stack trace, call:
            # `keras.config.disable_traceback_filtering()`
            raise e.with_traceback(filtered_tb) from None
        finally:
            del filtered_tb

    return error_handler


def inject_argument_info_in_traceback(fn, object_name=None):
    """Add information about call argument values to an error message.

    Arguments:
        fn: Function to wrap. Exceptions raised by the this function will be
            re-raised with additional information added to the error message,
            displaying the values of the different arguments that the function
            was called with.
        object_name: String, display name of the class/function being called,
            e.g. `'layer "layer_name" (LayerClass)'`.

    Returns:
        A wrapped version of `fn`.
    """
    if backend.backend() == "tensorflow":
        from tensorflow import errors as tf_errors
    else:
        tf_errors = None

    @wraps(fn)
    def error_handler(*args, **kwargs):
        if not is_traceback_filtering_enabled():
            return fn(*args, **kwargs)

        signature = None
        bound_signature = None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if hasattr(e, "_keras_call_info_injected"):
                # Only inject info for the innermost failing call
                raise e
            signature = inspect.signature(fn)
            try:
                # The first argument is `self`, so filter it out
                bound_signature = signature.bind(*args, **kwargs)
            except TypeError:
                # Likely unbindable arguments
                raise e

            # Add argument context
            arguments_context = []
            for arg in list(signature.parameters.values()):
                if arg.name in bound_signature.arguments:
                    value = tree.map_structure(
                        format_argument_value,
                        bound_signature.arguments[arg.name],
                    )
                else:
                    value = arg.default
                arguments_context.append(f"  • {arg.name}={value}")
            if arguments_context:
                arguments_context = "\n".join(arguments_context)
                # Get original error message and append information to it.
                if tf_errors is not None and isinstance(e, tf_errors.OpError):
                    message = e.message
                elif e.args:
                    # Canonically, the 1st argument in an exception is the error
                    # message. This works for all built-in Python exceptions.
                    message = e.args[0]
                else:
                    message = ""
                display_name = f"{object_name if object_name else fn.__name__}"
                message = (
                    f"Exception encountered when calling {display_name}.\n\n"
                    f"\x1b[1m{message}\x1b[0m\n\n"
                    f"Arguments received by {display_name}:\n"
                    f"{arguments_context}"
                )

                # Reraise exception, with added context
                if tf_errors is not None and isinstance(e, tf_errors.OpError):
                    new_e = e.__class__(e.node_def, e.op, message, e.error_code)
                else:
                    try:
                        # For standard exceptions such as ValueError, TypeError,
                        # etc.
                        new_e = e.__class__(message)
                    except TypeError:
                        # For any custom error that doesn't have a standard
                        # signature.
                        new_e = RuntimeError(message)
                new_e._keras_call_info_injected = True
            else:
                new_e = e
            raise new_e.with_traceback(e.__traceback__) from None
        finally:
            del signature
            del bound_signature

    return error_handler


def format_argument_value(value):
    if backend.is_tensor(value):
        # Simplified representation for eager / graph tensors
        # to keep messages readable
        if backend.backend() == "tensorflow":
            tensor_cls = "tf.Tensor"
        elif backend.backend() == "jax":
            tensor_cls = "jnp.ndarray"
        elif backend.backend() == "torch":
            tensor_cls = "torch.Tensor"
        elif backend.backend() == "numpy":
            tensor_cls = "np.ndarray"
        else:
            tensor_cls = "array"

        return (
            f"{tensor_cls}(shape={value.shape}, "
            f"dtype={backend.standardize_dtype(value.dtype)})"
        )
    return repr(value)
