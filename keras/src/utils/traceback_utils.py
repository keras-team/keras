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


def inject_argument_info_in_error(e, fn, args, kwargs, object_name=None):
    """Add call argument info to an already-caught exception.

    Processes the exception in-place on the error path only, avoiding
    per-call overhead on the happy path.

    Returns the augmented exception, or ``None`` if augmentation was not
    possible (e.g. arguments could not be bound to the signature).
    """
    if backend.backend() == "tensorflow":
        from tensorflow import errors as tf_errors
    else:
        tf_errors = None

    try:
        signature = inspect.signature(fn)
        bound_signature = signature.bind(*args, **kwargs)
    except (ValueError, TypeError):
        return None

    arguments_context = []
    for arg in signature.parameters.values():
        if arg.name in bound_signature.arguments:
            try:
                value = tree.map_structure(
                    format_argument_value,
                    bound_signature.arguments[arg.name],
                )
            except Exception:
                return None
        else:
            value = arg.default
        arguments_context.append(f"  • {arg.name}={value}")

    if not arguments_context:
        return None

    arguments_context = "\n".join(arguments_context)
    if tf_errors is not None and isinstance(e, tf_errors.OpError):
        message = e.message
    elif e.args:
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

    if tf_errors is not None and isinstance(e, tf_errors.OpError):
        new_e = e.__class__(e.node_def, e.op, message, e.error_code)
    else:
        try:
            new_e = e.__class__(message)
        except Exception:
            new_e = RuntimeError(message)
    try:
        new_e._keras_call_info_injected = True
    except Exception:
        return None
    return new_e


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
