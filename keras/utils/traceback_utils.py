# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities related to Keras exception stack trace prettifying."""

import inspect
import os
import sys
import traceback
import types

import tensorflow.compat.v2 as tf

_EXCLUDED_PATHS = (
    os.path.abspath(os.path.join(__file__, "..", "..")),
    os.path.join("tensorflow", "python"),
)


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
    """Filter out Keras-internal stack trace frames in exceptions raised by
    fn."""
    if sys.version_info.major != 3 or sys.version_info.minor < 7:
        return fn

    def error_handler(*args, **kwargs):
        if not tf.debugging.is_traceback_filtering_enabled():
            return fn(*args, **kwargs)

        filtered_tb = None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            filtered_tb = _process_traceback_frames(e.__traceback__)
            # To get the full stack trace, call:
            # `tf.debugging.disable_traceback_filtering()`
            raise e.with_traceback(filtered_tb) from None
        finally:
            del filtered_tb

    return tf.__internal__.decorator.make_decorator(fn, error_handler)


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

    def error_handler(*args, **kwargs):
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
                    value = tf.nest.map_structure(
                        format_argument_value,
                        bound_signature.arguments[arg.name],
                    )
                else:
                    value = arg.default
                arguments_context.append(f"  • {arg.name}={value}")

            if arguments_context:
                arguments_context = "\n".join(arguments_context)
                # Get original error message and append information to it.
                if isinstance(e, tf.errors.OpError):
                    message = e.message
                elif e.args:
                    # Canonically, the 1st argument in an exception is the error
                    # message.  This works for all built-in Python exceptions.
                    message = e.args[0]
                else:
                    message = ""
                display_name = f"{object_name if object_name else fn.__name__}"
                message = (
                    f"Exception encountered when calling {display_name}.\n\n"
                    f"{message}\n\n"
                    f"Call arguments received by {display_name}:\n"
                    f"{arguments_context}"
                )

                # Reraise exception, with added context
                if isinstance(e, tf.errors.OpError):
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

    return tf.__internal__.decorator.make_decorator(fn, error_handler)


def format_argument_value(value):
    if isinstance(value, tf.Tensor):
        # Simplified representation for eager / graph tensors
        # to keep messages readable
        return f"tf.Tensor(shape={value.shape}, dtype={value.dtype.name})"
    return repr(value)
