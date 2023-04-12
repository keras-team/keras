import sys
import threading

from absl import logging

from keras_core.api_export import keras_core_export

INTERACTIVE_LOGGING = threading.local()
INTERACTIVE_LOGGING.enable = True


@keras_core_export("keras_core.utils.enable_interactive_logging")
def enable_interactive_logging():
    """Turn on interactive logging.

    When interactive logging is enabled, Keras displays logs via stdout.
    This provides the best experience when using Keras in an interactive
    environment such as a shell or a notebook.
    """
    INTERACTIVE_LOGGING.enable = True


@keras_core_export("keras_core.utils.disable_interactive_logging")
def disable_interactive_logging():
    """Turn off interactive logging.

    When interactive logging is disabled, Keras sends logs to `absl.logging`.
    This is the best option when using Keras in a non-interactive
    way, such as running a training or inference job on a server.
    """
    INTERACTIVE_LOGGING.enable = False


@keras_core_export("keras_core.utils.is_interactive_logging_enabled")
def is_interactive_logging_enabled():
    """Check if interactive logging is enabled.

    To switch between writing logs to stdout and `absl.logging`, you may use
    `keras.utils.enable_interactive_logging()` and
    `keras.utils.disable_interactie_logging()`.

    Returns:
      Boolean (True if interactive logging is enabled and False otherwise).
    """
    # Use `getattr` in case `INTERACTIVE_LOGGING`
    # does not have the `enable` attribute.
    return getattr(INTERACTIVE_LOGGING, "enable", True)


def print_msg(message, line_break=True):
    """Print the message to absl logging or stdout."""
    if is_interactive_logging_enabled():
        if line_break:
            sys.stdout.write(message + "\n")
        else:
            sys.stdout.write(message)
        sys.stdout.flush()
    else:
        logging.info(message)
