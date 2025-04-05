import contextlib

from keras.src.backend.common import global_state


@contextlib.contextmanager
def keras_option_scope(use_legacy_config=True):
    use_legacy_config_prev_value = global_state.get_global_attribute(
        "use_legacy_config", None
    )
    global_state.set_global_attribute("use_legacy_config", use_legacy_config)
    try:
        yield
    finally:
        global_state.set_global_attribute(
            "use_legacy_config", use_legacy_config_prev_value
        )
