"""Registry of quantization modes.

This module is the single dispatch point for quantization behavior. Each
quantization mode (`"int8"`, `"int4"`, `"float8"`, `"ternary"`,
`"gptq"`, `"awq"`) is described by one `QuantizationMode` descriptor that
owns:

- the mode's config class and default-config resolution,
- the policy-string codec: routing a `"int4/128"`-style string to its
  dtype-policy class, and naming policies after quantization (each policy
  class parses its own grammar),
- per-layer hyperparameter resolution (block size, weight bits, group size),
- model-level orchestration hooks (calibration for structure-aware modes).

The registry is internal API (`keras.src.quantizers`). Adding a mode is a
registration rather than an edit of every resolution helper:

```python
from keras.src.quantizers.mode_registry import QuantizationMode
from keras.src.quantizers.mode_registry import register_quantization_mode

class MyMode(QuantizationMode):
    name = "my_mode"
    ...

register_quantization_mode(MyMode())
```

This module must stay import-light: it is consulted lazily from
`keras.src.dtype_policies` and `keras.src.layers.layer`, so importing it must
not pull in layers or policies at module level.
"""

_MODES = {}  # name -> QuantizationMode, in registration order.


class QuantizationMode:
    """Descriptor for one quantization mode.

    Subclasses set `name` and `config_cls` and override the hooks whose
    behavior differs from the defaults derived from those attributes.
    """

    # The mode identifier, e.g. `"int8"`. Also the root of the policy-string
    # grammar (`"int8_from_float32"`, `"int4/128_from_float32"`).
    name = None

    # The `QuantizationConfig` subclass for this mode, or `None` if the mode
    # constructs its default config another way.
    config_cls = None

    # Whether `quantize(mode)` without a config is an error (calibration
    # modes need datasets that only an explicit config can carry).
    requires_config = False

    # Whether `Model.quantize` must resolve a quantization layer structure
    # (pre-block layers + sequential blocks) before mutating any layer.
    requires_layer_structure = False

    # Storage byte multiplier used by `Model.quantization_summary` (packed
    # sub-byte formats store two values per byte).
    summary_byte_multiplier = 1

    # --- Config resolution ------------------------------------------------

    def default_config(self):
        """Returns the config used when `quantize(mode)` is called bare."""
        if self.requires_config:
            raise ValueError(self._missing_config_error())
        return self.config_cls()

    def _missing_config_error(self):
        return (
            f"For {self.name.upper()}, you must pass a config object in the "
            "`config` argument."
        )

    def validate_config(self, config):
        """Validates a user-provided config object for this mode."""
        if (
            self.requires_config
            and self.config_cls is not None
            and not isinstance(config, self.config_cls)
        ):
            raise ValueError(
                f"Mode '{self.name}' requires a valid `config` argument "
                f"of type `{self.config_cls.__name__}`. "
                f"Received: {type(config)}"
            )

    def config_from_policy(self, policy):
        """Builds the config equivalent to a quantized dtype policy.

        Used by the `Layer.dtype_policy` setter to forward the policy's full
        parameters into `quantize()`. Returns `None` when the mode's config
        cannot be derived from a bare policy (the subsequent `quantize` call
        then raises the mode's missing-config error). A mode may instead
        raise to refuse policy-triggered quantization outright.
        """
        del policy
        if self.requires_config:
            return None
        return self.default_config()

    # --- Per-layer hyperparameter resolution ------------------------------

    # Mode-specific `resolve_*` helpers live on the concrete descriptors
    # (e.g. `Int4Mode.resolve_block_size`). They all share the precedence:
    # explicit config > layer's quantized dtype policy > DTypePolicyMap
    # entry > mode-specific fallback.

    # --- Policy-string codec ----------------------------------------------

    def policy_from_string(self, mode_str, source_name):
        """Builds the dtype policy for a `<mode>_from_<source>` string.

        The default is the generic `QuantizedDTypePolicy`; a mode with a
        dedicated policy class (`Int4DTypePolicy`, `GPTQDTypePolicy`, ...)
        overrides this to build it.
        """
        from keras.src.dtype_policies.dtype_policy import QuantizedDTypePolicy

        return QuantizedDTypePolicy(mode_str, source_name)

    def policy_suffix(self, layer, config):
        """The mode fragment used to name the policy after quantization.

        E.g. `"int8"`, `"int4/128"`, `"gptq/4/128"`; `quantize()` appends
        `_from_<source>` to it.
        """
        del layer, config
        return self.name

    # --- Layer capability -------------------------------------------------

    def supports_layer(self, layer):
        """Whether this mode claims support for `layer`.

        Layers primarily declare support by listing the mode in their
        `variable_serialization_spec`; this hook lets an externally
        registered mode claim layers it can quantize generically without
        the layer having to know about it.
        """
        del layer
        return False

    # --- Model-level orchestration ----------------------------------------

    def finalize_model_quantization(self, model, config, structure, filters):
        """Hook run by `Model.quantize` after the per-layer walk.

        Structure-aware modes run their calibration pass here.
        """
        del model, config, structure, filters


def register_quantization_mode(mode):
    """Registers a `QuantizationMode` descriptor.

    Accepts an instance or a class (instantiated with no arguments), and
    returns its argument unchanged so it can also be used as a class
    decorator. Registration order is observable (validation errors render
    the registered names), so the built-in modes register explicitly in
    `keras.src.quantizers.modes` instead, where the order is written down
    rather than left to the import order.

    The descriptor is validated at registration time, not at first use:
    the name must be a non-empty string, must not be registered already,
    must not contain the policy-grammar separators ("/" and "_from_"),
    must not shadow a standard dtype or mixed-precision policy name, and
    must not share a prefix with a built-in mode (built-in names are
    routed by `str.startswith` over policy strings; externally registered
    modes match only their exact grammar, so collisions between them are
    unambiguous).
    """
    from keras.src import backend
    from keras.src.dtype_policies.dtype_policy import QUANTIZATION_MODES

    descriptor = mode() if isinstance(mode, type) else mode
    name = descriptor.name
    if not isinstance(name, str) or not name:
        raise ValueError(
            "A quantization mode must define a non-empty string `name`. "
            f"Received: name={name!r}"
        )
    if name in _MODES:
        raise ValueError(
            f"A quantization mode named '{name}' is already registered."
        )
    if "/" in name or "_from_" in name:
        raise ValueError(
            f"Cannot register quantization mode '{name}': its name must "
            "not contain '/' or '_from_', which are the policy-string "
            "grammar separators."
        )
    if name not in QUANTIZATION_MODES:
        try:
            backend.standardize_dtype(name)
            is_standard_dtype = True
        except ValueError:
            is_standard_dtype = False
        if is_standard_dtype or name.startswith("mixed_"):
            raise ValueError(
                f"Cannot register quantization mode '{name}': its name "
                "conflicts with a standard dtype or mixed-precision "
                "policy name."
            )
    for existing in _MODES:
        builtin_involved = (
            existing in QUANTIZATION_MODES or name in QUANTIZATION_MODES
        )
        if builtin_involved and (
            existing.startswith(name) or name.startswith(existing)
        ):
            raise ValueError(
                f"Cannot register quantization mode '{name}': its name "
                f"collides with registered mode '{existing}'. Built-in "
                "mode names are routed by prefix over policy strings, so "
                "no mode name may share a prefix with a built-in mode."
            )
    has_config_source = (
        descriptor.config_cls is not None
        or descriptor.requires_config
        or type(descriptor).default_config
        is not QuantizationMode.default_config
    )
    if not has_config_source:
        raise ValueError(
            f"Quantization mode '{name}' must define `config_cls`, set "
            "`requires_config = True`, or override `default_config()`."
        )
    _MODES[name] = descriptor
    # Return the argument, not the descriptor: as a class decorator this
    # must leave the class bound to its name, still subclassable.
    return mode


def unregister_quantization_mode(name):
    """Removes a registered mode (intended for tests)."""
    _MODES.pop(name, None)


def get_mode(name):
    """Returns the descriptor registered under `name`, or `None`."""
    return _MODES.get(name)


def is_registered(name):
    return name in _MODES


def registered_mode_names():
    """All registered mode names, as a tuple, in registration order."""
    return tuple(_MODES)
