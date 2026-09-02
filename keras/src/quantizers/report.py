"""Observability helpers for `model.quantize()`.

This module defines :class:`QuantizationReport`, a small structured record of
what happened during a `model.quantize()` call (which layers were quantized,
which were skipped and why, and any recorded errors).
"""

from keras.src.api_export import keras_export


@keras_export("keras.quantizers.QuantizationReport")
class QuantizationReport:
    """Structured summary of a single `model.quantize()` call.

    A report is returned by `model.quantize()` and also stored on the model as
    `model._quantization_report`. It records, per leaf layer, whether the layer
    was quantized, skipped (with the reason), or hit a recorded error.

    The possible skip reasons are available as class attributes on
    `QuantizationReport` (`SKIP_NO_SUPPORT`, `SKIP_FILTERED`,
    `SKIP_ALREADY_QUANTIZED`, `SKIP_OUTSIDE_STRUCTURE`).

    Args:
        mode: The resolved quantization mode for the call (e.g. `"int8"`).
            Defaults to `None`.

    Attributes:
        mode: The resolved quantization mode for the call.
        quantized: A list of `(path, mode, scheme)` tuples, one per quantized
            layer, where `scheme` is a short description of the resolved
            quantization scheme (e.g. the layer's dtype-policy name).
        skipped: A list of `(path, reason)` tuples. `reason` is one of
            `SKIP_NO_SUPPORT` (the layer does not implement quantization),
            `SKIP_FILTERED` (the layer was excluded by `filters`),
            `SKIP_ALREADY_QUANTIZED` (the layer was already quantized), or
            `SKIP_OUTSIDE_STRUCTURE` (a GPTQ/AWQ call whose quantization
            layer structure does not cover the layer).
        errors: A list of `(path, message)` tuples for any non-fatal errors
            recorded during the call. In the default flow real errors are
            allowed to propagate, so this list is normally empty; it exists so
            callers can attach recoverable diagnostics.
    """

    # Reasons a leaf layer can be skipped during quantization.
    SKIP_NO_SUPPORT = "no quantize support"
    SKIP_FILTERED = "filtered out"
    SKIP_ALREADY_QUANTIZED = "already quantized"
    SKIP_OUTSIDE_STRUCTURE = "outside quantization structure"

    def __init__(self, mode=None):
        self.mode = mode
        self.quantized = []
        self.skipped = []
        self.errors = []

    def add_quantized(self, path, mode, scheme):
        self.quantized.append((path, mode, scheme))

    def add_skipped(self, path, reason):
        self.skipped.append((path, reason))

    def add_error(self, path, error):
        self.errors.append((path, str(error)))

    @property
    def num_quantized(self):
        return len(self.quantized)

    @property
    def num_skipped(self):
        return len(self.skipped)

    @property
    def num_errors(self):
        return len(self.errors)

    def skipped_by_reason(self, reason):
        """Return the list of layer paths that were skipped for `reason`."""
        return [path for path, r in self.skipped if r == reason]

    def summary_warning(self, max_examples=5):
        """Return a single warning message, or `None` if nothing to warn about.

        This replaces the previous behavior of emitting one `UserWarning` per
        non-quantizable leaf layer with a single message that reports counts
        and up to `max_examples` example layer names.
        """
        unsupported = self.skipped_by_reason(self.SKIP_NO_SUPPORT)
        if not unsupported and not self.errors:
            return None

        parts = []
        if unsupported:
            examples = ", ".join(repr(p) for p in unsupported[:max_examples])
            if len(unsupported) > max_examples:
                examples += ", ..."
            parts.append(
                f"{len(unsupported)} layer(s) were skipped because they do "
                f"not support quantization (e.g. {examples})."
            )
        if self.errors:
            error_examples = ", ".join(
                repr(p) for p, _ in self.errors[:max_examples]
            )
            if len(self.errors) > max_examples:
                error_examples += ", ..."
            parts.append(
                f"{len(self.errors)} layer(s) reported errors "
                f"(e.g. {error_examples})."
            )
        parts.append(
            f"Quantized {self.num_quantized} layer(s) in mode "
            f"'{self.mode}'. Call `model.quantization_summary()` or inspect "
            "the returned `QuantizationReport` for details."
        )
        return "`model.quantize()`: " + " ".join(parts)

    def render(self):
        """Return the full report as a multi-line string."""
        lines = [f"Quantization report (mode='{self.mode}')"]
        lines.append("=" * 65)
        lines.append(f"Quantized {self.num_quantized} layer(s):")
        if self.quantized:
            for path, mode, scheme in self.quantized:
                lines.append(f"  - {path} ({mode}): {scheme}")
        else:
            lines.append("  (none)")

        lines.append(f"Skipped {self.num_skipped} layer(s):")
        if self.skipped:
            for path, reason in self.skipped:
                lines.append(f"  - {path}: {reason}")
        else:
            lines.append("  (none)")

        if self.errors:
            lines.append(f"Errors on {self.num_errors} layer(s):")
            for path, message in self.errors:
                lines.append(f"  - {path}: {message}")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"QuantizationReport(mode={self.mode!r}, "
            f"quantized={self.num_quantized}, skipped={self.num_skipped}, "
            f"errors={self.num_errors})"
        )
