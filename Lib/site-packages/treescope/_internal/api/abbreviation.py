"""Configurable settings for abbreviation."""

from treescope import context


abbreviation_threshold: context.ContextualValue[int | None] = (
    context.ContextualValue(
        module=__name__,
        qualname="abbreviation_threshold",
        initial_value=None,
    )
)
roundtrip_abbreviation_threshold: context.ContextualValue[int | None] = (
    context.ContextualValue(
        module=__name__,
        qualname="roundtrip_abbreviation_threshold",
        initial_value=None,
    )
)
