from __future__ import annotations

import sys
import typing


if typing.TYPE_CHECKING:
    import importlib_metadata as metadata
else:
    if sys.version_info >= (3, 10, 2):
        from importlib import metadata
    else:
        try:
            import importlib_metadata as metadata
        except ModuleNotFoundError:
            # helps bootstrapping when dependencies aren't installed
            from importlib import metadata


__all__ = [
    'metadata',
]
