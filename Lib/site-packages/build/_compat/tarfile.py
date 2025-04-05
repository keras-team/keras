from __future__ import annotations

import sys
import tarfile
import typing


if typing.TYPE_CHECKING:
    TarFile = tarfile.TarFile

else:
    # Per https://peps.python.org/pep-0706/, the "data" filter will become
    # the default in Python 3.14. The first series of releases with the filter
    # had a broken filter that could not process symlinks correctly.
    if (
        (3, 8, 18) <= sys.version_info < (3, 9)
        or (3, 9, 18) <= sys.version_info < (3, 10)
        or (3, 10, 13) <= sys.version_info < (3, 11)
        or (3, 11, 5) <= sys.version_info < (3, 12)
        or (3, 12) <= sys.version_info < (3, 14)
    ):

        class TarFile(tarfile.TarFile):
            extraction_filter = staticmethod(tarfile.data_filter)

    else:
        TarFile = tarfile.TarFile


__all__ = [
    'TarFile',
]
