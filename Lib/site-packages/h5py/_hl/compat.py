"""
Compatibility module for high-level h5py
"""
import sys
from os import fspath, fsencode, fsdecode
from ..version import hdf5_built_version_tuple

# HDF5 supported passing paths as UTF-8 for Windows from 1.10.6, but this
# was broken again in 1.14.4 - https://github.com/HDFGroup/hdf5/issues/5037 .
# The change was reverted in 1.14.6.
if (1, 14, 4) <= hdf5_built_version_tuple < (1, 14, 6):
    WINDOWS_ENCODING = "mbcs"
else:
    WINDOWS_ENCODING = "utf-8"


def filename_encode(filename):
    """
    Encode filename for use in the HDF5 library.

    Due to how HDF5 handles filenames on different systems, this should be
    called on any filenames passed to the HDF5 library. See the documentation on
    filenames in h5py for more information.
    """
    filename = fspath(filename)
    if sys.platform == "win32":
        if isinstance(filename, str):
            return filename.encode(WINDOWS_ENCODING, "strict")
        return filename
    return fsencode(filename)


def filename_decode(filename):
    """
    Decode filename used by HDF5 library.

    Due to how HDF5 handles filenames on different systems, this should be
    called on any filenames passed from the HDF5 library. See the documentation
    on filenames in h5py for more information.
    """
    if sys.platform == "win32":
        if isinstance(filename, bytes):
            return filename.decode(WINDOWS_ENCODING, "strict")
        elif isinstance(filename, str):
            return filename
        else:
            raise TypeError("expect bytes or str, not %s" % type(filename).__name__)
    return fsdecode(filename)
