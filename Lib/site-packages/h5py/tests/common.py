# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

import sys
import os
import shutil
import inspect
import tempfile
import subprocess
from contextlib import contextmanager
from functools import wraps

import numpy as np
from numpy.lib.recfunctions import repack_fields
import h5py

import unittest as ut


# Check if non-ascii filenames are supported
# Evidently this is the most reliable way to check
# See also h5py issue #263 and ipython #466
# To test for this, run the testsuite with LC_ALL=C
try:
    testfile, fname = tempfile.mkstemp(chr(0x03b7))
except UnicodeError:
    UNICODE_FILENAMES = False
else:
    UNICODE_FILENAMES = True
    os.close(testfile)
    os.unlink(fname)
    del fname
    del testfile


class TestCase(ut.TestCase):

    """
        Base class for unit tests.
    """

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp(prefix='h5py-test_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir)

    def mktemp(self, suffix='.hdf5', prefix='', dir=None):
        if dir is None:
            dir = self.tempdir
        return tempfile.mktemp(suffix, prefix, dir=dir)

    def mktemp_mpi(self, comm=None, suffix='.hdf5', prefix='', dir=None):
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        fname = None
        if comm.Get_rank() == 0:
            fname = self.mktemp(suffix, prefix, dir)
        fname = comm.bcast(fname, 0)
        return fname

    def setUp(self):
        self.f = h5py.File(self.mktemp(), 'w')

    def tearDown(self):
        try:
            if self.f:
                self.f.close()
        except:
            pass

    def assertSameElements(self, a, b):
        for x in a:
            match = False
            for y in b:
                if x == y:
                    match = True
            if not match:
                raise AssertionError("Item '%s' appears in a but not b" % x)

        for x in b:
            match = False
            for y in a:
                if x == y:
                    match = True
            if not match:
                raise AssertionError("Item '%s' appears in b but not a" % x)

    def assertArrayEqual(self, dset, arr, message=None, precision=None, check_alignment=True):
        """ Make sure dset and arr have the same shape, dtype and contents, to
            within the given precision, optionally ignoring differences in dtype alignment.

            Note that dset may be a NumPy array or an HDF5 dataset.
        """
        if precision is None:
            precision = 1e-5
        if message is None:
            message = ''
        else:
            message = ' (%s)' % message

        if np.isscalar(dset) or np.isscalar(arr):
            assert np.isscalar(dset) and np.isscalar(arr), \
                'Scalar/array mismatch ("%r" vs "%r")%s' % (dset, arr, message)
            dset = np.asarray(dset)
            arr = np.asarray(arr)

        assert dset.shape == arr.shape, \
            "Shape mismatch (%s vs %s)%s" % (dset.shape, arr.shape, message)
        if dset.dtype != arr.dtype:
            if check_alignment:
                normalized_dset_dtype = dset.dtype
                normalized_arr_dtype = arr.dtype
            else:
                normalized_dset_dtype = repack_fields(dset.dtype)
                normalized_arr_dtype = repack_fields(arr.dtype)

            assert normalized_dset_dtype == normalized_arr_dtype, \
                "Dtype mismatch (%s vs %s)%s" % (normalized_dset_dtype, normalized_arr_dtype, message)

            if not check_alignment:
                if normalized_dset_dtype != dset.dtype:
                    dset = repack_fields(np.asarray(dset))
                if normalized_arr_dtype != arr.dtype:
                    arr = repack_fields(np.asarray(arr))

        if arr.dtype.names is not None:
            for n in arr.dtype.names:
                message = '[FIELD %s] %s' % (n, message)
                self.assertArrayEqual(dset[n], arr[n], message=message, precision=precision, check_alignment=check_alignment)
        elif arr.dtype.kind in ('i', 'f'):
            assert np.all(np.abs(dset[...] - arr[...]) < precision), \
                "Arrays differ by more than %.3f%s" % (precision, message)
        elif arr.dtype.kind == 'O':
            for v1, v2 in zip(dset.flat, arr.flat):
                self.assertArrayEqual(v1, v2, message=message, precision=precision, check_alignment=check_alignment)
        else:
            assert np.all(dset[...] == arr[...]), \
                "Arrays are not equal (dtype %s) %s" % (arr.dtype.str, message)

    def assertNumpyBehavior(self, dset, arr, s, skip_fast_reader=False):
        """ Apply slicing arguments "s" to both dset and arr.

        Succeeds if the results of the slicing are identical, or the
        exception raised is of the same type for both.

        "arr" must be a Numpy array; "dset" may be a NumPy array or dataset.
        """
        exc = None
        try:
            arr_result = arr[s]
        except Exception as e:
            exc = type(e)

        s_fast = s if isinstance(s, tuple) else (s,)

        if exc is None:
            self.assertArrayEqual(dset[s], arr_result)

            if not skip_fast_reader:
                self.assertArrayEqual(
                    dset._fast_reader.read(s_fast),
                    arr_result,
                )
        else:
            with self.assertRaises(exc):
                dset[s]

            if not skip_fast_reader:
                with self.assertRaises(exc):
                    dset._fast_reader.read(s_fast)

NUMPY_RELEASE_VERSION = tuple([int(i) for i in np.__version__.split(".")[0:2]])

@contextmanager
def closed_tempfile(suffix='', text=None):
    """
    Context manager which yields the path to a closed temporary file with the
    suffix `suffix`. The file will be deleted on exiting the context. An
    additional argument `text` can be provided to have the file contain `text`.
    """
    with tempfile.NamedTemporaryFile(
        'w+t', suffix=suffix, delete=False
    ) as test_file:
        file_name = test_file.name
        if text is not None:
            test_file.write(text)
            test_file.flush()
    yield file_name
    shutil.rmtree(file_name, ignore_errors=True)


def insubprocess(f):
    """Runs a test in its own subprocess"""
    @wraps(f)
    def wrapper(request, *args, **kwargs):
        curr_test = inspect.getsourcefile(f) + "::" + request.node.name
        # get block around test name
        insub = "IN_SUBPROCESS_" + curr_test
        for c in "/\\,:.":
            insub = insub.replace(c, "_")
        defined = os.environ.get(insub, None)
        # fork process
        if defined:
            return f(request, *args, **kwargs)
        else:
            os.environ[insub] = '1'
            env = os.environ.copy()
            env[insub] = '1'
            env.update(getattr(f, 'subproc_env', {}))

            with closed_tempfile() as stdout:
                with open(stdout, 'w+t') as fh:
                    rtn = subprocess.call([sys.executable, '-m', 'pytest', curr_test],
                                          stdout=fh, stderr=fh, env=env)
                with open(stdout, 'rt') as fh:
                    out = fh.read()

            assert rtn == 0, "\n" + out
    return wrapper


def subproc_env(d):
    """Set environment variables for the @insubprocess decorator"""
    def decorator(f):
        f.subproc_env = d
        return f

    return decorator
