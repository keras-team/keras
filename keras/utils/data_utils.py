"""Utilities for file download and caching."""
from __future__ import absolute_import
from __future__ import print_function

import functools
import tarfile
import zipfile
import os
import sys
import shutil
import hashlib
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError
from six.moves.urllib.error import HTTPError

from ..utils.generic_utils import Progbar


if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        """Replacement for `urlretrive` for Python 2.

        Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
        `urllib` module, known to have issues with proxy management.

        # Arguments
            url: url to retrieve.
            filename: where to store the retrieved data locally.
            reporthook: a hook function that will be called once
                on establishment of the network connection and once
                after each block read thereafter.
                The hook will be passed three arguments;
                a count of blocks transferred so far,
                a block size in bytes, and the total size of the file.
            data: `data` argument passed to `urlopen`.
        """
        def chunk_read(response, chunk_size=8192, reporthook=None):
            total_size = response.info().get('Content-Length').strip()
            total_size = int(total_size)
            count = 0
            while 1:
                chunk = response.read(chunk_size)
                count += 1
                if not chunk:
                    reporthook(count, total_size, total_size)
                    break
                if reporthook:
                    reporthook(count, chunk_size, total_size)
                yield chunk

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve


class Archive(object):
    """ Defines an inteface to extract archive files like tar and zip files.
    """
    signature = None
    offset = 0
    file_type = None
    mime_type = None
    file_path = None

    def __init__(self, file_path):
        self.file_path = file_path

    @classmethod
    def is_match(cls, fpath):
        """ Returns True if the file's signature matches the supported archive format.
        """
        match = False
        if cls.signature is None:
            return match
        with open(fpath, 'rb') as f:
            f.seek(cls.offset)
            match = f.read(len(cls.signature)).startswith(cls.signature)
        return match

    def open(self, mode='rb'):
        """ Open the file, or archive object in subclasses.
        """
        return open(self.file_path, mode)

    def extractall(self, archive, path="."):
        """ Extract all archive contents if available.
        """
        return None

    def extractall_if_match(self, path="."):
        if (self.file_type is None) or not self.is_match(self.file_path):
            return False
        with self.open() as archive:
            try:
                self.extractall(archive, path=path)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                raise
        return True


class TarArchive(Archive):
    """ Tar file Archive, wraps python tarfile module.
    """
    signature = '\x75\x73\x74\x61\x72'
    offset = 0x101
    file_type = 'tar'
    mime_type = 'application/x-tar'

    @classmethod
    def is_match(cls, fpath):
        """ Returns True if the file's signature matches supported tar archive format.
        """
        return tarfile.is_tarfile(fpath)

    def open(self, mode='r'):
        """ Opens a tarfile.TarFile object at the file path.
        """
        return tarfile.open(self.file_path, mode)

    def extractall(self, archive, path="."):
        return archive.extractall(path)


class ZipArchive(Archive):
    """ Zip file Archive, wraps python zipfile module.
    """
    signature = '\x50\x4b\x03\x04'
    offset = 0
    file_type = 'zip'
    mime_type = 'compressed/zip'

    @classmethod
    def is_match(cls, fpath):
        """ Returns True if the file's signature matches that of a zip formatted file.
        """
        return zipfile.is_zipfile(fpath)

    def open(self):
        """ Returns True if the file's signature matches the supported archive format.
        """
        return zipfile.ZipFile(self.file_path, 'r')

    def extractall(self, archive, path="."):
        return archive.extractall(path)


def extract_archive(file_path, path='.', archive_formats='auto'):
    """Extracts an archive file if it matches one of the recognized formats

    # Arguments
        file_path: path to the archive file
        path: path to extract the archive file
        archive_formats: List of Archive formats to try extracting the file.
                         The default 'auto' is [TarArchive, ZipArchive].
                         Options are subclasses of the Archive class.
                         None or an empty list will return no matches found.
    """
    match_found = False
    if archive_formats is 'auto':
        archive_formats = [TarArchive, ZipArchive]
    for archive_type in archive_formats:
        archive = archive_type(file_path)
        match_found = archive.extractall_if_match(path)
        if match_found:
            return match_found
    return match_found


def get_file(fname, origin, untar=False,
             md5_hash=None, cache_subdir='datasets',
             file_hash=None,
             hash_algorithm='auto',
             extract=False,
             archive_formats='auto'):
    """Downloads a file from a URL if it not already in the cache.

    Passing the hash will verify the file after download which also
    means it will skip re-downloading if it is already present in the cache.
    Python example of how to get the sha256 hash:

    ```python
       >>> import os, hashlib
       >>> print hashlib.sha256(open('/path/to/file').read()).hexdigest()
       'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```

    # Arguments
        fname: name of the file
        origin: original URL of the file
        untar: Deprecated in favor of 'extract'.
               boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of 'file_hash'.
                  MD5 hash of the file for verification
        file_hash: The expected hash string of the file after download,
                   preferably sha256, but md5 is also supported.
        cache_subdir: Directory where the file is saved,
                      ~/.keras/ is the default and /tmp/.keras
                      is a backup default if the first does not work.
        hash_algorithm: Select the hash algorithm to verify the file.
                        options are 'md5', 'sha256', and 'auto'
                        The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_formats: List of Archive formats to try extracting the file.
                  The default 'auto' tries [TarArchive, ZipArchive].
                  Options are subclasses of the Archive class.
                  None or an empty list will return no matches found.

    # Returns
        Path to the downloaded file
    """
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of ' +
                      file_hash + ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)
        progbar = None

        def dl_progress(count, block_size, total_size, progbar=None):
            if progbar is None:
                progbar = Progbar(total_size)
            else:
                progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath,
                            functools.partial(dl_progress, progbar=progbar))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            extract_archive(fpath, datadir, archive_formats=[TarArchive])
        return untar_fpath

    if extract:
        extract_archive(fpath, datadir)

    return fpath


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """Validates a file against a SHA256 or MD5 hash.

    Python example of how to get the sha256 hash:

    ```python
       >>> import os, hashlib
       >>> print hashlib.sha256(open('/path/to/file').read()).hexdigest()
       'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```

    # Arguments
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file,
                    preferably sha256, but md5 is also supported.
        algorithm: hash algorithm, one of 'auto', 'sha256',
                   or the insecure 'md5'.
                   The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    # Returns
        Whether the file is valid
    """
    if (algorithm is 'sha256') or (algorithm is 'auto' and len(hash) is 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)
    if str(hasher.hexdigest()) == str(file_hash):
        return True
    else:
        return False
