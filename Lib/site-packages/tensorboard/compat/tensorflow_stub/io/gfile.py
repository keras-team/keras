# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A limited reimplementation of the TensorFlow FileIO API.

The TensorFlow version wraps the C++ FileSystem API.  Here we provide a
pure Python implementation, limited to the features required for
TensorBoard.  This allows running TensorBoard without depending on
TensorFlow for file operations.
"""

import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile

try:
    import botocore.exceptions
    import boto3

    S3_ENABLED = True
except ImportError:
    S3_ENABLED = False

try:
    import fsspec

    FSSPEC_ENABLED = True
except ImportError:
    FSSPEC_ENABLED = False

if sys.version_info < (3, 0):
    # In Python 2 FileExistsError is not defined and the
    # error manifests it as OSError.
    FileExistsError = OSError

from tensorboard.compat.tensorflow_stub import compat, errors


# A good default block size depends on the system in question.
# A somewhat conservative default chosen here.
_DEFAULT_BLOCK_SIZE = 16 * 1024 * 1024


# Registry of filesystems by prefix.
#
# Currently supports "s3://" URLs for S3 based on boto3 and falls
# back to local filesystem.
_REGISTERED_FILESYSTEMS = {}


def register_filesystem(prefix, filesystem):
    if ":" in prefix:
        raise ValueError("Filesystem prefix cannot contain a :")
    _REGISTERED_FILESYSTEMS[prefix] = filesystem


def get_filesystem(filename):
    """Return the registered filesystem for the given file."""
    filename = compat.as_str_any(filename)
    prefix = ""
    index = filename.find("://")
    if index >= 0:
        prefix = filename[:index]
    fs = _REGISTERED_FILESYSTEMS.get(prefix, None)
    if fs is None:
        fs = _get_fsspec_filesystem(filename)
    if fs is None:
        raise ValueError("No recognized filesystem for prefix %s" % prefix)
    return fs


@dataclasses.dataclass(frozen=True)
class StatData:
    """Data returned from the Stat call.

    Attributes:
      length: Length of the data content.
    """

    length: int


class LocalFileSystem:
    """Provides local fileystem access."""

    def exists(self, filename):
        """Determines whether a path exists or not."""
        return os.path.exists(compat.as_bytes(filename))

    def join(self, path, *paths):
        """Join paths with path delimiter."""
        return os.path.join(path, *paths)

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        mode = "rb" if binary_mode else "r"
        encoding = None if binary_mode else "utf8"
        if not exists(filename):
            raise errors.NotFoundError(
                None, None, "Not Found: " + compat.as_text(filename)
            )
        offset = None
        if continue_from is not None:
            offset = continue_from.get("opaque_offset", None)
        with io.open(filename, mode, encoding=encoding) as f:
            if offset is not None:
                f.seek(offset)
            data = f.read(size)
            # The new offset may not be `offset + len(data)`, due to decoding
            # and newline translation.
            # So, just measure it in whatever terms the underlying stream uses.
            continuation_token = {"opaque_offset": f.tell()}
            return (data, continuation_token)

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file, overwriting any existing
        contents.

        Args:
            filename: string, a path
            file_content: string, the contents
            binary_mode: bool, write as binary if True, otherwise text
        """
        self._write(filename, file_content, "wb" if binary_mode else "w")

    def append(self, filename, file_content, binary_mode=False):
        """Append string file contents to a file.

        Args:
            filename: string, a path
            file_content: string, the contents to append
            binary_mode: bool, write as binary if True, otherwise text
        """
        self._write(filename, file_content, "ab" if binary_mode else "a")

    def _write(self, filename, file_content, mode):
        encoding = None if "b" in mode else "utf8"
        with io.open(filename, mode, encoding=encoding) as f:
            compatify = compat.as_bytes if "b" in mode else compat.as_text
            f.write(compatify(file_content))

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        if isinstance(filename, str):
            return [
                # Convert the filenames to string from bytes.
                compat.as_str_any(matching_filename)
                for matching_filename in py_glob.glob(compat.as_bytes(filename))
            ]
        else:
            return [
                # Convert the filenames to string from bytes.
                compat.as_str_any(matching_filename)
                for single_filename in filename
                for matching_filename in py_glob.glob(
                    compat.as_bytes(single_filename)
                )
            ]

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        return os.path.isdir(compat.as_bytes(dirname))

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        if not self.isdir(dirname):
            raise errors.NotFoundError(None, None, "Could not find directory")

        entries = os.listdir(compat.as_str_any(dirname))
        entries = [compat.as_str_any(item) for item in entries]
        return entries

    def makedirs(self, path):
        """Creates a directory and all parent/intermediate directories."""
        os.makedirs(path, exist_ok=True)

    def stat(self, filename):
        """Returns file statistics for a given path."""
        # NOTE: Size of the file is given by .st_size as returned from
        # os.stat(), but we convert to .length
        try:
            file_length = os.stat(compat.as_bytes(filename)).st_size
        except OSError:
            raise errors.NotFoundError(None, None, "Could not find file")
        return StatData(file_length)


class S3FileSystem:
    """Provides filesystem access to S3."""

    def __init__(self):
        if not boto3:
            raise ImportError("boto3 must be installed for S3 support.")
        self._s3_endpoint = os.environ.get("S3_ENDPOINT", None)

    def bucket_and_path(self, url):
        """Split an S3-prefixed URL into bucket and path."""
        url = compat.as_str_any(url)
        if url.startswith("s3://"):
            url = url[len("s3://") :]
        idx = url.index("/")
        bucket = url[:idx]
        path = url[(idx + 1) :]
        return bucket, path

    def exists(self, filename):
        """Determines whether a path exists or not."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        r = client.list_objects(Bucket=bucket, Prefix=path, Delimiter="/")
        if r.get("Contents") or r.get("CommonPrefixes"):
            return True
        return False

    def join(self, path, *paths):
        """Join paths with a slash."""
        return "/".join((path,) + paths)

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        s3 = boto3.resource("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        args = {}

        # For the S3 case, we use continuation tokens of the form
        # {byte_offset: number}
        offset = 0
        if continue_from is not None:
            offset = continue_from.get("byte_offset", 0)

        endpoint = ""
        if size is not None:
            # TODO(orionr): This endpoint risks splitting a multi-byte
            # character or splitting \r and \n in the case of CRLFs,
            # producing decoding errors below.
            endpoint = offset + size

        if offset != 0 or endpoint != "":
            # Asked for a range, so modify the request
            args["Range"] = "bytes={}-{}".format(offset, endpoint)

        try:
            stream = s3.Object(bucket, path).get(**args)["Body"].read()
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] in ["416", "InvalidRange"]:
                if size is not None:
                    # Asked for too much, so request just to the end. Do this
                    # in a second request so we don't check length in all cases.
                    client = boto3.client("s3", endpoint_url=self._s3_endpoint)
                    obj = client.head_object(Bucket=bucket, Key=path)
                    content_length = obj["ContentLength"]
                    endpoint = min(content_length, offset + size)
                if offset == endpoint:
                    # Asked for no bytes, so just return empty
                    stream = b""
                else:
                    args["Range"] = "bytes={}-{}".format(offset, endpoint)
                    stream = s3.Object(bucket, path).get(**args)["Body"].read()
            else:
                raise
        # `stream` should contain raw bytes here (i.e., there has been neither
        # decoding nor newline translation), so the byte offset increases by
        # the expected amount.
        continuation_token = {"byte_offset": (offset + len(stream))}
        if binary_mode:
            return (bytes(stream), continuation_token)
        else:
            return (stream.decode("utf-8"), continuation_token)

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file.

        Args:
            filename: string, a path
            file_content: string, the contents
            binary_mode: bool, write as binary if True, otherwise text
        """
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        # Always convert to bytes for writing
        if binary_mode:
            if not isinstance(file_content, bytes):
                raise TypeError("File content type must be bytes")
        else:
            file_content = compat.as_bytes(file_content)
        client.put_object(Body=file_content, Bucket=bucket, Key=path)

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        # Only support prefix with * at the end and no ? in the string
        star_i = filename.find("*")
        quest_i = filename.find("?")
        if quest_i >= 0:
            raise NotImplementedError(
                "{} not supported by compat glob".format(filename)
            )
        if star_i != len(filename) - 1:
            # Just return empty so we can use glob from directory watcher
            #
            # TODO: Remove and instead handle in GetLogdirSubdirectories.
            # However, we would need to handle it for all non-local registered
            # filesystems in some way.
            return []
        filename = filename[:-1]
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        p = client.get_paginator("list_objects")
        keys = []
        for r in p.paginate(Bucket=bucket, Prefix=path):
            for o in r.get("Contents", []):
                key = o["Key"][len(path) :]
                if key:  # Skip the base dir, which would add an empty string
                    keys.append(filename + key)
        return keys

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        if not path.endswith("/"):
            path += "/"  # This will now only retrieve subdir content
        r = client.list_objects(Bucket=bucket, Prefix=path, Delimiter="/")
        if r.get("Contents") or r.get("CommonPrefixes"):
            return True
        return False

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        p = client.get_paginator("list_objects")
        if not path.endswith("/"):
            path += "/"  # This will now only retrieve subdir content
        keys = []
        for r in p.paginate(Bucket=bucket, Prefix=path, Delimiter="/"):
            keys.extend(
                o["Prefix"][len(path) : -1] for o in r.get("CommonPrefixes", [])
            )
            for o in r.get("Contents", []):
                key = o["Key"][len(path) :]
                if key:  # Skip the base dir, which would add an empty string
                    keys.append(key)
        return keys

    def makedirs(self, dirname):
        """Creates a directory and all parent/intermediate directories."""
        if not self.exists(dirname):
            client = boto3.client("s3", endpoint_url=self._s3_endpoint)
            bucket, path = self.bucket_and_path(dirname)
            if not path.endswith("/"):
                path += "/"  # This will make sure we don't override a file
            client.put_object(Body="", Bucket=bucket, Key=path)

    def stat(self, filename):
        """Returns file statistics for a given path."""
        # NOTE: Size of the file is given by ContentLength from S3,
        # but we convert to .length
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        try:
            obj = client.head_object(Bucket=bucket, Key=path)
            return StatData(obj["ContentLength"])
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                raise errors.NotFoundError(None, None, "Could not find file")
            else:
                raise


class FSSpecFileSystem:
    """Provides filesystem access via fsspec.

    The current gfile interface doesn't map perfectly to the fsspec interface
    leading to some notable inefficiencies.

    * Reads and writes to files cause the file to be reopened each time which
      can cause a performance hit when accessing local file systems.
    * walk doesn't use the native fsspec walk function so performance may be
      slower.

    See https://github.com/tensorflow/tensorboard/issues/5286 for more info on
    limitations.
    """

    SEPARATOR = "://"
    CHAIN_SEPARATOR = "::"

    def _validate_path(self, path):
        parts = path.split(self.CHAIN_SEPARATOR)
        for part in parts[:-1]:
            if self.SEPARATOR in part:
                raise errors.InvalidArgumentError(
                    None,
                    None,
                    "fsspec URL must only have paths in the last chained filesystem, got {}".format(
                        path
                    ),
                )

    def _translate_errors(func):
        def func_wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except FileNotFoundError as e:
                raise errors.NotFoundError(None, None, str(e))

        return func_wrapper

    def _fs_path(self, filename):
        if isinstance(filename, bytes):
            filename = filename.decode("utf-8")
        self._validate_path(filename)

        fs, path = fsspec.core.url_to_fs(filename)
        return fs, path

    @_translate_errors
    def exists(self, filename):
        """Determines whether a path exists or not."""
        fs, path = self._fs_path(filename)
        return fs.exists(path)

    def _join(self, sep, paths):
        """
        _join joins the paths with the given separator.
        """
        result = []
        for part in paths:
            if part.startswith(sep):
                result = []
            if result and result[-1] and not result[-1].endswith(sep):
                result.append(sep)
            result.append(part)
        return "".join(result)

    @_translate_errors
    def join(self, path, *paths):
        """Join paths with a slash."""
        self._validate_path(path)

        before, sep, last_path = path.rpartition(self.CHAIN_SEPARATOR)
        chain_prefix = before + sep
        protocol, path = fsspec.core.split_protocol(last_path)
        fs = fsspec.get_filesystem_class(protocol)
        if protocol:
            chain_prefix += protocol + self.SEPARATOR
        return chain_prefix + self._join(fs.sep, ((path,) + paths))

    @_translate_errors
    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string.

        Args:
            filename: string, a path
            binary_mode: bool, read as binary if True, otherwise text
            size: int, number of bytes or characters to read, otherwise
                read all the contents of the file (from the continuation
                marker, if present).
            continue_from: An opaque value returned from a prior invocation of
                `read(...)` marking the last read position, so that reading
                may continue from there.  Otherwise read from the beginning.

        Returns:
            A tuple of `(data, continuation_token)` where `data' provides either
            bytes read from the file (if `binary_mode == true`) or the decoded
            string representation thereof (otherwise), and `continuation_token`
            is an opaque value that can be passed to the next invocation of
            `read(...) ' in order to continue from the last read position.
        """
        fs, path = self._fs_path(filename)

        mode = "rb" if binary_mode else "r"
        encoding = None if binary_mode else "utf8"
        if not exists(filename):
            raise errors.NotFoundError(
                None, None, "Not Found: " + compat.as_text(filename)
            )
        with fs.open(path, mode, encoding=encoding) as f:
            if continue_from is not None:
                if not f.seekable():
                    raise errors.InvalidArgumentError(
                        None,
                        None,
                        "{} is not seekable".format(filename),
                    )
                offset = continue_from.get("opaque_offset", None)
                if offset is not None:
                    f.seek(offset)

            data = f.read(size)
            # The new offset may not be `offset + len(data)`, due to decoding
            # and newline translation.
            # So, just measure it in whatever terms the underlying stream uses.
            continuation_token = (
                {"opaque_offset": f.tell()} if f.seekable() else {}
            )
            return (data, continuation_token)

    @_translate_errors
    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file.

        Args:
            filename: string, a path
            file_content: string, the contents
            binary_mode: bool, write as binary if True, otherwise text
        """
        self._write(filename, file_content, "wb" if binary_mode else "w")

    @_translate_errors
    def append(self, filename, file_content, binary_mode=False):
        """Append string file contents to a file.

        Args:
            filename: string, a path
            file_content: string, the contents to append
            binary_mode: bool, write as binary if True, otherwise text
        """
        self._write(filename, file_content, "ab" if binary_mode else "a")

    def _write(self, filename, file_content, mode):
        fs, path = self._fs_path(filename)
        encoding = None if "b" in mode else "utf8"
        with fs.open(path, mode, encoding=encoding) as f:
            compatify = compat.as_bytes if "b" in mode else compat.as_text
            f.write(compatify(file_content))

    def _get_chain_protocol_prefix(self, filename):
        chain_prefix, chain_sep, last_path = filename.rpartition(
            self.CHAIN_SEPARATOR
        )
        protocol, sep, _ = last_path.rpartition(self.SEPARATOR)
        return chain_prefix + chain_sep + protocol + sep

    @_translate_errors
    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        if isinstance(filename, bytes):
            filename = filename.decode("utf-8")

        fs, path = self._fs_path(filename)
        files = fs.glob(path)

        # check if applying the original chaining is required.
        if (
            self.SEPARATOR not in filename
            and self.CHAIN_SEPARATOR not in filename
        ):
            return files

        prefix = self._get_chain_protocol_prefix(filename)

        return [
            (
                file
                if (self.SEPARATOR in file or self.CHAIN_SEPARATOR in file)
                else prefix + file
            )
            for file in files
        ]

    @_translate_errors
    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        fs, path = self._fs_path(dirname)
        return fs.isdir(path)

    @_translate_errors
    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        fs, path = self._fs_path(dirname)
        files = fs.listdir(path, detail=False)
        files = [os.path.basename(fname) for fname in files]
        return files

    @_translate_errors
    def makedirs(self, dirname):
        """Creates a directory and all parent/intermediate directories."""
        fs, path = self._fs_path(dirname)
        return fs.makedirs(path, exist_ok=True)

    @_translate_errors
    def stat(self, filename):
        """Returns file statistics for a given path."""
        fs, path = self._fs_path(filename)
        return StatData(fs.size(path))


_FSSPEC_FILESYSTEM = FSSpecFileSystem()


def _get_fsspec_filesystem(filename):
    """
    _get_fsspec_filesystem checks if the provided protocol is known to fsspec
    and if so returns the filesystem wrapper for it.
    """
    if not FSSPEC_ENABLED:
        return None

    segment = filename.partition(FSSpecFileSystem.CHAIN_SEPARATOR)[0]
    protocol = segment.partition(FSSpecFileSystem.SEPARATOR)[0]
    if fsspec.get_filesystem_class(protocol):
        return _FSSPEC_FILESYSTEM
    else:
        return None


register_filesystem("", LocalFileSystem())
if S3_ENABLED:
    register_filesystem("s3", S3FileSystem())


class GFile:
    # Only methods needed for TensorBoard are implemented.

    def __init__(self, filename, mode):
        if mode not in ("r", "rb", "br", "w", "wb", "bw"):
            raise NotImplementedError(
                "mode {} not supported by compat GFile".format(mode)
            )
        self.filename = compat.as_bytes(filename)
        self.fs = get_filesystem(self.filename)
        self.fs_supports_append = hasattr(self.fs, "append")
        self.buff = None
        # The buffer offset and the buffer chunk size are measured in the
        # natural units of the underlying stream, i.e. bytes for binary mode,
        # or characters in text mode.
        self.buff_chunk_size = _DEFAULT_BLOCK_SIZE
        self.buff_offset = 0
        self.continuation_token = None
        self.write_temp = None
        self.write_started = False
        self.binary_mode = "b" in mode
        self.write_mode = "w" in mode
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        self.buff = None
        self.buff_offset = 0
        self.continuation_token = None

    def __iter__(self):
        return self

    def _read_buffer_to_offset(self, new_buff_offset):
        old_buff_offset = self.buff_offset
        read_size = min(len(self.buff), new_buff_offset) - old_buff_offset
        self.buff_offset += read_size
        return self.buff[old_buff_offset : old_buff_offset + read_size]

    def read(self, n=None):
        """Reads contents of file to a string.

        Args:
            n: int, number of bytes or characters to read, otherwise
                read all the contents of the file

        Returns:
            Subset of the contents of the file as a string or bytes.
        """
        if self.write_mode:
            raise errors.PermissionDeniedError(
                None, None, "File not opened in read mode"
            )

        result = None
        if self.buff and len(self.buff) > self.buff_offset:
            # read from local buffer
            if n is not None:
                chunk = self._read_buffer_to_offset(self.buff_offset + n)
                if len(chunk) == n:
                    return chunk
                result = chunk
                n -= len(chunk)
            else:
                # add all local buffer and update offsets
                result = self._read_buffer_to_offset(len(self.buff))

        # read from filesystem
        read_size = max(self.buff_chunk_size, n) if n is not None else None
        (self.buff, self.continuation_token) = self.fs.read(
            self.filename, self.binary_mode, read_size, self.continuation_token
        )
        self.buff_offset = 0

        # add from filesystem
        if n is not None:
            chunk = self._read_buffer_to_offset(n)
        else:
            # add all local buffer and update offsets
            chunk = self._read_buffer_to_offset(len(self.buff))
        result = result + chunk if result else chunk

        return result

    def write(self, file_content):
        """Writes string file contents to file, clearing contents of the file
        on first write and then appending on subsequent calls.

        Args:
            file_content: string, the contents
        """
        if not self.write_mode:
            raise errors.PermissionDeniedError(
                None, None, "File not opened in write mode"
            )
        if self.closed:
            raise errors.FailedPreconditionError(
                None, None, "File already closed"
            )

        if self.fs_supports_append:
            if not self.write_started:
                # write the first chunk to truncate file if it already exists
                self.fs.write(self.filename, file_content, self.binary_mode)
                self.write_started = True

            else:
                # append the later chunks
                self.fs.append(self.filename, file_content, self.binary_mode)
        else:
            # add to temp file, but wait for flush to write to final filesystem
            if self.write_temp is None:
                mode = "w+b" if self.binary_mode else "w+"
                self.write_temp = tempfile.TemporaryFile(mode)

            compatify = compat.as_bytes if self.binary_mode else compat.as_text
            self.write_temp.write(compatify(file_content))

    def __next__(self):
        line = None
        while True:
            if not self.buff:
                # read one unit into the buffer
                line = self.read(1)
                if line and (line[-1] == "\n" or not self.buff):
                    return line
                if not self.buff:
                    raise StopIteration()
            else:
                index = self.buff.find("\n", self.buff_offset)
                if index != -1:
                    # include line until now plus newline
                    chunk = self.read(index + 1 - self.buff_offset)
                    line = line + chunk if line else chunk
                    return line

                # read one unit past end of buffer
                chunk = self.read(len(self.buff) + 1 - self.buff_offset)
                line = line + chunk if line else chunk
                if line and (line[-1] == "\n" or not self.buff):
                    return line
                if not self.buff:
                    raise StopIteration()

    def next(self):
        return self.__next__()

    def flush(self):
        if self.closed:
            raise errors.FailedPreconditionError(
                None, None, "File already closed"
            )

        if not self.fs_supports_append:
            if self.write_temp is not None:
                # read temp file from the beginning
                self.write_temp.flush()
                self.write_temp.seek(0)
                chunk = self.write_temp.read()
                if chunk is not None:
                    # write full contents and keep in temp file
                    self.fs.write(self.filename, chunk, self.binary_mode)
                    self.write_temp.seek(len(chunk))

    def close(self):
        self.flush()
        if self.write_temp is not None:
            self.write_temp.close()
            self.write_temp = None
            self.write_started = False
        self.closed = True


def exists(filename):
    """Determines whether a path exists or not.

    Args:
      filename: string, a path

    Returns:
      True if the path exists, whether its a file or a directory.
      False if the path does not exist and there are no filesystem errors.

    Raises:
      errors.OpError: Propagates any errors reported by the FileSystem API.
    """
    return get_filesystem(filename).exists(filename)


def glob(filename):
    """Returns a list of files that match the given pattern(s).

    Args:
      filename: string or iterable of strings. The glob pattern(s).

    Returns:
      A list of strings containing filenames that match the given pattern(s).

    Raises:
      errors.OpError: If there are filesystem / directory listing errors.
    """
    return get_filesystem(filename).glob(filename)


def isdir(dirname):
    """Returns whether the path is a directory or not.

    Args:
      dirname: string, path to a potential directory

    Returns:
      True, if the path is a directory; False otherwise
    """
    return get_filesystem(dirname).isdir(dirname)


def listdir(dirname):
    """Returns a list of entries contained within a directory.

    The list is in arbitrary order. It does not contain the special entries "."
    and "..".

    Args:
      dirname: string, path to a directory

    Returns:
      [filename1, filename2, ... filenameN] as strings

    Raises:
      errors.NotFoundError if directory doesn't exist
    """
    return get_filesystem(dirname).listdir(dirname)


def makedirs(path):
    """Creates a directory and all parent/intermediate directories.

    It succeeds if path already exists and is writable.

    Args:
      path: string, name of the directory to be created
    """
    return get_filesystem(path).makedirs(path)


def walk(top, topdown=True, onerror=None):
    """Recursive directory tree generator for directories.

    Args:
      top: string, a Directory name
      topdown: bool, Traverse pre order if True, post order if False.
      onerror: optional handler for errors. Should be a function, it will be
        called with the error as argument. Rethrowing the error aborts the walk.

    Errors that happen while listing directories are ignored.

    Yields:
      Each yield is a 3-tuple:  the pathname of a directory, followed by lists
      of all its subdirectories and leaf files.
      (dirname, [subdirname, subdirname, ...], [filename, filename, ...])
      as strings
    """
    top = compat.as_str_any(top)
    fs = get_filesystem(top)
    try:
        listing = listdir(top)
    except errors.NotFoundError as err:
        if onerror:
            onerror(err)
        else:
            return

    files = []
    subdirs = []
    for item in listing:
        full_path = fs.join(top, compat.as_str_any(item))
        if isdir(full_path):
            subdirs.append(item)
        else:
            files.append(item)

    here = (top, subdirs, files)

    if topdown:
        yield here

    for subdir in subdirs:
        joined_subdir = fs.join(top, compat.as_str_any(subdir))
        for subitem in walk(joined_subdir, topdown, onerror=onerror):
            yield subitem

    if not topdown:
        yield here


def stat(filename):
    """Returns file statistics for a given path.

    Args:
      filename: string, path to a file

    Returns:
      FileStatistics struct that contains information about the path

    Raises:
      errors.OpError: If the operation fails.
    """
    return get_filesystem(filename).stat(filename)


# Used for tests only
def _write_string_to_file(filename, file_content):
    """Writes a string to a given file.

    Args:
      filename: string, path to a file
      file_content: string, contents that need to be written to the file

    Raises:
      errors.OpError: If there are errors during the operation.
    """
    with GFile(filename, mode="w") as f:
        f.write(compat.as_text(file_content))


# Used for tests only
def _read_file_to_string(filename, binary_mode=False):
    """Reads the entire contents of a file to a string.

    Args:
      filename: string, path to a file
      binary_mode: whether to open the file in binary mode or not. This changes
        the type of the object returned.

    Returns:
      contents of the file as a string or bytes.

    Raises:
      errors.OpError: Raises variety of errors that are subtypes e.g.
      `NotFoundError` etc.
    """
    if binary_mode:
        f = GFile(filename, mode="rb")
    else:
        f = GFile(filename, mode="r")
    return f.read()
