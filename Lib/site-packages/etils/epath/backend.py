# Copyright 2024 The etils Authors.
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

"""`os.path` API backend."""

from __future__ import annotations

import abc
import contextlib
import datetime
import functools
import glob as glob_lib
import os
import pathlib
import shutil
import stat as stat_lib
import typing
from typing import Callable, Iterator, NoReturn, Optional, Union

from etils.epath import stat_utils
from etils.epath.typing import PathLike  # pylint: disable=g-importing-member

if typing.TYPE_CHECKING:
  import fsspec


class Backend(abc.ABC):
  """Abstract backend class."""

  @abc.abstractmethod
  def open(
      self,
      path: PathLike,
      mode: str,
  ) -> typing.IO[Union[str, bytes]]:
    """`open`. Encoding should be utf-8."""
    raise NotImplementedError

  @abc.abstractmethod
  def exists(self, path: PathLike) -> bool:
    raise NotImplementedError

  @abc.abstractmethod
  def isdir(self, path: PathLike) -> bool:
    raise NotImplementedError

  @abc.abstractmethod
  def listdir(self, path: PathLike) -> list[str]:
    raise NotImplementedError

  @abc.abstractmethod
  def glob(self, path: PathLike) -> list[str]:
    raise NotImplementedError

  @abc.abstractmethod
  def walk(
      self,
      top: PathLike,
      *,
      top_down: bool = True,
      on_error: Callable[[OSError], object] | None = None,
  ) -> Iterator[tuple[PathLike, list[str], list[str]]]:
    raise NotImplementedError

  @abc.abstractmethod
  def makedirs(
      self,
      path: PathLike,
      *,
      exist_ok: bool = False,
      mode: Optional[int] = None,
  ) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def mkdir(
      self,
      path: PathLike,
      *,
      exist_ok: bool = False,
      mode: Optional[int] = None,
  ) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def rmtree(self, path: PathLike) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def remove(self, path: PathLike) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def rename(self, path: PathLike, dst: PathLike) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def replace(self, path: PathLike, dst: PathLike) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def copy(self, path: PathLike, dst: PathLike, overwrite: bool) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def stat(self, path: PathLike) -> stat_utils.StatResult:
    raise NotImplementedError


class _OsPathBackend(Backend):
  """`os.path` backend."""

  def open(
      self,
      path: PathLike,
      mode: str,
  ) -> typing.IO[Union[str, bytes]]:
    if 'b' in mode:
      encoding = None
    else:
      encoding = 'utf-8'
    return open(path, mode, encoding=encoding)

  def exists(self, path: PathLike) -> bool:
    return os.path.exists(path)

  def isdir(self, path: PathLike) -> bool:
    return os.path.isdir(path)

  def listdir(self, path: PathLike) -> list[str]:
    # GFile filter backup files per default.
    return [p for p in os.listdir(path) if not p.endswith('~')]

  def glob(self, path: PathLike) -> list[str]:
    return glob_lib.glob(path)

  def walk(
      self,
      top: PathLike,
      *,
      top_down: bool = True,
      on_error: Callable[[OSError], object] | None = None,
  ) -> Iterator[tuple[PathLike, list[str], list[str]]]:
    if hasattr(pathlib.Path, 'walk'):  # Python 3.12
      yield from pathlib.Path(top).walk(top_down=top_down, on_error=on_error)
    else:  # Backward compatibility
      # Note that `os.walk` is inconsistent for `symlinks` (always marked as
      # filenames), but should be fine.
      yield from os.walk(top, topdown=top_down, onerror=on_error)

  def makedirs(
      self,
      path: PathLike,
      *,
      exist_ok: bool = False,
      mode: Optional[int] = None,
  ) -> None:
    mode = 0o777 if mode is None else mode
    os.makedirs(path, exist_ok=exist_ok, mode=mode)

  def mkdir(
      self,
      path: PathLike,
      *,
      exist_ok: bool = False,
      mode: Optional[int] = None,
  ) -> None:
    mode = 0o777 if mode is None else mode
    try:
      os.mkdir(path, mode=mode)
    except FileExistsError:
      if self.isdir(path):  # No-op if directory already exists
        if exist_ok:
          pass
        else:  # Overwriting file raise an error
          raise
      else:
        raise

  def rmtree(self, path: PathLike) -> None:
    try:
      shutil.rmtree(path)
    except NotADirectoryError:
      self.remove(path)

  def remove(self, path: PathLike) -> None:
    try:
      os.remove(path)
    except IsADirectoryError:
      os.rmdir(path)
    except PermissionError:
      # On Mac, `PermissionError` is raised instead of `IsADirectoryError`
      if self.isdir(path):
        os.rmdir(path)
      else:
        raise

  def rename(self, path: PathLike, dst: PathLike) -> None:
    if self.exists(dst):
      raise FileExistsError(
          f'Cannot rename {path}. Destination {dst} already exists.'
      )
    os.rename(path, dst)

  def replace(self, path: PathLike, dst: PathLike) -> None:
    if self.isdir(dst):
      raise IsADirectoryError(f'Cannot overwrite: {dst} is a directory')
    os.replace(path, dst)

  def copy(self, path: PathLike, dst: PathLike, overwrite: bool) -> None:
    if not overwrite and self.exists(dst):
      raise FileExistsError(f'{dst} already exists. Cannot copy {path}.')
    shutil.copyfile(path, dst)

  def stat(self, path: PathLike) -> stat_utils.StatResult:
    st = os.stat(path)
    if os.name == 'nt':
      owner = None
      group = None
    else:
      import grp  # pylint: disable=g-import-not-at-top
      import pwd  # pylint: disable=g-import-not-at-top

      owner = pwd.getpwuid(st.st_uid).pw_name
      group = grp.getgrgid(st.st_gid).gr_name

    return stat_utils.StatResult(
        is_directory=stat_lib.S_ISDIR(st.st_mode),
        length=st.st_size,
        mtime=int(st.st_mtime),
        owner=owner,
        group=group,
        mode=st.st_mode,
    )


class _TfBackend(Backend):
  """TensorFlow backend."""

  @property
  def tf(self):
    try:
      import tensorflow  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    except ImportError as e:
      raise ImportError(
          f'{e}. To use epath.Path with gs://, TensorFlow should be installed.'
      ) from None
    return tensorflow

  @property
  def gfile(self):
    return self.tf.io.gfile

  @contextlib.contextmanager
  def open(
      self,
      path: PathLike,
      mode: str,
  ) -> Iterator[typing.IO[Union[str, bytes]]]:
    with self.gfile.GFile(path, mode) as f:  # pytype: disable=bad-return-type
      try:
        yield f
      except self.tf.errors.NotFoundError as e:
        raise FileNotFoundError(e) from None

  def exists(self, path: PathLike) -> bool:
    return self.gfile.exists(path)

  def isdir(self, path: PathLike) -> bool:
    return self.gfile.isdir(path)

  def listdir(self, path: PathLike) -> list[str]:
    return self.gfile.listdir(path)

  def glob(self, path: PathLike) -> list[str]:
    return self.gfile.glob(path)

  def walk(
      self,
      top: PathLike,
      *,
      top_down: bool = True,
      on_error: Callable[[OSError], object] | None = None,
  ) -> Iterator[tuple[PathLike, list[str], list[str]]]:
    yield from self.gfile.walk(top, topdown=top_down, onerror=on_error)

  def makedirs(
      self,
      path: PathLike,
      *,
      exist_ok: bool = False,
      mode: Optional[int] = None,
  ) -> None:
    mode = 0o777 if mode is None else mode
    if mode != 0o777:
      # tf.io.gfile do not support setting `mode=`
      raise NotImplementedError(
          'makedirs with custom `mode=` not supported for tf.io.gfile backend.'
          ' Please open an issue.'
      )
    # TF do not have a `exist_ok=` kwargs, so have to first check existence.
    # This has performance impact but can be disabled with `exist_ok=True`.
    if not exist_ok and self.exists(path):
      raise FileExistsError(f'{path} already exists.')

    try:
      self.gfile.makedirs(path)
    except self.tf.errors.FailedPreconditionError as e:
      if 'not a directory' in str(e):
        raise FileExistsError(str(e)) from None
      else:
        raise OSError(str(e)) from None

  def mkdir(
      self,
      path: PathLike,
      *,
      exist_ok: bool = False,
      mode: Optional[int] = None,
  ) -> None:
    mode = 0o777 if mode is None else mode
    if mode != 0o777:
      # tf.io.gfile do not support setting `mode=`
      raise NotImplementedError(
          'mkdir with custom `mode=` not supported for tf.io.gfile backend.'
          ' Please open an issue.'
      )

    if not exist_ok and self.exists(path):
      raise FileExistsError(f'{path} already exists.')

    try:
      self.gfile.mkdir(path)
    except self.tf.errors.NotFoundError as e:
      raise FileNotFoundError(str(e)) from None
    else:
      if not self.isdir(path):  # TF do not raises error for files
        raise FileExistsError(f'Cannot create dir. {path} is not a directory')

  def rmtree(self, path: PathLike) -> None:
    try:
      self.gfile.rmtree(path)
    except self.tf.errors.NotFoundError as e:
      raise FileNotFoundError(str(e)) from None

  def remove(self, path: PathLike) -> None:
    try:
      self.gfile.remove(path)
    except self.tf.errors.FailedPreconditionError as e:  # Dir not empty
      raise OSError(str(e)) from None
    except self.tf.errors.NotFoundError as e:
      raise FileNotFoundError(str(e)) from None

  def rename(self, path: PathLike, dst: PathLike) -> None:
    try:
      self.gfile.rename(path, dst)
    except self.tf.errors.OpError as e:
      self._reraise_error(e)

  def replace(self, path: PathLike, dst: PathLike) -> None:
    try:
      self.gfile.rename(path, dst, overwrite=True)
    except self.tf.errors.OpError as e:
      self._reraise_error(e)

  def copy(self, path: PathLike, dst: PathLike, overwrite: bool) -> None:
    if overwrite and self.isdir(dst):  # For consistency with rename, replace
      raise IsADirectoryError(
          f'Cannot copy {path}. Destination {dst} is a directory'
      ) from None
    try:
      self.gfile.copy(path, dst, overwrite=overwrite)
    except self.tf.errors.OpError as e:
      self._reraise_error(e)

  def _reraise_error(self, e) -> NoReturn:
    """Reraise the TF error."""
    e_msg = str(e)
    if isinstance(e, self.tf.errors.FailedPreconditionError):
      if 'not a directory' in e_msg.lower():
        raise NotADirectoryError(e_msg) from None
      if 'is a directory' in e_msg.lower():
        raise IsADirectoryError(e_msg) from None
      else:
        raise OSError(e_msg) from None
    if isinstance(e, self.tf.errors.AlreadyExistsError):
      e_msg = str(e)
      if 'is a directory' in e_msg.lower():
        raise IsADirectoryError(e_msg) from None
      else:
        raise FileExistsError(e_msg) from None
    if isinstance(e, self.tf.errors.NotFoundError):
      raise FileNotFoundError(e_msg) from None
    else:
      raise  # pylint: disable=misplaced-bare-raise

  def stat(self, path: PathLike) -> stat_utils.StatResult:
    st = self.gfile.stat(path)
    return stat_utils.StatResult(
        is_directory=st.is_directory,
        length=st.length,
        mtime=st.mtime_nsec // 1_000_000_000,
        owner=None,  # Not available.
        group=None,  # Not available.
        mode=None,
    )


class _FileSystemSpecBackend(Backend):
  """FileSystemSpec backend entirely relying on fsspec."""

  @functools.lru_cache()
  def _get_filesystem(self, name: str) -> fsspec.AbstractFileSystem:
    """Caches the filesystem."""
    try:
      import fsspec  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      raise ImportError(
          "To use epath.Path with gs://, fsspec should be installed.'"
      ) from e
    return fsspec.filesystem(name)

  def fs(self, path: PathLike) -> fsspec.AbstractFileSystem:
    """Returns the proper fsspec filsystem: GCS, S3 or file."""
    path = os.fspath(path)
    if path.startswith('gs://'):
      return self._get_filesystem('gcs')
    elif path.startswith('s3://'):
      return self._get_filesystem('s3')
    elif path.startswith('az://'):
      return self._get_filesystem('az')
    elif path.startswith('hf://'):
      return self._get_filesystem('hf')
    else:
      return self._get_filesystem('file')

  def open(self, path: PathLike, mode: str) -> typing.IO[Union[str, bytes]]:
    return self.fs(path).open(path, mode=mode)

  def exists(self, path: PathLike) -> bool:
    return self.fs(path).exists(path)

  def isdir(self, path: PathLike) -> bool:
    return self.fs(path).isdir(path)

  def listdir(self, path: PathLike) -> list[str]:
    paths = self.fs(path).listdir(path, detail=False)
    return [os.path.basename(p) for p in paths if not p.endswith('~')]

  def glob(self, path: PathLike) -> list[str]:
    protocol = _get_protocol(path)
    return [protocol + p for p in self.fs(path).glob(path)]

  def walk(
      self,
      top: PathLike,
      *,
      top_down: bool = True,
      on_error: Callable[[OSError], object] | None = None,
  ) -> Iterator[tuple[PathLike, list[str], list[str]]]:

    if on_error is None:
      on_error = 'omit'  # default behavior for pathlib.Path.walk
    yield from self.fs(top).walk(  # pytype: disable=bad-return-type
        top,
        topdown=top_down,
        on_error=on_error,
        max_depth=None,
    )

  def makedirs(
      self,
      path: PathLike,
      *,
      exist_ok: bool = False,
      mode: Optional[int] = None,
  ) -> None:
    mode = 0o777 if mode is None else mode
    if mode != 0o777:
      # FileSystemSpec backend do not support setting `mode=`
      raise NotImplementedError(
          'makedirs with custom `mode=` not supported for FileSystemSpec'
          ' backend. Please open an issue.'
      )
    return self.fs(path).makedirs(path, exist_ok=exist_ok)

  def mkdir(
      self,
      path: PathLike,
      *,
      exist_ok: bool = False,
      mode: Optional[int] = None,
  ) -> None:
    mode = 0o777 if mode is None else mode
    if mode != 0o777:
      # FileSystemSpec backend do not support setting `mode=`
      raise NotImplementedError(
          'mkdir with custom `mode=` not supported for FileSystemSpec backend.'
          ' Please open an issue.'
      )
    try:
      return self.fs(path).mkdir(path, create_parents=False)
    except FileExistsError:
      if exist_ok and self.isdir(path):
        return
      raise FileExistsError(
          f'The operation failed because the specified {path=} already exists.'
      ) from None

  def rmtree(self, path: PathLike) -> None:
    return self.fs(path).rm(path, recursive=True)

  def remove(self, path: PathLike) -> None:
    try:
      return self.fs(path).rm(path, recursive=False)
    except (IsADirectoryError, ValueError):
      return self.fs(path).rmdir(path)

  def rename(self, path: PathLike, dst: PathLike) -> None:
    if self.exists(dst):
      raise FileExistsError(f'{dst} already exists. Cannot rename {path}.')
    if self.isdir(path) and not self.isdir(dst):
      if self.exists(dst):
        raise FileExistsError(
            f'Cannot rename a file ({path}) to a directory ({dst})'
        )
    # Check that `dst` is in an existing folder
    dst_folder = os.path.dirname(dst)
    if not self.exists(dst_folder):
      raise FileNotFoundError(f'folder {dst_folder} does not exist')
    # Stringify paths, because PosixPaths do not implement len:
    path, dst = os.fspath(path), os.fspath(dst)
    return self.fs(path).rename(path, dst, recursive=True)

  def replace(self, path: PathLike, dst: PathLike) -> None:
    if self.isdir(dst):
      raise IsADirectoryError(f'Cannot overwrite: {dst} is a directory')
    if not self.exists(path):
      raise FileNotFoundError(f'Cannot replace: path {path} does not exist')
    if self.isdir(path) and self.exists(dst):
      raise NotADirectoryError(
          f'Cannot replace a directory {path} by a file {dst}'
      )
    # Check that `dst` is in an existing folder
    dst_folder = os.path.dirname(dst)
    if not self.exists(dst_folder):
      raise FileNotFoundError(f'folder {dst_folder} does not exist')
    # Stringify paths, because PosixPaths do not implement len:
    path, dst = os.fspath(path), os.fspath(dst)
    return self.fs(path).rename(path, dst, recursive=True)

  def copy(self, path: PathLike, dst: PathLike, overwrite: bool) -> None:
    if not overwrite and self.exists(dst):
      raise FileExistsError(f'{dst} already exists. Cannot copy {path}.')
    is_dir_dst = self.isdir(dst)
    if self.isdir(path) and not is_dir_dst:
      raise IsADirectoryError(
          f'Cannot copy to {dst}. Path {path} is a directory'
      )
    if overwrite and is_dir_dst:
      raise IsADirectoryError(
          f'Cannot overwrite {path}. Destination {dst} is a directory'
      )
    # Stringify paths, because PosixPaths do not implement len:
    path, dst = os.fspath(path), os.fspath(dst)
    chunksize = 1024 * 1024  # 1 MB
    with self.open(path, mode='rb') as src:
      with self.open(dst, mode='wb') as dst:
        while True:
          data = src.read(chunksize)
          if not data:
            break
          dst.write(data)

  def stat(self, path: PathLike) -> stat_utils.StatResult:
    info = self.fs(path).info(path)

    mtime_obj = info.get('mtime')
    if mtime_obj is None:
      mtime = 0
    elif isinstance(mtime_obj, datetime.datetime):
      # obtain the POSIX timestamp
      mtime_timestamp = mtime_obj.timestamp()
      # convert to sec since the epoch
      mtime = int(mtime_timestamp)
    elif isinstance(mtime_obj, (int, float)):
      mtime = int(mtime_obj)
    else:
      raise RuntimeError(f'Unsupported mtime type: {type(mtime_obj)}')

    return stat_utils.StatResult(
        is_directory=info.get('type') == 'directory',
        length=info.get('size'),
        mtime=mtime,
        owner=info.get('owner'),
        group=info.get('group'),
        mode=info.get('mode'),
    )


def _get_protocol(path: PathLike) -> str:
  """Extract the protocol."""
  path = os.fspath(path)
  if '://' in path:
    return path.split('://', 1)[0] + '://'
  else:
    return ''


tf_backend = _TfBackend()
os_backend = _OsPathBackend()
fsspec_backend = _FileSystemSpecBackend()
