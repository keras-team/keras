# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import heapq
import logging
import time
from typing import Any
import warnings

filelock: Any | None = None
try:
  import filelock  # type: ignore[no-redef]
except ImportError:
  pass

from jax._src import path as pathlib
from jax._src.compilation_cache_interface import CacheInterface

logger = logging.getLogger(__name__)


_CACHE_SUFFIX = "-cache"
_ATIME_SUFFIX = "-atime"


def _is_local_filesystem(path: str) -> bool:
  return path.startswith("file://") or "://" not in path


class LRUCache(CacheInterface):
  """Bounded cache with least-recently-used (LRU) eviction policy.

  This implementation includes cache reading, writing and eviction
  based on the LRU policy.

  Notably, when ``max_size`` is set to -1, the cache eviction
  is disabled, and the LRU cache functions as a normal cache
  without any size limitations.
  """

  def __init__(self, path: str, *, max_size: int, lock_timeout_secs: float | None = 10):
    """Args:

      path: The path to the cache directory.
      max_size: The maximum size of the cache in bytes. Caching will be
        disabled if this value is set to ``0``. A special value of ``-1``
        indicates no limit, allowing the cache size to grow indefinitely.
      lock_timeout_secs: (optional) The timeout for acquiring a file lock.
    """
    if not _is_local_filesystem(path) and not pathlib.epath_installed:
      raise RuntimeError("Please install the `etils[epath]` package to specify a cache directory on a non-local filesystem")

    self.path = self._path = pathlib.Path(path)
    self.path.mkdir(parents=True, exist_ok=True)

    self.eviction_enabled = max_size != -1  # no eviction if `max_size` is set to -1

    if self.eviction_enabled:
      if filelock is None:
        raise RuntimeError("Please install the `filelock` package to set `jax_compilation_cache_max_size`")

      self.max_size = max_size
      self.lock_timeout_secs = lock_timeout_secs

      self.lock_path = self.path / ".lockfile"
      if _is_local_filesystem(path):
        self.lock = filelock.FileLock(self.lock_path)
      else:
        self.lock = filelock.SoftFileLock(self.lock_path)

  def get(self, key: str) -> bytes | None:
    """Retrieves the cached value for the given key.

    Args:
      key: The key for which the cache value is retrieved.

    Returns:
      The cached data as bytes if available; ``None`` otherwise.
    """
    if not key:
      raise ValueError("key cannot be empty")

    cache_path = self.path / f"{key}{_CACHE_SUFFIX}"
    atime_path = self.path / f"{key}{_ATIME_SUFFIX}"

    if self.eviction_enabled:
      self.lock.acquire(timeout=self.lock_timeout_secs)

    try:
      if not cache_path.exists():
        logger.debug(f"Cache miss for key: {key!r}")
        return None

      logger.debug(f"Cache hit for key: {key!r}")

      val = cache_path.read_bytes()

      timestamp = time.time_ns().to_bytes(8, "little")
      atime_path.write_bytes(timestamp)

      return val

    finally:
      if self.eviction_enabled:
        self.lock.release()

  def put(self, key: str, val: bytes) -> None:
    """Adds a new entry to the cache.

    If a cache item with the same key already exists, no action
    will be taken, even if the value is different.

    Args:
      key: The key under which the data will be stored.
      val: The data to be stored.
    """
    if not key:
      raise ValueError("key cannot be empty")

    # prevent adding entries that exceed the maximum size limit of the cache
    if self.eviction_enabled and len(val) > self.max_size:
      msg = (f"Cache value for key {key!r} of size {len(val)} bytes exceeds "
             f"the maximum cache size of {self.max_size} bytes")
      warnings.warn(msg)
      return

    cache_path = self.path / f"{key}{_CACHE_SUFFIX}"
    atime_path = self.path / f"{key}{_ATIME_SUFFIX}"

    if self.eviction_enabled:
      self.lock.acquire(timeout=self.lock_timeout_secs)

    try:
      if cache_path.exists():
        return

      self._evict_if_needed(additional_size=len(val))

      cache_path.write_bytes(val)

      timestamp = time.time_ns().to_bytes(8, "little")
      atime_path.write_bytes(timestamp)

    finally:
      if self.eviction_enabled:
        self.lock.release()

  def _evict_if_needed(self, *, additional_size: int = 0) -> None:
    """Evicts the least recently used items from the cache if necessary
    to ensure the cache does not exceed its maximum size.

    Args:
      additional_size: The size of the new entry being added to the cache.
        This is included to account for the new entry when checking if
        eviction is needed.
    """
    if not self.eviction_enabled:
      return

    # a priority queue, each element is a tuple `(file_atime, key, file_size)`
    h: list[tuple[int, str, int]] = []
    dir_size = 0
    for cache_path in self.path.glob(f"*{_CACHE_SUFFIX}"):
      file_stat = cache_path.stat()

      # `pathlib` and `etils[epath]` have different API for obtaining the size
      # of a file, and we need to support them both.
      # See also https://github.com/google/etils/issues/630
      file_size = file_stat.st_size if not pathlib.epath_installed else file_stat.length  # pytype: disable=attribute-error

      key = cache_path.name.removesuffix(_CACHE_SUFFIX)
      atime_path = self.path / f"{key}{_ATIME_SUFFIX}"
      file_atime = int.from_bytes(atime_path.read_bytes(), "little")

      dir_size += file_size
      heapq.heappush(h, (file_atime, key, file_size))

    target_size = self.max_size - additional_size
    # evict files until the directory size is less than or equal
    # to `target_size`
    while dir_size > target_size:
      file_atime, key, file_size = heapq.heappop(h)

      logger.debug("Evicting cache entry %r: file size %d bytes, "
                   "target cache size %d bytes", key, file_size, target_size)

      cache_path = self.path / f"{key}{_CACHE_SUFFIX}"
      atime_path = self.path / f"{key}{_ATIME_SUFFIX}"

      cache_path.unlink()
      atime_path.unlink()

      dir_size -= file_size
