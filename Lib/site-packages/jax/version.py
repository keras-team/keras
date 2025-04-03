# Copyright 2018 The JAX Authors.
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

# This file is included as part of both jax and jaxlib. It is also
# eval()-ed by setup.py, so it should not have any dependencies.
from __future__ import annotations

import datetime
import os
import pathlib
import subprocess

_version = "0.5.0"
# The following line is overwritten by build scripts in distributions &
# releases. Do not modify this manually, or jax/jaxlib build will fail.
_release_version: str = '0.5.0'

# The following line is overwritten by build scripts in distributions &
# releases. Do not modify this manually, or jax/jaxlib build will fail.
_git_hash: str | None = None

def _get_version_string() -> str:
  # The build/source distribution for jax & jaxlib overwrites _release_version.
  # In this case we return it directly.
  if _release_version is not None:
    return _release_version
  return _version_from_git_tree(_version) or _version_from_todays_date(_version)


def _version_from_todays_date(base_version: str) -> str:
  datestring = datetime.date.today().strftime("%Y%m%d")
  return f"{base_version}.dev{datestring}"


def _version_from_git_tree(base_version: str) -> str | None:
  try:
    root_directory = os.path.dirname(os.path.realpath(__file__))

    # Get date string from date of most recent git commit, and the abbreviated
    # hash of that commit.
    p = subprocess.Popen(["git", "show", "-s", "--format=%at-%h", "HEAD"],
                         cwd=root_directory,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = p.communicate()
    timestamp, commit_hash = stdout.decode().strip().split('-', 1)
    datestring = datetime.date.fromtimestamp(int(timestamp)).strftime("%Y%m%d")
    assert datestring.isnumeric()
    assert commit_hash.isalnum()
  except:
    return None
  else:
    version = f"{base_version}.dev{datestring}+{commit_hash}"
    suffix = os.environ.get("JAX_CUSTOM_VERSION_SUFFIX", None)
    if suffix:
      return version + "." + suffix
    return version


def _get_version_for_build() -> str:
  """Determine the version at build time.

  The returned version string depends on which environment variables are set:
  - if JAX_RELEASE or JAXLIB_RELEASE are set: version looks like "0.4.16"
  - if JAX_NIGHTLY or JAXLIB_NIGHTLY are set: version looks like "0.4.16.dev20230906"
  - if none are set: version looks like "0.4.16.dev20230906+ge58560fdc
  """
  if _release_version is not None:
    return _release_version
  if os.environ.get('JAX_NIGHTLY') or os.environ.get('JAXLIB_NIGHTLY'):
    return _version_from_todays_date(_version)
  if os.environ.get('JAX_RELEASE') or os.environ.get('JAXLIB_RELEASE'):
    return _version
  return _version_from_git_tree(_version) or _version_from_todays_date(_version)


def _write_version(fname: str) -> None:
  """Used by setup.py to write the specified version info into the source tree."""
  release_version = _get_version_for_build()
  old_version_string = "_release_version: str = '0.5.0'"
  new_version_string = f"_release_version: str = {release_version!r}"
  fhandle = pathlib.Path(fname)
  contents = fhandle.read_text()
  # Expect two occurrences: one above, and one here.
  if contents.count(old_version_string) != 2:
    raise RuntimeError(f"Build: could not find {old_version_string!r} in {fname}")
  contents = contents.replace(old_version_string, new_version_string)

  githash = os.environ.get("JAX_GIT_HASH")
  if githash:
    old_githash_string = "_git_hash: str | None = None"
    new_githash_string = f"_git_hash: str = {githash!r}"
    if contents.count(old_githash_string) != 2:
      raise RuntimeError(f"Build: could not find {old_githash_string!r} in {fname}")
    contents = contents.replace(old_githash_string, new_githash_string)
  fhandle.write_text(contents)


def _get_cmdclass(pkg_source_path):
  from setuptools.command.build_py import build_py as build_py_orig  # pytype: disable=import-error
  from setuptools.command.sdist import sdist as sdist_orig  # pytype: disable=import-error

  class _build_py(build_py_orig):
    def run(self):
      if _release_version is None:
        this_file_in_build_dir = os.path.join(self.build_lib, pkg_source_path,
                                              os.path.basename(__file__))
        # super().run() only copies files from source -> build if they are
        # missing or outdated. Because _write_version(...) modifies the copy of
        # this file in the build tree, re-building from the same JAX directory
        # would not automatically re-copy a clean version, and _write_version
        # would fail without this deletion. See jax-ml/jax#18252.
        if os.path.isfile(this_file_in_build_dir):
          os.unlink(this_file_in_build_dir)
      super().run()
      if _release_version is None:
        _write_version(this_file_in_build_dir)

  class _sdist(sdist_orig):
    def make_release_tree(self, base_dir, files):
      super().make_release_tree(base_dir, files)
      if _release_version is None:
        _write_version(os.path.join(base_dir, pkg_source_path,
                                    os.path.basename(__file__)))

  return dict(sdist=_sdist, build_py=_build_py)


__version__ = _get_version_string()
_minimum_jaxlib_version = "0.5.0"

def _version_as_tuple(version_str):
  return tuple(int(i) for i in version_str.split(".") if i.isdigit())

__version_info__ = _version_as_tuple(__version__)
_minimum_jaxlib_version_info = _version_as_tuple(_minimum_jaxlib_version)
