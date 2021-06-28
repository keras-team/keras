# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
"""Keras API compatibility tests.

This test ensures all changes to the public API of Keras are intended.

If this test fails, it means a change has been made to the public API. Backwards
incompatible changes are not allowed. You can run the test with
"--update_goldens" flag set to "True" to update goldens when making changes to
the public Keras python API.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import argparse
import os
import re
import sys

import six

from google.protobuf import message
from google.protobuf import text_format

from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.tools.api.lib import api_objects_pb2
from tensorflow.tools.api.lib import python_object_to_proto_visitor
from tensorflow.tools.common import public_api
from tensorflow.tools.common import traverse


# FLAGS defined at the bottom:
FLAGS = None
# DEFINE_boolean, update_goldens, default False:
_UPDATE_GOLDENS_HELP = """
     Update stored golden files if API is updated. WARNING: All API changes
     have to be authorized by TensorFlow leads.
"""

# DEFINE_boolean, verbose_diffs, default True:
_VERBOSE_DIFFS_HELP = """
     If set to true, print line by line diffs on all libraries. If set to
     false, only print which libraries have differences.
"""

# Initialized with _InitPathConstants function below.
_API_GOLDEN_FOLDER_V1 = None
_API_GOLDEN_FOLDER_V2 = None


def _InitPathConstants():
  global _API_GOLDEN_FOLDER_V1
  global _API_GOLDEN_FOLDER_V2
  root_golden_path_v2 = os.path.join(
      tf.compat.v1.resource_loader.get_data_files_path(),
      '..', 'golden', 'v2', 'tensorflow.keras.pbtxt')

  if FLAGS.update_goldens:
    root_golden_path_v2 = os.path.realpath(root_golden_path_v2)
  # Get API directories based on the root golden file. This way
  # we make sure to resolve symbolic links before creating new files.
  _API_GOLDEN_FOLDER_V2 = os.path.dirname(root_golden_path_v2)
  _API_GOLDEN_FOLDER_V1 = os.path.normpath(
      os.path.join(_API_GOLDEN_FOLDER_V2, '..', 'v1'))


_TEST_README_FILE = os.path.join(
    tf.compat.v1.resource_loader.get_data_files_path(), 'README.txt')
_UPDATE_WARNING_FILE = os.path.join(
    tf.compat.v1.resource_loader.get_data_files_path(),
    'API_UPDATE_WARNING.txt')


def _KeyToFilePath(key, api_version):
  """From a given key, construct a filepath.

  Filepath will be inside golden folder for api_version.

  Args:
    key: a string used to determine the file path
    api_version: a number indicating the tensorflow API version, e.g. 1 or 2.

  Returns:
    A string of file path to the pbtxt file which describes the public API
  """

  def _ReplaceCapsWithDash(matchobj):
    match = matchobj.group(0)
    return '-%s' % (match.lower())

  case_insensitive_key = re.sub('([A-Z]{1})', _ReplaceCapsWithDash,
                                six.ensure_str(key))
  api_folder = (
      _API_GOLDEN_FOLDER_V2 if api_version == 2 else _API_GOLDEN_FOLDER_V1)
  return os.path.join(api_folder, '%s.pbtxt' % case_insensitive_key)


def _FileNameToKey(filename):
  """From a given filename, construct a key we use for api objects."""

  def _ReplaceDashWithCaps(matchobj):
    match = matchobj.group(0)
    return match[1].upper()

  base_filename = os.path.basename(filename)
  base_filename_without_ext = os.path.splitext(base_filename)[0]
  api_object_key = re.sub('((-[a-z]){1})', _ReplaceDashWithCaps,
                          six.ensure_str(base_filename_without_ext))
  return api_object_key


def _VerifyNoSubclassOfMessageVisitor(path, parent, unused_children):
  """A Visitor that crashes on subclasses of generated proto classes."""
  # If the traversed object is a proto Message class
  if not (isinstance(parent, type) and issubclass(parent, message.Message)):
    return
  if parent is message.Message:
    return
  # Check that it is a direct subclass of Message.
  if message.Message not in parent.__bases__:
    raise NotImplementedError(
        'Object tf.%s is a subclass of a generated proto Message. '
        'They are not yet supported by the API tools.' % path)


def _FilterGoldenProtoDict(golden_proto_dict, omit_golden_symbols_map):
  """Filter out golden proto dict symbols that should be omitted."""
  if not omit_golden_symbols_map:
    return golden_proto_dict
  filtered_proto_dict = dict(golden_proto_dict)
  for key, symbol_list in six.iteritems(omit_golden_symbols_map):
    api_object = api_objects_pb2.TFAPIObject()
    api_object.CopyFrom(filtered_proto_dict[key])
    filtered_proto_dict[key] = api_object
    module_or_class = None
    if api_object.HasField('tf_module'):
      module_or_class = api_object.tf_module
    elif api_object.HasField('tf_class'):
      module_or_class = api_object.tf_class
    if module_or_class is not None:
      for members in (module_or_class.member, module_or_class.member_method):
        filtered_members = [m for m in members if m.name not in symbol_list]
        # Two steps because protobuf repeated fields disallow slice assignment.
        del members[:]
        members.extend(filtered_members)
  return filtered_proto_dict


class ApiCompatibilityTest(tf.test.TestCase):

  def __init__(self, *args, **kwargs):
    super(ApiCompatibilityTest, self).__init__(*args, **kwargs)

    self._update_golden_warning = file_io.read_file_to_string(
        _UPDATE_WARNING_FILE)

    self._test_readme_message = file_io.read_file_to_string(_TEST_README_FILE)

  def _AssertProtoDictEquals(self,
                             expected_dict,
                             actual_dict,
                             verbose=False,
                             update_goldens=False,
                             additional_missing_object_message='',
                             api_version=2):
    """Diff given dicts of protobufs and report differences a readable way.

    Args:
      expected_dict: a dict of TFAPIObject protos constructed from golden files.
      actual_dict: a ict of TFAPIObject protos constructed by reading from the
        TF package linked to the test.
      verbose: Whether to log the full diffs, or simply report which files were
        different.
      update_goldens: Whether to update goldens when there are diffs found.
      additional_missing_object_message: Message to print when a symbol is
        missing.
      api_version: TensorFlow API version to test.
    """
    diffs = []
    verbose_diffs = []

    expected_keys = set(expected_dict.keys())
    actual_keys = set(actual_dict.keys())
    only_in_expected = expected_keys - actual_keys
    only_in_actual = actual_keys - expected_keys
    all_keys = expected_keys | actual_keys

    # This will be populated below.
    updated_keys = []

    for key in all_keys:
      diff_message = ''
      verbose_diff_message = ''
      # First check if the key is not found in one or the other.
      if key in only_in_expected:
        diff_message = 'Object %s expected but not found (removed). %s' % (
            key, additional_missing_object_message)
        verbose_diff_message = diff_message
      elif key in only_in_actual:
        diff_message = 'New object %s found (added).' % key
        verbose_diff_message = diff_message
      else:
        # Do not truncate diff
        self.maxDiff = None  # pylint: disable=invalid-name
        # Now we can run an actual proto diff.
        try:
          self.assertProtoEquals(expected_dict[key], actual_dict[key])
        except AssertionError as e:
          updated_keys.append(key)
          diff_message = 'Change detected in python object: %s.' % key
          verbose_diff_message = str(e)

      # All difference cases covered above. If any difference found, add to the
      # list.
      if diff_message:
        diffs.append(diff_message)
        verbose_diffs.append(verbose_diff_message)

    # If diffs are found, handle them based on flags.
    if diffs:
      diff_count = len(diffs)
      logging.error(self._test_readme_message)
      logging.error('%d differences found between API and golden.', diff_count)

      if update_goldens:
        # Write files if requested.
        logging.warning(self._update_golden_warning)

        # If the keys are only in expected, some objects are deleted.
        # Remove files.
        for key in only_in_expected:
          filepath = _KeyToFilePath(key, api_version)
          tf.io.gfile.remove(filepath)

        # If the files are only in actual (current library), these are new
        # modules. Write them to files. Also record all updates in files.
        for key in only_in_actual | set(updated_keys):
          filepath = _KeyToFilePath(key, api_version)
          file_io.write_string_to_file(
              filepath, text_format.MessageToString(actual_dict[key]))
      else:
        # Include the actual differences to help debugging.
        for d, verbose_d in zip(diffs, verbose_diffs):
          logging.error('    %s', d)
          logging.error('    %s', verbose_d)
        # Fail if we cannot fix the test by updating goldens.
        self.fail('%d differences found between API and golden.' % diff_count)

    else:
      logging.info('No differences found between API and golden.')

  def _checkBackwardsCompatibility(self,
                                   root,
                                   golden_file_patterns,
                                   api_version,
                                   additional_private_map=None,
                                   omit_golden_symbols_map=None):
    # Extract all API stuff.
    visitor = python_object_to_proto_visitor.PythonObjectToProtoVisitor(
        default_path='tensorflow.keras')

    public_api_visitor = public_api.PublicAPIVisitor(visitor)
    if additional_private_map:
      public_api_visitor.private_map.update(additional_private_map)
    public_api_visitor.set_root_name('tf.keras')

    traverse.traverse(root, public_api_visitor)
    proto_dict = visitor.GetProtos()

    # Read all golden files.
    golden_file_list = tf.compat.v1.gfile.Glob(golden_file_patterns)

    def _ReadFileToProto(filename):
      """Read a filename, create a protobuf from its contents."""
      ret_val = api_objects_pb2.TFAPIObject()
      text_format.Merge(file_io.read_file_to_string(filename), ret_val)
      return ret_val

    golden_proto_dict = {
        _FileNameToKey(filename): _ReadFileToProto(filename)
        for filename in golden_file_list
    }
    golden_proto_dict = _FilterGoldenProtoDict(golden_proto_dict,
                                               omit_golden_symbols_map)

    # Diff them. Do not fail if called with update.
    # If the test is run to update goldens, only report diffs but do not fail.
    self._AssertProtoDictEquals(
        golden_proto_dict,
        proto_dict,
        verbose=FLAGS.verbose_diffs,
        update_goldens=FLAGS.update_goldens,
        api_version=api_version)

  def testAPIBackwardsCompatibility(self):
    api_version = 1
    if hasattr(tf, '_major_api_version') and tf._major_api_version == 2:
      api_version = 2
    golden_file_patterns = [
        os.path.join(
            tf.compat.v1.resource_loader.get_root_dir_with_all_resources(),
            _KeyToFilePath('*', api_version))]

    self._checkBackwardsCompatibility(
        tf.keras,
        golden_file_patterns,
        api_version,
        # Skip compat.v1 and compat.v2 since they are validated
        # in separate tests.
        additional_private_map={'tf.compat': ['v1', 'v2']},
        omit_golden_symbols_map={})

  def testAPIBackwardsCompatibilityV1(self):
    api_version = 1
    golden_file_patterns = os.path.join(
        tf.compat.v1.resource_loader.get_root_dir_with_all_resources(),
        _KeyToFilePath('*', api_version))
    self._checkBackwardsCompatibility(
        tf.compat.v1.keras,
        golden_file_patterns,
        api_version,
        additional_private_map={
            'tf': ['pywrap_tensorflow'],
            'tf.compat': ['v1', 'v2'],
        },
        omit_golden_symbols_map={})

  def testAPIBackwardsCompatibilityV2(self):
    api_version = 2
    golden_file_patterns = [os.path.join(
        tf.compat.v1.resource_loader.get_root_dir_with_all_resources(),
        _KeyToFilePath('*', api_version))]
    self._checkBackwardsCompatibility(
        tf.compat.v2.keras,
        golden_file_patterns,
        api_version,
        additional_private_map={'tf.compat': ['v1', 'v2']},
        omit_golden_symbols_map={})


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--update_goldens', type=bool, default=False, help=_UPDATE_GOLDENS_HELP)
  parser.add_argument(
      '--verbose_diffs', type=bool, default=True, help=_VERBOSE_DIFFS_HELP)
  FLAGS, unparsed = parser.parse_known_args()
  _InitPathConstants()

  # Now update argv, so that unittest library does not get confused.
  sys.argv = [sys.argv[0]] + unparsed
  tf.test.main()
