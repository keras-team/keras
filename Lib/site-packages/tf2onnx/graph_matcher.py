# SPDX-License-Identifier: Apache-2.0

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities that match patterns in a tf.Graph."""

from itertools import permutations
import six


class OpTypePattern(object):
    """A tree pattern that matches TF expressions with certain op types."""

    def __init__(self, op_type, name=None, inputs=None, allow_reorder=None):
        """Initializes an OpTypePattern.

        Args:
          op_type: string that specifies the allowed types of the root. It can be
            (1) an op type, e.g. 'Conv2D',
            (2) '*', i.e. wildcard, or
            (3) multiple op types separated by '|', e.g., 'Relu|Relu6'.
            We could use regex strings, which might be worthwhile when we have many
            similar TF op types.
          name: Optional string. The name of the pattern that can be looked up in
            MatchResult.
          inputs: Optional list of `OpTypePattern`s or strings that specify the
            patterns for the inputs of a matching op. If None, this pattern accepts
            any inputs of a matching op.
          allow_reorder: Optional boolean that overrides allow_reorder in GraphMatcher
            for this pattern's immediate inputs.
        """
        self._op_type = op_type
        self._name = name
        self.allow_reorder = allow_reorder
        if inputs is None:
            inputs = []
        self._inputs = [
            input_pattern if isinstance(input_pattern, OpTypePattern) else
            OpTypePattern(input_pattern) for input_pattern in inputs
        ]
        self.op_type_set = set(op_type.split('|')) if op_type else set()

    @property
    def op_type(self):
        return self._op_type

    @property
    def inputs(self):
        return self._inputs

    @property
    def name(self):
        return self._name


class MatchResult(object):
    r"""Encapsulates the result of a match done by GraphMatcher.

    MatchResult contains a map from OpTypePattern to the matching op and tensor.
    When the matching op has multiple output tensors, the matching tensor is the
    output tensor used by the matching op of the parent pattern. E.g., when we
    match graph

        -         +
       / \y0   y1/ \
      x    split    z
            |
            y         (nodes are ops; edges are going up)

    against add_pattern defined as

      y1_pattern = OpTypePattern('*')
      z_pattern = OpTypePattern('*')
      add_pattern = OpTypePattern('+', inputs=[y1_pattern, z_pattern])

    the matching op of `y1_pattern` is `split`, and the matching tensor of
    `y1_pattern`
    is `y1` not `y0`.
    """

    def __init__(self):
        self._pattern_to_op_tensor = {}
        self._name_to_pattern = {}

    def add(self, pattern, op, tensor):
        self._pattern_to_op_tensor[pattern] = op, tensor
        if pattern.name is not None:
            # allow this so we can apply subgraphs multiple times
            # if pattern.name in self._name_to_pattern:
            #   raise ValueError(
            #       'Name %s is already bound to another pattern' % pattern.name)
            self._name_to_pattern[pattern.name] = pattern

    def _to_pattern(self, pattern_or_name):
        if isinstance(pattern_or_name, OpTypePattern):
            return pattern_or_name

        if isinstance(pattern_or_name, six.text_type):
            return self._name_to_pattern.get(pattern_or_name)

        raise ValueError('pattern_or_name has type %s. Expect OpTypePattern or str.'
                         % type(pattern_or_name))

    def get_op(self, pattern_or_name, default=None):
        """
        For now, if the op can not be effectively obtained, then the function will return the default
        instead of an error.
        """
        op_and_tensor = self._pattern_to_op_tensor.get(self._to_pattern(pattern_or_name))
        if op_and_tensor:
            return op_and_tensor[0]
        return default

    def get_tensor(self, pattern_or_name, default=None):
        """
        For now, if the tensor can not be effectively obtained, then the function will return the default
        instead of an error.
        """
        op_and_tensor = self._pattern_to_op_tensor.get(self._to_pattern(pattern_or_name))
        if op_and_tensor:
            return op_and_tensor[1]
        return default

    def get_nodes(self):
        return [n[0] for n in self._pattern_to_op_tensor.values()]


class GraphMatcher(object):
    """Checks if a particular subgraph matches a given pattern."""

    def __init__(self, pattern, allow_reorder=False):
        """Initializes a GraphMatcher.

        Args:
          pattern: The `OpTypePattern` against which `GraphMatcher` matches
            subgraphs.
        """
        self._pattern = pattern
        self._allow_reorder = allow_reorder

    @staticmethod
    def _is_op_type_same(op, pattern):
        if pattern.op_type == "*":
            return True

        if op.type in pattern.op_type_set:
            return True

        return False

    def _match_pattern(self, pattern, op, tensor, depth=0):
        """Returns whether an TF expression rooted at `op` matches `pattern`.

        If there is a match, adds to `self._match_result` the matching op and tensor
        with key `pattern`.

        Args:
          pattern: An `OpTypePattern`.
          op: A `tf.Operation` to match against the pattern.
          tensor: the output `tf.Tensor` of `op` that is used by the matching op of
            `pattern`'s parent. Can be None if `pattern` is already the root of the
            pattern tree.

        Returns:
          if matched return True and match_list whose elem is [pattern, op, tensor]
          else return False
        the condition that op is matched with pattern:
        1 op is same:
          if pattern.op_type is None or *, then treat as same
          or op.type in pattern.op_type.split("|")
        2 op.inputs are same with pattern.inputs:
          if not pattern.inputs, then treat as same
          otherwise, iteratively compare input nodes with pattern.
        """
        #print("MATCHING", depth, pattern.op_type, op.type, tensor)
        match_list = []
        if pattern.op_type is None:
            return True, match_list

        if self._is_op_type_same(op, pattern):
            match_list.append([pattern, op, tensor])
        else:
            return False, match_list

        if not pattern.inputs:
            # If pattern.inputs is empty, skips the rest and accepts all the inputs.
            return True, match_list

        if not op or len(op.inputs) != len(pattern.inputs):
            return False, match_list

        allow_reorder = pattern.allow_reorder
        if allow_reorder is None:
            allow_reorder = self._allow_reorder
        if allow_reorder:
            pattern_inputs_list = permutations(pattern.inputs)
        else:
            pattern_inputs_list = [pattern.inputs]

        for possible_pattern_inputs in pattern_inputs_list:
            pat = list(zip(op.input, op.inputs, possible_pattern_inputs))
            match_flag_of_inputs = []
            match_lists_of_inputs = []
            for input_tensor, input_op, input_pattern in pat:
                flag, match_list_of_input = self._match_pattern(input_pattern, input_op, input_tensor, depth + 1)
                match_flag_of_inputs.append(flag)
                match_lists_of_inputs.extend(match_list_of_input)

            if all(match_flag_of_inputs):
                match_list.extend(match_lists_of_inputs)
                return True, match_list
        return False, match_list

    def _parse_match_list_to_match_result(self, match_list):
        for pattern, op, tensor in match_list:
            self._match_result.add(pattern, op, tensor)

    def match_op(self, op):
        """Matches `op` against `self._pattern`.

        Args:
          op: `tf.Operation` to match against the pattern.

        Returns:
          Returns a `MatchResult` if `op` matches the pattern; otherwise, returns
          None.
        """
        self._match_result = MatchResult()
        match_flag, match_list = self._match_pattern(self._pattern, op, tensor=None)
        if not match_flag:
            return None
        self._parse_match_list_to_match_result(match_list)
        return self._match_result

    def match_ops(self, ops):
        """Matches each operation in `ops` against `self._pattern`.

        Args:
          ops: collection of `tf.Operation` to match against the pattern.

        Yields:
          `MatchResult` for each `tf.Operation` that matches the pattern.
        """
        for op in ops:
            match_result = self.match_op(op)
            if match_result:
                yield match_result

    def match_graph(self, graph):
        """Matches each operation in `graph` against `self._pattern`.

        Args:
          graph: `tf.Graph` containing operations to match.

        Yields:
          `MatchResult` for each `tf.Operation` in `graph` that matches the pattern.
        """
        # Python 3.3.2+ implements `yield from`, but for now:
        for match_result in self.match_ops(graph.get_operations()):
            yield match_result
