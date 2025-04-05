# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for utilities working with arbitrarily nested structures."""

import collections
import doctest
import types
from typing import Any, Iterator, Mapping
import unittest

from absl.testing import parameterized
import attr
import numpy as np
import tree
import wrapt

STRUCTURE1 = (((1, 2), 3), 4, (5, 6))
STRUCTURE2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
STRUCTURE_DIFFERENT_NUM_ELEMENTS = ("spam", "eggs")
STRUCTURE_DIFFERENT_NESTING = (((1, 2), 3), 4, 5, (6,))


class DoctestTest(parameterized.TestCase):

  def testDoctest(self):
    extraglobs = {
        "collections": collections,
        "tree": tree,
    }
    num_failed, num_attempted = doctest.testmod(
        tree, extraglobs=extraglobs, optionflags=doctest.ELLIPSIS)
    self.assertGreater(num_attempted, 0, "No doctests found.")
    self.assertEqual(num_failed, 0, "{} doctests failed".format(num_failed))


class NestTest(parameterized.TestCase):

  def assertAllEquals(self, a, b):
    self.assertTrue((np.asarray(a) == b).all())

  def testAttrsFlattenAndUnflatten(self):

    class BadAttr(object):
      """Class that has a non-iterable __attrs_attrs__."""
      __attrs_attrs__ = None

    @attr.s
    class SampleAttr(object):
      field1 = attr.ib()
      field2 = attr.ib()

    field_values = [1, 2]
    sample_attr = SampleAttr(*field_values)
    self.assertFalse(tree._is_attrs(field_values))
    self.assertTrue(tree._is_attrs(sample_attr))
    flat = tree.flatten(sample_attr)
    self.assertEqual(field_values, flat)
    restructured_from_flat = tree.unflatten_as(sample_attr, flat)
    self.assertIsInstance(restructured_from_flat, SampleAttr)
    self.assertEqual(restructured_from_flat, sample_attr)

    # Check that flatten fails if attributes are not iterable
    with self.assertRaisesRegex(TypeError, "object is not iterable"):
      flat = tree.flatten(BadAttr())

  @parameterized.parameters([
      (1, 2, 3),
      ({"B": 10, "A": 20}, [1, 2], 3),
      ((1, 2), [3, 4], 5),
      (collections.namedtuple("Point", ["x", "y"])(1, 2), 3, 4),
      wrapt.ObjectProxy(
          (collections.namedtuple("Point", ["x", "y"])(1, 2), 3, 4))
  ])
  def testAttrsMapStructure(self, *field_values):
    @attr.s
    class SampleAttr(object):
      field3 = attr.ib()
      field1 = attr.ib()
      field2 = attr.ib()

    structure = SampleAttr(*field_values)
    new_structure = tree.map_structure(lambda x: x, structure)
    self.assertEqual(structure, new_structure)

  def testFlattenAndUnflatten(self):
    structure = ((3, 4), 5, (6, 7, (9, 10), 8))
    flat = ["a", "b", "c", "d", "e", "f", "g", "h"]
    self.assertEqual(tree.flatten(structure), [3, 4, 5, 6, 7, 9, 10, 8])
    self.assertEqual(
        tree.unflatten_as(structure, flat),
        (("a", "b"), "c", ("d", "e", ("f", "g"), "h")))
    point = collections.namedtuple("Point", ["x", "y"])
    structure = (point(x=4, y=2), ((point(x=1, y=0),),))
    flat = [4, 2, 1, 0]
    self.assertEqual(tree.flatten(structure), flat)
    restructured_from_flat = tree.unflatten_as(structure, flat)
    self.assertEqual(restructured_from_flat, structure)
    self.assertEqual(restructured_from_flat[0].x, 4)
    self.assertEqual(restructured_from_flat[0].y, 2)
    self.assertEqual(restructured_from_flat[1][0][0].x, 1)
    self.assertEqual(restructured_from_flat[1][0][0].y, 0)

    self.assertEqual([5], tree.flatten(5))
    self.assertEqual([np.array([5])], tree.flatten(np.array([5])))

    self.assertEqual("a", tree.unflatten_as(5, ["a"]))
    self.assertEqual(
        np.array([5]), tree.unflatten_as("scalar", [np.array([5])]))

    with self.assertRaisesRegex(ValueError, "Structure is a scalar"):
      tree.unflatten_as("scalar", [4, 5])

    with self.assertRaisesRegex(TypeError, "flat_sequence"):
      tree.unflatten_as([4, 5], "bad_sequence")

    with self.assertRaises(ValueError):
      tree.unflatten_as([5, 6, [7, 8]], ["a", "b", "c"])

  def testFlattenDictOrder(self):
    ordered = collections.OrderedDict([("d", 3), ("b", 1), ("a", 0), ("c", 2)])
    plain = {"d": 3, "b": 1, "a": 0, "c": 2}
    ordered_flat = tree.flatten(ordered)
    plain_flat = tree.flatten(plain)
    self.assertEqual([0, 1, 2, 3], ordered_flat)
    self.assertEqual([0, 1, 2, 3], plain_flat)

  def testUnflattenDictOrder(self):
    ordered = collections.OrderedDict([("d", 0), ("b", 0), ("a", 0), ("c", 0)])
    plain = {"d": 0, "b": 0, "a": 0, "c": 0}
    seq = [0, 1, 2, 3]
    ordered_reconstruction = tree.unflatten_as(ordered, seq)
    plain_reconstruction = tree.unflatten_as(plain, seq)
    self.assertEqual(
        collections.OrderedDict([("d", 3), ("b", 1), ("a", 0), ("c", 2)]),
        ordered_reconstruction)
    self.assertEqual({"d": 3, "b": 1, "a": 0, "c": 2}, plain_reconstruction)

  def testFlattenAndUnflatten_withDicts(self):
    # A nice messy mix of tuples, lists, dicts, and `OrderedDict`s.
    named_tuple = collections.namedtuple("A", ("b", "c"))
    mess = [
        "z",
        named_tuple(3, 4),
        {
            "c": [
                1,
                collections.OrderedDict([
                    ("b", 3),
                    ("a", 2),
                ]),
            ],
            "b": 5
        },
        17
    ]

    flattened = tree.flatten(mess)
    self.assertEqual(flattened, ["z", 3, 4, 5, 1, 2, 3, 17])

    structure_of_mess = [
        14,
        named_tuple("a", True),
        {
            "c": [
                0,
                collections.OrderedDict([
                    ("b", 9),
                    ("a", 8),
                ]),
            ],
            "b": 3
        },
        "hi everybody",
    ]

    self.assertEqual(mess, tree.unflatten_as(structure_of_mess, flattened))

    # Check also that the OrderedDict was created, with the correct key order.
    unflattened_ordered_dict = tree.unflatten_as(
        structure_of_mess, flattened)[2]["c"][1]
    self.assertIsInstance(unflattened_ordered_dict, collections.OrderedDict)
    self.assertEqual(list(unflattened_ordered_dict.keys()), ["b", "a"])

  def testFlatten_numpyIsNotFlattened(self):
    structure = np.array([1, 2, 3])
    flattened = tree.flatten(structure)
    self.assertLen(flattened, 1)

  def testFlatten_stringIsNotFlattened(self):
    structure = "lots of letters"
    flattened = tree.flatten(structure)
    self.assertLen(flattened, 1)
    self.assertEqual(structure, tree.unflatten_as("goodbye", flattened))

  def testFlatten_bytearrayIsNotFlattened(self):
    structure = bytearray("bytes in an array", "ascii")
    flattened = tree.flatten(structure)
    self.assertLen(flattened, 1)
    self.assertEqual(flattened, [structure])
    self.assertEqual(structure,
                     tree.unflatten_as(bytearray("hello", "ascii"), flattened))

  def testUnflattenSequenceAs_notIterableError(self):
    with self.assertRaisesRegex(TypeError, "flat_sequence must be a sequence"):
      tree.unflatten_as("hi", "bye")

  def testUnflattenSequenceAs_wrongLengthsError(self):
    with self.assertRaisesRegex(
        ValueError,
        "Structure had 2 elements, but flat_sequence had 3 elements."):
      tree.unflatten_as(["hello", "world"], ["and", "goodbye", "again"])

  def testUnflattenSequenceAs_defaultdict(self):
    structure = collections.defaultdict(
        list, [("a", [None]), ("b", [None, None])])
    sequence = [1, 2, 3]
    expected = collections.defaultdict(
        list, [("a", [1]), ("b", [2, 3])])
    self.assertEqual(expected, tree.unflatten_as(structure, sequence))

  def testIsSequence(self):
    self.assertFalse(tree.is_nested("1234"))
    self.assertFalse(tree.is_nested(b"1234"))
    self.assertFalse(tree.is_nested(u"1234"))
    self.assertFalse(tree.is_nested(bytearray("1234", "ascii")))
    self.assertTrue(tree.is_nested([1, 3, [4, 5]]))
    self.assertTrue(tree.is_nested(((7, 8), (5, 6))))
    self.assertTrue(tree.is_nested([]))
    self.assertTrue(tree.is_nested({"a": 1, "b": 2}))
    self.assertFalse(tree.is_nested(set([1, 2])))
    ones = np.ones([2, 3])
    self.assertFalse(tree.is_nested(ones))
    self.assertFalse(tree.is_nested(np.tanh(ones)))
    self.assertFalse(tree.is_nested(np.ones((4, 5))))

  # pylint does not correctly recognize these as class names and
  # suggests to use variable style under_score naming.
  # pylint: disable=invalid-name
  Named0ab = collections.namedtuple("named_0", ("a", "b"))
  Named1ab = collections.namedtuple("named_1", ("a", "b"))
  SameNameab = collections.namedtuple("same_name", ("a", "b"))
  SameNameab2 = collections.namedtuple("same_name", ("a", "b"))
  SameNamexy = collections.namedtuple("same_name", ("x", "y"))
  SameName1xy = collections.namedtuple("same_name_1", ("x", "y"))
  SameName1xy2 = collections.namedtuple("same_name_1", ("x", "y"))
  NotSameName = collections.namedtuple("not_same_name", ("a", "b"))
  # pylint: enable=invalid-name

  class SameNamedType1(SameNameab):
    pass

  # pylint: disable=g-error-prone-assert-raises
  def testAssertSameStructure(self):
    tree.assert_same_structure(STRUCTURE1, STRUCTURE2)
    tree.assert_same_structure("abc", 1.0)
    tree.assert_same_structure(b"abc", 1.0)
    tree.assert_same_structure(u"abc", 1.0)
    tree.assert_same_structure(bytearray("abc", "ascii"), 1.0)
    tree.assert_same_structure("abc", np.array([0, 1]))

  def testAssertSameStructure_differentNumElements(self):
    with self.assertRaisesRegex(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         "More specifically: Substructure "
         r'"type=tuple str=\(\(1, 2\), 3\)" is a sequence, while '
         'substructure "type=str str=spam" is not\n'
         "Entire first structure:\n"
         r"\(\(\(\., \.\), \.\), \., \(\., \.\)\)\n"
         "Entire second structure:\n"
         r"\(\., \.\)")):
      tree.assert_same_structure(STRUCTURE1, STRUCTURE_DIFFERENT_NUM_ELEMENTS)

  def testAssertSameStructure_listVsNdArray(self):
    with self.assertRaisesRegex(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         r'More specifically: Substructure "type=list str=\[0, 1\]" '
         r'is a sequence, while substructure "type=ndarray str=\[0 1\]" '
         "is not")):
      tree.assert_same_structure([0, 1], np.array([0, 1]))

  def testAssertSameStructure_intVsList(self):
    with self.assertRaisesRegex(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         r'More specifically: Substructure "type=list str=\[0, 1\]" '
         'is a sequence, while substructure "type=int str=0" '
         "is not")):
      tree.assert_same_structure(0, [0, 1])

  def testAssertSameStructure_tupleVsList(self):
    self.assertRaises(
        TypeError, tree.assert_same_structure, (0, 1), [0, 1])

  def testAssertSameStructure_differentNesting(self):
    with self.assertRaisesRegex(
        ValueError,
        ("don't have the same nested structure\\.\n\n"
         "First structure: .*?\n\nSecond structure: ")):
      tree.assert_same_structure(STRUCTURE1, STRUCTURE_DIFFERENT_NESTING)

  def testAssertSameStructure_tupleVsNamedTuple(self):
    self.assertRaises(TypeError, tree.assert_same_structure, (0, 1),
                      NestTest.Named0ab("a", "b"))

  def testAssertSameStructure_sameNamedTupleDifferentContents(self):
    tree.assert_same_structure(NestTest.Named0ab(3, 4),
                               NestTest.Named0ab("a", "b"))

  def testAssertSameStructure_differentNamedTuples(self):
    self.assertRaises(TypeError, tree.assert_same_structure,
                      NestTest.Named0ab(3, 4), NestTest.Named1ab(3, 4))

  def testAssertSameStructure_sameNamedTupleDifferentStructuredContents(self):
    with self.assertRaisesRegex(
        ValueError,
        ("don't have the same nested structure\\.\n\n"
         "First structure: .*?\n\nSecond structure: ")):
      tree.assert_same_structure(NestTest.Named0ab(3, 4),
                                 NestTest.Named0ab([3], 4))

  def testAssertSameStructure_differentlyNestedLists(self):
    with self.assertRaisesRegex(
        ValueError,
        ("don't have the same nested structure\\.\n\n"
         "First structure: .*?\n\nSecond structure: ")):
      tree.assert_same_structure([[3], 4], [3, [4]])

  def testAssertSameStructure_listStructureWithAndWithoutTypes(self):
    structure1_list = [[[1, 2], 3], 4, [5, 6]]
    with self.assertRaisesRegex(TypeError, "don't have the same sequence type"):
      tree.assert_same_structure(STRUCTURE1, structure1_list)
    tree.assert_same_structure(STRUCTURE1, STRUCTURE2, check_types=False)
    tree.assert_same_structure(STRUCTURE1, structure1_list, check_types=False)

  def testAssertSameStructure_dictionaryDifferentKeys(self):
    with self.assertRaisesRegex(ValueError, "don't have the same set of keys"):
      tree.assert_same_structure({"a": 1}, {"b": 1})

  def testAssertSameStructure_sameNameNamedTuples(self):
    tree.assert_same_structure(NestTest.SameNameab(0, 1),
                               NestTest.SameNameab2(2, 3))

  def testAssertSameStructure_sameNameNamedTuplesNested(self):
    # This assertion is expected to pass: two namedtuples with the same
    # name and field names are considered to be identical.
    tree.assert_same_structure(
        NestTest.SameNameab(NestTest.SameName1xy(0, 1), 2),
        NestTest.SameNameab2(NestTest.SameName1xy2(2, 3), 4))

  def testAssertSameStructure_sameNameNamedTuplesDifferentStructure(self):
    expected_message = "The two structures don't have the same.*"
    with self.assertRaisesRegex(ValueError, expected_message):
      tree.assert_same_structure(
          NestTest.SameNameab(0, NestTest.SameNameab2(1, 2)),
          NestTest.SameNameab2(NestTest.SameNameab(0, 1), 2))

  def testAssertSameStructure_differentNameNamedStructures(self):
    self.assertRaises(TypeError, tree.assert_same_structure,
                      NestTest.SameNameab(0, 1), NestTest.NotSameName(2, 3))

  def testAssertSameStructure_sameNameDifferentFieldNames(self):
    self.assertRaises(TypeError, tree.assert_same_structure,
                      NestTest.SameNameab(0, 1), NestTest.SameNamexy(2, 3))

  def testAssertSameStructure_classWrappingNamedTuple(self):
    self.assertRaises(TypeError, tree.assert_same_structure,
                      NestTest.SameNameab(0, 1), NestTest.SameNamedType1(2, 3))
  # pylint: enable=g-error-prone-assert-raises

  def testMapStructure(self):
    structure2 = (((7, 8), 9), 10, (11, 12))
    structure1_plus1 = tree.map_structure(lambda x: x + 1, STRUCTURE1)
    tree.assert_same_structure(STRUCTURE1, structure1_plus1)
    self.assertAllEquals(
        [2, 3, 4, 5, 6, 7],
        tree.flatten(structure1_plus1))
    structure1_plus_structure2 = tree.map_structure(
        lambda x, y: x + y, STRUCTURE1, structure2)
    self.assertEqual(
        (((1 + 7, 2 + 8), 3 + 9), 4 + 10, (5 + 11, 6 + 12)),
        structure1_plus_structure2)

    self.assertEqual(3, tree.map_structure(lambda x: x - 1, 4))

    self.assertEqual(7, tree.map_structure(lambda x, y: x + y, 3, 4))

    # Empty structures
    self.assertEqual((), tree.map_structure(lambda x: x + 1, ()))
    self.assertEqual([], tree.map_structure(lambda x: x + 1, []))
    self.assertEqual({}, tree.map_structure(lambda x: x + 1, {}))
    empty_nt = collections.namedtuple("empty_nt", "")
    self.assertEqual(empty_nt(), tree.map_structure(lambda x: x + 1,
                                                    empty_nt()))

    # This is checking actual equality of types, empty list != empty tuple
    self.assertNotEqual((), tree.map_structure(lambda x: x + 1, []))

    with self.assertRaisesRegex(TypeError, "callable"):
      tree.map_structure("bad", structure1_plus1)

    with self.assertRaisesRegex(ValueError, "at least one structure"):
      tree.map_structure(lambda x: x)

    with self.assertRaisesRegex(ValueError, "same number of elements"):
      tree.map_structure(lambda x, y: None, (3, 4), (3, 4, 5))

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      tree.map_structure(lambda x, y: None, 3, (3,))

    with self.assertRaisesRegex(TypeError, "same sequence type"):
      tree.map_structure(lambda x, y: None, ((3, 4), 5), [(3, 4), 5])

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      tree.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)))

    structure1_list = [[[1, 2], 3], 4, [5, 6]]
    with self.assertRaisesRegex(TypeError, "same sequence type"):
      tree.map_structure(lambda x, y: None, STRUCTURE1, structure1_list)

    tree.map_structure(lambda x, y: None, STRUCTURE1, structure1_list,
                       check_types=False)

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      tree.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)),
                         check_types=False)

    with self.assertRaisesRegex(ValueError, "Only valid keyword argument.*foo"):
      tree.map_structure(lambda x: None, STRUCTURE1, foo="a")

    with self.assertRaisesRegex(ValueError, "Only valid keyword argument.*foo"):
      tree.map_structure(lambda x: None, STRUCTURE1, check_types=False, foo="a")

  def testMapStructureWithStrings(self):
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    inp_a = ab_tuple(a="foo", b=("bar", "baz"))
    inp_b = ab_tuple(a=2, b=(1, 3))
    out = tree.map_structure(lambda string, repeats: string * repeats,
                             inp_a,
                             inp_b)
    self.assertEqual("foofoo", out.a)
    self.assertEqual("bar", out.b[0])
    self.assertEqual("bazbazbaz", out.b[1])

    nt = ab_tuple(a=("something", "something_else"),
                  b="yet another thing")
    rev_nt = tree.map_structure(lambda x: x[::-1], nt)
    # Check the output is the correct structure, and all strings are reversed.
    tree.assert_same_structure(nt, rev_nt)
    self.assertEqual(nt.a[0][::-1], rev_nt.a[0])
    self.assertEqual(nt.a[1][::-1], rev_nt.a[1])
    self.assertEqual(nt.b[::-1], rev_nt.b)

  def testAssertShallowStructure(self):
    inp_ab = ["a", "b"]
    inp_abc = ["a", "b", "c"]
    with self.assertRaisesRegex(
        ValueError,
        tree._STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
            input_length=len(inp_ab),
            shallow_length=len(inp_abc))):
      tree._assert_shallow_structure(inp_abc, inp_ab)

    inp_ab1 = [(1, 1), (2, 2)]
    inp_ab2 = [[1, 1], [2, 2]]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        tree._STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(inp_ab2[0]),
            input_type=type(inp_ab1[0]))):
      tree._assert_shallow_structure(shallow_tree=inp_ab2, input_tree=inp_ab1)

    tree._assert_shallow_structure(inp_ab2, inp_ab1, check_types=False)

    inp_ab1 = {"a": (1, 1), "b": {"c": (2, 2)}}
    inp_ab2 = {"a": (1, 1), "b": {"d": (2, 2)}}

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        tree._SHALLOW_TREE_HAS_INVALID_KEYS.format(["d"])):
      tree._assert_shallow_structure(inp_ab2, inp_ab1)

    inp_ab = collections.OrderedDict([("a", 1), ("b", (2, 3))])
    inp_ba = collections.OrderedDict([("b", (2, 3)), ("a", 1)])
    tree._assert_shallow_structure(inp_ab, inp_ba)

    # regression test for b/130633904
    tree._assert_shallow_structure({0: "foo"}, ["foo"], check_types=False)

  def testFlattenUpTo(self):
    # Shallow tree ends at scalar.
    input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
    shallow_tree = [[True, True], [False, True]]
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [[2, 2], [3, 3], [4, 9], [5, 5]])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])

    # Shallow tree ends at string.
    input_tree = [[("a", 1), [("b", 2), [("c", 3), [("d", 4)]]]]]
    shallow_tree = [["level_1", ["level_2", ["level_3", ["level_4"]]]]]
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree,
                                                              input_tree)
    input_tree_flattened = tree.flatten(input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    self.assertEqual(input_tree_flattened, ["a", 1, "b", 2, "c", 3, "d", 4])

    # Make sure dicts are correctly flattened, yielding values, not keys.
    input_tree = {"a": 1, "b": {"c": 2}, "d": [3, (4, 5)]}
    shallow_tree = {"a": 0, "b": 0, "d": [0, 0]}
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [1, {"c": 2}, 3, (4, 5)])

    # Namedtuples.
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    input_tree = ab_tuple(a=[0, 1], b=2)
    shallow_tree = ab_tuple(a=0, b=1)
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [[0, 1], 2])

    # Attrs.
    @attr.s
    class ABAttr(object):
      a = attr.ib()
      b = attr.ib()
    input_tree = ABAttr(a=[0, 1], b=2)
    shallow_tree = ABAttr(a=0, b=1)
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [[0, 1], 2])

    # Nested dicts, OrderedDicts and namedtuples.
    input_tree = collections.OrderedDict(
        [("a", ab_tuple(a=[0, {"b": 1}], b=2)),
         ("c", {"d": 3, "e": collections.OrderedDict([("f", 4)])})])
    shallow_tree = input_tree
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree, [0, 1, 2, 3, 4])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", {"d": 3, "e": 1})])
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      3,
                      collections.OrderedDict([("f", 4)])])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", 0)])
    input_tree_flattened_as_shallow_tree = tree.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      {"d": 3, "e": collections.OrderedDict([("f", 4)])}])

    ## Shallow non-list edge-case.
    # Using iterable elements.
    input_tree = ["input_tree"]
    shallow_tree = "shallow_tree"
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = ["input_tree_0", "input_tree_1"]
    shallow_tree = "shallow_tree"
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = [0]
    shallow_tree = 9
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = [0, 1]
    shallow_tree = 9
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Both non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = "shallow_tree"
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = 0
    flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Input non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = ["shallow_tree"]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree))):
      flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = "input_tree"
    shallow_tree = ["shallow_tree_9", "shallow_tree_8"]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree))):
      flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = [9]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree))):
      flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = 0
    shallow_tree = [9, 8]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree))):
      flattened_input_tree = tree.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = tree.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

  def testByteStringsNotTreatedAsIterable(self):
    structure = [u"unicode string", b"byte string"]
    flattened_structure = tree.flatten_up_to(structure, structure)
    self.assertEqual(structure, flattened_structure)

  def testFlattenWithPathUpTo(self):

    def get_paths_and_values(shallow_tree, input_tree):
      path_value_pairs = tree.flatten_with_path_up_to(shallow_tree, input_tree)
      paths = [p for p, _ in path_value_pairs]
      values = [v for _, v in path_value_pairs]
      return paths, values

    # Shallow tree ends at scalar.
    input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
    shallow_tree = [[True, True], [False, True]]
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths,
                     [(0, 0), (0, 1), (1, 0), (1, 1)])
    self.assertEqual(flattened_input_tree, [[2, 2], [3, 3], [4, 9], [5, 5]])
    self.assertEqual(flattened_shallow_tree_paths,
                     [(0, 0), (0, 1), (1, 0), (1, 1)])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])

    # Shallow tree ends at string.
    input_tree = [[("a", 1), [("b", 2), [("c", 3), [("d", 4)]]]]]
    shallow_tree = [["level_1", ["level_2", ["level_3", ["level_4"]]]]]
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    input_tree_flattened_paths = [
        p for p, _ in tree.flatten_with_path(input_tree)
    ]
    input_tree_flattened = tree.flatten(input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [(0, 0), (0, 1, 0), (0, 1, 1, 0), (0, 1, 1, 1, 0)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [("a", 1), ("b", 2), ("c", 3), ("d", 4)])

    self.assertEqual(input_tree_flattened_paths,
                     [(0, 0, 0), (0, 0, 1),
                      (0, 1, 0, 0), (0, 1, 0, 1),
                      (0, 1, 1, 0, 0), (0, 1, 1, 0, 1),
                      (0, 1, 1, 1, 0, 0), (0, 1, 1, 1, 0, 1)])
    self.assertEqual(input_tree_flattened, ["a", 1, "b", 2, "c", 3, "d", 4])

    # Make sure dicts are correctly flattened, yielding values, not keys.
    input_tree = {"a": 1, "b": {"c": 2}, "d": [3, (4, 5)]}
    shallow_tree = {"a": 0, "b": 0, "d": [0, 0]}
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",), ("b",), ("d", 0), ("d", 1)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [1, {"c": 2}, 3, (4, 5)])

    # Namedtuples.
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    input_tree = ab_tuple(a=[0, 1], b=2)
    shallow_tree = ab_tuple(a=0, b=1)
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",), ("b",)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [[0, 1], 2])

    # Nested dicts, OrderedDicts and namedtuples.
    input_tree = collections.OrderedDict(
        [("a", ab_tuple(a=[0, {"b": 1}], b=2)),
         ("c", {"d": 3, "e": collections.OrderedDict([("f", 4)])})])
    shallow_tree = input_tree
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a", "a", 0),
                      ("a", "a", 1, "b"),
                      ("a", "b"),
                      ("c", "d"),
                      ("c", "e", "f")])
    self.assertEqual(input_tree_flattened_as_shallow_tree, [0, 1, 2, 3, 4])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", {"d": 3, "e": 1})])
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",),
                      ("c", "d"),
                      ("c", "e")])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      3,
                      collections.OrderedDict([("f", 4)])])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", 0)])
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",), ("c",)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      {"d": 3, "e": collections.OrderedDict([("f", 4)])}])

    ## Shallow non-list edge-case.
    # Using iterable elements.
    input_tree = ["input_tree"]
    shallow_tree = "shallow_tree"
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = ["input_tree_0", "input_tree_1"]
    shallow_tree = "shallow_tree"
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Test case where len(shallow_tree) < len(input_tree)
    input_tree = {"a": "A", "b": "B", "c": "C"}
    shallow_tree = {"a": 1, "c": 2}

    # Using non-iterable elements.
    input_tree = [0]
    shallow_tree = 9
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = [0, 1]
    shallow_tree = 9
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Both non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = "shallow_tree"
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = 0
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Input non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = ["shallow_tree"]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ_WITH_PATH.format(
            path=[], input_type=type(input_tree))):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,)])
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = "input_tree"
    shallow_tree = ["shallow_tree_9", "shallow_tree_8"]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ_WITH_PATH.format(
            path=[], input_type=type(input_tree))):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,), (1,)])
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = [9]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ_WITH_PATH.format(
            path=[], input_type=type(input_tree))):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,)])
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = 0
    shallow_tree = [9, 8]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ_WITH_PATH.format(
            path=[], input_type=type(input_tree))):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,), (1,)])
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    # Test that error messages include paths.
    input_tree = {"a": {"b": {0, 1}}}
    structure = {"a": {"b": [0, 1]}}
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        tree._IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ_WITH_PATH.format(
            path=["a", "b"], input_type=type(input_tree["a"]["b"]))):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(structure, input_tree)
    (flattened_tree_paths,
     flattened_tree) = get_paths_and_values(structure, structure)
    self.assertEqual(flattened_tree_paths, [("a", "b", 0,), ("a", "b", 1,)])
    self.assertEqual(flattened_tree, structure["a"]["b"])

  def testMapStructureUpTo(self):
    # Named tuples.
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    op_tuple = collections.namedtuple("op_tuple", "add, mul")
    inp_val = ab_tuple(a=2, b=3)
    inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
    out = tree.map_structure_up_to(
        inp_val,
        lambda val, ops: (val + ops.add) * ops.mul,
        inp_val,
        inp_ops,
        check_types=False)
    self.assertEqual(out.a, 6)
    self.assertEqual(out.b, 15)

    # Lists.
    data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
    name_list = ["evens", ["odds", "primes"]]
    out = tree.map_structure_up_to(
        name_list, lambda name, sec: "first_{}_{}".format(len(sec), name),
        name_list, data_list)
    self.assertEqual(out, ["first_4_evens", ["first_5_odds", "first_3_primes"]])

  # We cannot define namedtuples within @parameterized argument lists.
  # pylint: disable=invalid-name
  Foo = collections.namedtuple("Foo", ["a", "b"])
  Bar = collections.namedtuple("Bar", ["c", "d"])
  # pylint: enable=invalid-name

  @parameterized.parameters([
      dict(inputs=[], expected=[]),
      dict(inputs=[23, "42"], expected=[((0,), 23), ((1,), "42")]),
      dict(inputs=[[[[108]]]], expected=[((0, 0, 0, 0), 108)]),
      dict(inputs=Foo(a=3, b=Bar(c=23, d=42)),
           expected=[(("a",), 3), (("b", "c"), 23), (("b", "d"), 42)]),
      dict(inputs=Foo(a=Bar(c=23, d=42), b=Bar(c=0, d="thing")),
           expected=[(("a", "c"), 23), (("a", "d"), 42), (("b", "c"), 0),
                     (("b", "d"), "thing")]),
      dict(inputs=Bar(c=42, d=43),
           expected=[(("c",), 42), (("d",), 43)]),
      dict(inputs=Bar(c=[42], d=43),
           expected=[(("c", 0), 42), (("d",), 43)]),
      dict(inputs=wrapt.ObjectProxy(Bar(c=[42], d=43)),
           expected=[(("c", 0), 42), (("d",), 43)]),
  ])
  def testFlattenWithPath(self, inputs, expected):
    self.assertEqual(tree.flatten_with_path(inputs), expected)

  @parameterized.named_parameters([
      dict(testcase_name="Tuples", s1=(1, 2), s2=(3, 4),
           check_types=True, expected=(((0,), 4), ((1,), 6))),
      dict(testcase_name="Dicts", s1={"a": 1, "b": 2}, s2={"b": 4, "a": 3},
           check_types=True, expected={"a": (("a",), 4), "b": (("b",), 6)}),
      dict(testcase_name="Mixed", s1=(1, 2), s2=[3, 4],
           check_types=False, expected=(((0,), 4), ((1,), 6))),
      dict(testcase_name="Nested",
           s1={"a": [2, 3], "b": [1, 2, 3]},
           s2={"b": [5, 6, 7], "a": [8, 9]},
           check_types=True,
           expected={"a": [(("a", 0), 10), (("a", 1), 12)],
                     "b": [(("b", 0), 6), (("b", 1), 8), (("b", 2), 10)]}),
  ])
  def testMapWithPathCompatibleStructures(self, s1, s2, check_types, expected):
    def path_and_sum(path, *values):
      return path, sum(values)

    result = tree.map_structure_with_path(
        path_and_sum, s1, s2, check_types=check_types)
    self.assertEqual(expected, result)

  @parameterized.named_parameters([
      dict(testcase_name="Tuples", s1=(1, 2, 3), s2=(4, 5),
           error_type=ValueError),
      dict(testcase_name="Dicts", s1={"a": 1}, s2={"b": 2},
           error_type=ValueError),
      dict(testcase_name="Nested",
           s1={"a": [2, 3, 4], "b": [1, 3]},
           s2={"b": [5, 6], "a": [8, 9]},
           error_type=ValueError)
  ])
  def testMapWithPathIncompatibleStructures(self, s1, s2, error_type):
    with self.assertRaises(error_type):
      tree.map_structure_with_path(lambda path, *s: 0, s1, s2)

  def testMappingProxyType(self):
    structure = types.MappingProxyType({"a": 1, "b": (2, 3)})
    expected = types.MappingProxyType({"a": 4, "b": (5, 6)})
    self.assertEqual(tree.flatten(structure), [1, 2, 3])
    self.assertEqual(tree.unflatten_as(structure, [4, 5, 6]), expected)
    self.assertEqual(tree.map_structure(lambda v: v + 3, structure), expected)

  def testTraverseListsToTuples(self):
    structure = [(1, 2), [3], {"a": [4]}]
    self.assertEqual(
        ((1, 2), (3,), {"a": (4,)}),
        tree.traverse(
            lambda x: tuple(x) if isinstance(x, list) else x,
            structure,
            top_down=False))

  def testTraverseEarlyTermination(self):
    structure = [(1, [2]), [3, (4, 5, 6)]]
    visited = []
    def visit(x):
      visited.append(x)
      return "X" if isinstance(x, tuple) and len(x) > 2 else None

    output = tree.traverse(visit, structure)
    self.assertEqual([(1, [2]), [3, "X"]], output)
    self.assertEqual(
        [[(1, [2]), [3, (4, 5, 6)]],
         (1, [2]), 1, [2], 2, [3, (4, 5, 6)], 3, (4, 5, 6)],
        visited)

  def testMapStructureAcrossSubtreesDict(self):
    shallow = {"a": 1, "b": {"c": 2}}
    deep1 = {"a": 2, "b": {"c": 3, "d": 2}, "e": 4}
    deep2 = {"a": 3, "b": {"c": 2, "d": 3}, "e": 1}
    summed = tree.map_structure_up_to(
        shallow, lambda *args: sum(args), deep1, deep2)
    expected = {"a": 5, "b": {"c": 5}}
    self.assertEqual(summed, expected)
    concatenated = tree.map_structure_up_to(
        shallow, lambda *args: args, deep1, deep2)
    expected = {"a": (2, 3), "b": {"c": (3, 2)}}
    self.assertEqual(concatenated, expected)

  def testMapStructureAcrossSubtreesNoneValues(self):
    shallow = [1, [None]]
    deep1 = [1, [2, 3]]
    deep2 = [2, [3, 4]]
    summed = tree.map_structure_up_to(
        shallow, lambda *args: sum(args), deep1, deep2)
    expected = [3, [5]]
    self.assertEqual(summed, expected)

  def testMapStructureAcrossSubtreesList(self):
    shallow = [1, [1]]
    deep1 = [1, [2, 3]]
    deep2 = [2, [3, 4]]
    summed = tree.map_structure_up_to(
        shallow, lambda *args: sum(args), deep1, deep2)
    expected = [3, [5]]
    self.assertEqual(summed, expected)

  def testMapStructureAcrossSubtreesTuple(self):
    shallow = (1, (1,))
    deep1 = (1, (2, 3))
    deep2 = (2, (3, 4))
    summed = tree.map_structure_up_to(
        shallow, lambda *args: sum(args), deep1, deep2)
    expected = (3, (5,))
    self.assertEqual(summed, expected)

  def testMapStructureAcrossSubtreesNamedTuple(self):
    Foo = collections.namedtuple("Foo", ["x", "y"])
    Bar = collections.namedtuple("Bar", ["x"])
    shallow = Bar(1)
    deep1 = Foo(1, (1, 0))
    deep2 = Foo(2, (2, 0))
    summed = tree.map_structure_up_to(
        shallow, lambda *args: sum(args), deep1, deep2)
    expected = Bar(3)
    self.assertEqual(summed, expected)

  def testMapStructureAcrossSubtreesListTuple(self):
    # Tuples and lists can be used interchangeably between shallow structure
    # and input structures. Output takes on type of the shallow structure
    shallow = [1, (1,)]
    deep1 = [1, [2, 3]]
    deep2 = [2, [3, 4]]
    summed = tree.map_structure_up_to(shallow, lambda *args: sum(args), deep1,
                                      deep2)
    expected = [3, (5,)]
    self.assertEqual(summed, expected)

    shallow = [1, [1]]
    deep1 = [1, (2, 3)]
    deep2 = [2, (3, 4)]
    summed = tree.map_structure_up_to(shallow, lambda *args: sum(args), deep1,
                                      deep2)
    expected = [3, [5]]
    self.assertEqual(summed, expected)

  def testNoneNodeIncluded(self):
    structure = ((1, None))
    self.assertEqual(tree.flatten(structure), [1, None])

  def testCustomClassMapWithPath(self):

    class ExampleClass(Mapping[Any, Any]):
      """Small example custom class."""

      def __init__(self, *args, **kwargs):
        self._mapping = dict(*args, **kwargs)

      def __getitem__(self, k: Any) -> Any:
        return self._mapping[k]

      def __len__(self) -> int:
        return len(self._mapping)

      def __iter__(self) -> Iterator[Any]:
        return iter(self._mapping)

    def mapper(path, value):
      full_path = "/".join(path)
      return f"{full_path}_{value}"

    test_input = ExampleClass({"first": 1, "nested": {"second": 2, "third": 3}})
    output = tree.map_structure_with_path(mapper, test_input)
    expected = ExampleClass({
        "first": "first_1",
        "nested": {
            "second": "nested/second_2",
            "third": "nested/third_3"
        }
    })
    self.assertEqual(output, expected)


if __name__ == "__main__":
  unittest.main()
