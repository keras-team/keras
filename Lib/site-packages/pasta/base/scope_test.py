# coding=utf-8
"""Tests for scope."""
# Copyright 2017 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import textwrap
import unittest

from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils


class ScopeTest(test_utils.TestCase):

  def test_top_level_imports(self):
    self.maxDiff = None
    source = textwrap.dedent("""\
        import aaa
        import bbb, ccc.ddd
        import aaa.bbb.ccc
        from eee import fff
        from ggg.hhh import iii, jjj
        """)
    tree = ast.parse(source)
    nodes = tree.body

    node_1_aaa = nodes[0].names[0]
    node_2_bbb = nodes[1].names[0]
    node_2_ccc_ddd = nodes[1].names[1]
    node_3_aaa_bbb_ccc = nodes[2].names[0]
    node_4_eee = nodes[3]
    node_4_fff = nodes[3].names[0]
    node_5_ggg_hhh = nodes[4]
    node_5_iii = nodes[4].names[0]
    node_5_jjj = nodes[4].names[1]

    s = scope.analyze(tree)

    self.assertItemsEqual(
        s.names.keys(), {
            'aaa', 'bbb', 'ccc', 'fff', 'iii', 'jjj'
        })
    self.assertItemsEqual(
        s.external_references.keys(), {
            'aaa', 'bbb', 'ccc', 'ccc.ddd', 'aaa.bbb', 'aaa.bbb.ccc', 'eee',
            'eee.fff', 'ggg', 'ggg.hhh', 'ggg.hhh.iii', 'ggg.hhh.jjj'
        })
    self.assertItemsEqual(s.external_references['aaa'], [
        scope.ExternalReference('aaa', node_1_aaa, s.names['aaa']),
        scope.ExternalReference('aaa', node_3_aaa_bbb_ccc, s.names['aaa']),
    ])
    self.assertItemsEqual(s.external_references['bbb'], [
        scope.ExternalReference('bbb', node_2_bbb, s.names['bbb']),
    ])
    self.assertItemsEqual(s.external_references['ccc'], [
        scope.ExternalReference('ccc', node_2_ccc_ddd, s.names['ccc']),
    ])
    self.assertItemsEqual(s.external_references['ccc.ddd'], [
        scope.ExternalReference('ccc.ddd', node_2_ccc_ddd,
                                s.names['ccc'].attrs['ddd']),
    ])
    self.assertItemsEqual(s.external_references['aaa.bbb'], [
        scope.ExternalReference('aaa.bbb', node_3_aaa_bbb_ccc,
                                s.names['aaa'].attrs['bbb']),
    ])
    self.assertItemsEqual(s.external_references['aaa.bbb.ccc'], [
        scope.ExternalReference('aaa.bbb.ccc', node_3_aaa_bbb_ccc,
                                s.names['aaa'].attrs['bbb'].attrs['ccc']),
    ])
    self.assertItemsEqual(s.external_references['eee'], [
        scope.ExternalReference('eee', node_4_eee, None),
    ])
    self.assertItemsEqual(s.external_references['eee.fff'], [
        scope.ExternalReference('eee.fff', node_4_fff, s.names['fff']),
    ])
    self.assertItemsEqual(s.external_references['ggg'], [
        scope.ExternalReference('ggg', node_5_ggg_hhh, None),
    ])
    self.assertItemsEqual(s.external_references['ggg.hhh'], [
        scope.ExternalReference('ggg.hhh', node_5_ggg_hhh, None),
    ])
    self.assertItemsEqual(s.external_references['ggg.hhh.iii'], [
        scope.ExternalReference('ggg.hhh.iii', node_5_iii, s.names['iii']),
    ])
    self.assertItemsEqual(s.external_references['ggg.hhh.jjj'], [
        scope.ExternalReference('ggg.hhh.jjj', node_5_jjj, s.names['jjj']),
    ])
    
    self.assertIs(s.names['aaa'].definition, node_1_aaa)
    self.assertIs(s.names['bbb'].definition, node_2_bbb)
    self.assertIs(s.names['ccc'].definition, node_2_ccc_ddd)
    self.assertIs(s.names['fff'].definition, node_4_fff)
    self.assertIs(s.names['iii'].definition, node_5_iii)
    self.assertIs(s.names['jjj'].definition, node_5_jjj)

    self.assertItemsEqual(s.names['aaa'].reads, [node_3_aaa_bbb_ccc])
    for ref in {'bbb', 'ccc', 'fff', 'iii', 'jjj'}:
      self.assertEqual(s.names[ref].reads, [], 'Expected no reads for %s' % ref)

  def test_if_nested_imports(self):
    source = textwrap.dedent("""\
        if a:
          import aaa
        elif b:
          import bbb
        else:
          import ccc
        """)
    tree = ast.parse(source)
    nodes = tree.body

    node_aaa, node_bbb, node_ccc = ast_utils.find_nodes_by_type(tree, ast.alias)

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa', 'bbb', 'ccc', 'a', 'b'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa', 'bbb', 'ccc'})
    
    self.assertEqual(s.names['aaa'].definition, node_aaa)
    self.assertEqual(s.names['bbb'].definition, node_bbb)
    self.assertEqual(s.names['ccc'].definition, node_ccc)

    self.assertIsNone(s.names['a'].definition)
    self.assertIsNone(s.names['b'].definition)

    for ref in {'aaa', 'bbb', 'ccc'}:
      self.assertEqual(s.names[ref].reads, [],
                       'Expected no reads for %s' % ref)

  def test_try_nested_imports(self):
    source = textwrap.dedent("""\
        try:
          import aaa
        except:
          import bbb
        finally:
          import ccc
        """)
    tree = ast.parse(source)
    nodes = tree.body

    node_aaa, node_bbb, node_ccc = ast_utils.find_nodes_by_type(tree, ast.alias)

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa', 'bbb', 'ccc'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa', 'bbb', 'ccc'})
    
    self.assertEqual(s.names['aaa'].definition, node_aaa)
    self.assertEqual(s.names['bbb'].definition, node_bbb)
    self.assertEqual(s.names['ccc'].definition, node_ccc)

    for ref in {'aaa', 'bbb', 'ccc'}:
      self.assertEqual(s.names[ref].reads, [],
                       'Expected no reads for %s' % ref)

  def test_functiondef_nested_imports(self):
    source = textwrap.dedent("""\
        def foo(bar):
          import aaa
        """)
    tree = ast.parse(source)
    nodes = tree.body

    node_aaa = ast_utils.find_nodes_by_type(tree, ast.alias)[0]

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})

  def test_classdef_nested_imports(self):
    source = textwrap.dedent("""\
        class Foo():
          import aaa
        """)
    tree = ast.parse(source)
    nodes = tree.body

    node_aaa = nodes[0].body[0].names[0]

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'Foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})

  def test_multilevel_import_reads(self):
    source = textwrap.dedent("""\
        import aaa.bbb.ccc
        aaa.bbb.ccc.foo()
        """)
    tree = ast.parse(source)
    nodes = tree.body

    node_ref = nodes[1].value.func.value

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa'})
    self.assertItemsEqual(s.external_references.keys(),
                          {'aaa', 'aaa.bbb', 'aaa.bbb.ccc'})
    self.assertItemsEqual(s.names['aaa'].reads, [node_ref.value.value])
    self.assertItemsEqual(s.names['aaa'].attrs['bbb'].reads, [node_ref.value])
    self.assertItemsEqual(s.names['aaa'].attrs['bbb'].attrs['ccc'].reads,
                          [node_ref])

  def test_import_reads_in_functiondef(self):
    source = textwrap.dedent("""\
        import aaa
        @aaa.x
        def foo(bar):
          return aaa
        """)
    tree = ast.parse(source)
    nodes = tree.body

    return_value = nodes[1].body[0].value
    decorator = nodes[1].decorator_list[0].value

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa', 'foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads, [decorator, return_value])

  def test_import_reads_in_classdef(self):
    source = textwrap.dedent("""\
        import aaa
        @aaa.x
        class Foo(aaa.Bar):
          pass
        """)
    tree = ast.parse(source)
    nodes = tree.body

    node_aaa = nodes[0].names[0]
    decorator = nodes[1].decorator_list[0].value
    base = nodes[1].bases[0].value

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa', 'Foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads, [decorator, base])

  def test_import_masked_by_function_arg(self):
    source = textwrap.dedent("""\
        import aaa
        def foo(aaa=aaa):
          return aaa
        """)
    tree = ast.parse(source)
    nodes = tree.body

    argval = nodes[1].args.defaults[0]

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa', 'foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads, [argval])

  def test_import_masked_by_assign(self):
    source = textwrap.dedent("""\
        import aaa
        def foo():
          aaa = 123
          return aaa
        aaa
        """)
    tree = ast.parse(source)
    nodes = tree.body

    node_aaa = nodes[2].value

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa', 'foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads, [node_aaa])

  def test_import_in_decortator(self):
    source = textwrap.dedent("""\
        import aaa
        @aaa.wrapper
        def foo(aaa=1):
          pass
        """)
    tree = ast.parse(source)
    nodes = tree.body

    decorator = nodes[1].decorator_list[0].value

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa', 'foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads, [decorator])

  @test_utils.requires_features('type_annotations')
  def test_import_in_return_type(self):
    source = textwrap.dedent("""\
        import aaa
        def foo() -> aaa.Foo:
          pass
        """)
    tree = ast.parse(source)
    nodes = tree.body

    func = nodes[1]

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa', 'foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads, [func.returns.value])

  @test_utils.requires_features('type_annotations')
  def test_import_in_argument_type(self):
    source = textwrap.dedent("""\
        import aaa
        def foo(bar: aaa.Bar):
          pass
        """)
    tree = ast.parse(source)
    nodes = tree.body

    func = nodes[1]

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa', 'foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads,
                          [func.args.args[0].annotation.value])

  def test_import_attribute_references(self):
    source = textwrap.dedent("""\
        import aaa.bbb.ccc, ddd.eee
        aaa.x()
        aaa.bbb.y()
        aaa.bbb.ccc.z()
        """)
    tree = ast.parse(source)
    nodes = tree.body

    call1 = nodes[1].value.func.value
    call2 = nodes[2].value.func.value
    call3 = nodes[3].value.func.value

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa', 'ddd'})
    self.assertItemsEqual(s.external_references.keys(),
                          {'aaa', 'aaa.bbb', 'aaa.bbb.ccc', 'ddd', 'ddd.eee'})
    self.assertItemsEqual(s.names['aaa'].reads,
                          [call1, call2.value, call3.value.value])
    self.assertItemsEqual(s.names['aaa'].attrs['bbb'].reads,
                          [call2, call3.value])
    self.assertItemsEqual(s.names['aaa'].attrs['bbb'].attrs['ccc'].reads,
                          [call3])

  def test_lookup_scope(self):
    src = textwrap.dedent("""\
        import a
        def b(c, d, e=1):
          class F(d):
            g = 1
          return c
        """)
    t = ast.parse(src)
    import_node, func_node = t.body
    class_node, return_node = func_node.body

    sc = scope.analyze(t)
    import_node_scope = sc.lookup_scope(import_node)
    self.assertIs(import_node_scope.node, t)
    self.assertIs(import_node_scope, sc)
    self.assertItemsEqual(import_node_scope.names, ['a', 'b'])

    func_node_scope = sc.lookup_scope(func_node)
    self.assertIs(func_node_scope.node, func_node)
    self.assertIs(func_node_scope.parent_scope, sc)
    self.assertItemsEqual(func_node_scope.names, ['c', 'd', 'e', 'F'])

    class_node_scope = sc.lookup_scope(class_node)
    self.assertIs(class_node_scope.node, class_node)
    self.assertIs(class_node_scope.parent_scope, func_node_scope)
    self.assertItemsEqual(class_node_scope.names, ['g'])

    return_node_scope = sc.lookup_scope(return_node)
    self.assertIs(return_node_scope.node, func_node)
    self.assertIs(return_node_scope, func_node_scope)
    self.assertItemsEqual(return_node_scope.names, ['c', 'd', 'e', 'F'])

    self.assertIs(class_node_scope.lookup_scope(func_node),
                  func_node_scope)

    self.assertIsNone(sc.lookup_scope(ast.Name(id='foo')))

  def test_class_methods(self):
    source = textwrap.dedent("""\
        import aaa
        class C:
          def aaa(self):
            return aaa

          def bbb(self):
            return aaa
        """)
    tree = ast.parse(source)
    importstmt, classdef = tree.body
    method_aaa, method_bbb = classdef.body

    s = scope.analyze(tree)

    self.assertItemsEqual(s.names.keys(), {'aaa', 'C'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads,
                          [method_aaa.body[0].value, method_bbb.body[0].value])
    # TODO: Test references to C.aaa, C.bbb once supported

  def test_vararg_kwarg_references_in_function_body(self):
    source = textwrap.dedent("""\
        def aaa(bbb, *ccc, **ddd):
          ccc
          ddd
        eee(ccc, ddd)
        """)
    tree = ast.parse(source)
    funcdef, call = tree.body
    ccc_expr, ddd_expr = funcdef.body

    sc = scope.analyze(tree)

    func_scope = sc.lookup_scope(funcdef)
    self.assertIn('ccc', func_scope.names)
    self.assertItemsEqual(func_scope.names['ccc'].reads, [ccc_expr.value])
    self.assertIn('ddd', func_scope.names)
    self.assertItemsEqual(func_scope.names['ddd'].reads, [ddd_expr.value])


def suite():
  result = unittest.TestSuite()
  result.addTests(unittest.makeSuite(ScopeTest))
  return result


if __name__ == '__main__':
  unittest.main()
