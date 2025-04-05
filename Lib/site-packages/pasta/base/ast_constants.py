"""Constants relevant to ast code."""

import ast

NODE_TYPE_TO_TOKENS = {
    ast.Add: ('+',),
    ast.And: ('and',),
    ast.BitAnd: ('&',),
    ast.BitOr: ('|',),
    ast.BitXor: ('^',),
    ast.Div: ('/',),
    ast.Eq: ('==',),
    ast.FloorDiv: ('//',),
    ast.Gt: ('>',),
    ast.GtE: ('>=',),
    ast.In: ('in',),
    ast.Invert: ('~',),
    ast.Is: ('is',),
    ast.IsNot: ('is', 'not',),
    ast.LShift: ('<<',),
    ast.Lt: ('<',),
    ast.LtE: ('<=',),
    ast.Mod: ('%',),
    ast.Mult: ('*',),
    ast.Not: ('not',),
    ast.NotEq: ('!=',),
    ast.NotIn: ('not', 'in',),
    ast.Or: ('or',),
    ast.Pow: ('**',),
    ast.RShift: ('>>',),
    ast.Sub: ('-',),
    ast.UAdd: ('+',),
    ast.USub: ('-',),
}


if hasattr(ast, 'MatMult'):
  NODE_TYPE_TO_TOKENS[ast.MatMult] = ('@',)
