# coding=utf-8
"""Token generator for analyzing source code in logical units.

This module contains the TokenGenerator used for annotating a parsed syntax tree
with source code formatting.
"""
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
import collections
import contextlib
import itertools
import tokenize
from six import StringIO

from pasta.base import formatting as fmt
from pasta.base import fstring_utils

# Alias for extracting token names
TOKENS = tokenize
Token = collections.namedtuple('Token', ('type', 'src', 'start', 'end', 'line'))
FORMATTING_TOKENS = (TOKENS.INDENT, TOKENS.DEDENT, TOKENS.NL, TOKENS.NEWLINE,
                     TOKENS.COMMENT)


class TokenGenerator(object):
  """Helper for sequentially parsing Python source code, token by token.

  Holds internal state during parsing, including:
  _tokens: List of tokens in the source code, as parsed by `tokenize` module.
  _parens: Stack of open parenthesis at the current point in parsing.
  _hints: Number of open parentheses, brackets, etc. at the current point.
  _scope_stack: Stack containing tuples of nodes where the last parenthesis that
    was open is related to one of the nodes on the top of the stack.
  _lines: Full lines of the source code.
  _i: Index of the last token that was parsed. Initially -1.
  _loc: (lineno, column_offset) pair of the position in the source that has been
     parsed to. This should be either the start or end of the token at index _i.

  Arguments:
    ignore_error_tokens: If True, will ignore error tokens. Otherwise, an error
      token will cause an exception. This is useful when the source being parsed
      contains invalid syntax, e.g. if it is in an fstring context.
  """

  def __init__(self, source, ignore_error_token=False):
    self.lines = source.splitlines(True)
    self._tokens = list(_generate_tokens(source, ignore_error_token))
    self._parens = []
    self._hints = 0
    self._scope_stack = []
    self._len = len(self._tokens)
    self._i = -1
    self._loc = self.loc_begin()

  def chars_consumed(self):
    return len(self._space_between((1, 0), self._tokens[self._i].end))

  def loc_begin(self):
    """Get the start column of the current location parsed to."""
    if self._i < 0:
      return (1, 0)
    return self._tokens[self._i].start

  def loc_end(self):
    """Get the end column of the current location parsed to."""
    if self._i < 0:
      return (1, 0)
    return self._tokens[self._i].end

  def peek(self):
    """Get the next token without advancing."""
    if self._i + 1 >= self._len:
      return None
    return self._tokens[self._i + 1]

  def peek_non_whitespace(self):
    """Get the next non-whitespace token without advancing."""
    return self.peek_conditional(lambda t: t.type not in FORMATTING_TOKENS)

  def peek_conditional(self, condition):
    """Get the next token of the given type without advancing."""
    return next((t for t in self._tokens[self._i + 1:] if condition(t)), None)

  def next(self, advance=True):
    """Consume the next token and optionally advance the current location."""
    self._i += 1
    if self._i >= self._len:
      return None
    if advance:
      self._loc = self._tokens[self._i].end
    return self._tokens[self._i]

  def rewind(self, amount=1):
    """Rewind the token iterator."""
    self._i -= amount

  def whitespace(self, max_lines=None, comment=False):
    """Parses whitespace from the current _loc to the next non-whitespace.

    Arguments:
      max_lines: (optional int) Maximum number of lines to consider as part of
        the whitespace. Valid values are None, 0 and 1.
      comment: (boolean) If True, look for a trailing comment even when not in
        a parenthesized scope.

    Pre-condition:
      `_loc' represents the point before which everything has been parsed and
      after which nothing has been parsed.
    Post-condition:
      `_loc' is exactly at the character that was parsed to.
    """
    next_token = self.peek()
    if not comment and next_token and next_token.type == TOKENS.COMMENT:
      return ''
    def predicate(token):
      return (token.type in (TOKENS.INDENT, TOKENS.DEDENT) or
              token.type == TOKENS.COMMENT and (comment or self._hints) or
              token.type == TOKENS.ERRORTOKEN and token.src == ' ' or
              max_lines is None and token.type in (TOKENS.NL, TOKENS.NEWLINE))
    whitespace = list(self.takewhile(predicate, advance=False))
    next_token = self.peek()

    result = ''
    for tok in itertools.chain(whitespace,
                               ((next_token,) if next_token else ())):
      result += self._space_between(self._loc, tok.start)
      if tok != next_token:
        result += tok.src
        self._loc = tok.end
      else:
        self._loc = tok.start

    # Eat a single newline character
    if ((max_lines is None or max_lines > 0) and
        next_token and next_token.type in (TOKENS.NL, TOKENS.NEWLINE)):
      result += self.next().src

    return result

  def block_whitespace(self, indent_level):
    """Parses whitespace from the current _loc to the end of the block."""
    # Get the normal suffix lines, but don't advance the token index unless
    # there is no indentation to account for
    start_i = self._i
    full_whitespace = self.whitespace(comment=True)
    if not indent_level:
      return full_whitespace
    self._i = start_i

    # Trim the full whitespace into only lines that match the indentation level
    lines = full_whitespace.splitlines(True)
    try:
      last_line_idx = next(i for i, line in reversed(list(enumerate(lines)))
                           if line.startswith(indent_level + '#'))
    except StopIteration:
      # No comment lines at the end of this block
      self._loc = self._tokens[self._i].end
      return ''
    lines = lines[:last_line_idx + 1]

    # Advance the current location to the last token in the lines we've read
    end_line = self._tokens[self._i].end[0] + 1 + len(lines)
    list(self.takewhile(lambda tok: tok.start[0] < end_line))
    self._loc = self._tokens[self._i].end
    return ''.join(lines)

  def dots(self, num_dots):
    """Parse a number of dots.
    
    This is to work around an oddity in python3's tokenizer, which treats three
    `.` tokens next to each other in a FromImport's level as an ellipsis. This
    parses until the expected number of dots have been seen.
    """
    result = ''
    dots_seen = 0
    prev_loc = self._loc
    while dots_seen < num_dots:
      tok = self.next()
      assert tok.src in ('.', '...')
      result += self._space_between(prev_loc, tok.start) + tok.src
      dots_seen += tok.src.count('.')
      prev_loc = self._loc
    return result

  def open_scope(self, node, single_paren=False):
    """Open a parenthesized scope on the given node."""
    result = ''
    parens = []
    start_i = self._i
    start_loc = prev_loc = self._loc

    # Eat whitespace or '(' tokens one at a time
    for tok in self.takewhile(
        lambda t: t.type in FORMATTING_TOKENS or t.src == '('):
      # Stores all the code up to and including this token
      result += self._space_between(prev_loc, tok.start)

      if tok.src == '(' and single_paren and parens:
        self.rewind()
        self._loc = tok.start
        break

      result += tok.src
      if tok.src == '(':
        # Start a new scope
        parens.append(result)
        result = ''
        start_i = self._i
        start_loc = self._loc
      prev_loc = self._loc

    if parens:
      # Add any additional whitespace on to the last open-paren
      next_tok = self.peek()
      parens[-1] += result + self._space_between(self._loc, next_tok.start)
      self._loc = next_tok.start
      # Add each paren onto the stack
      for paren in parens:
        self._parens.append(paren)
        self._scope_stack.append(_scope_helper(node))
    else:
      # No parens were encountered, then reset like this method did nothing
      self._i = start_i
      self._loc = start_loc

  def close_scope(self, node, prefix_attr='prefix', suffix_attr='suffix',
                  trailing_comma=False, single_paren=False):
    """Close a parenthesized scope on the given node, if one is open."""
    # Ensures the prefix + suffix are not None
    if fmt.get(node, prefix_attr) is None:
      fmt.set(node, prefix_attr, '')
    if fmt.get(node, suffix_attr) is None:
      fmt.set(node, suffix_attr, '')

    if not self._parens or node not in self._scope_stack[-1]:
      return
    symbols = {')'}
    if trailing_comma:
      symbols.add(',')
    parsed_to_i = self._i
    parsed_to_loc = prev_loc = self._loc
    encountered_paren = False
    result = ''

    for tok in self.takewhile(
        lambda t: t.type in FORMATTING_TOKENS or t.src in symbols):
      # Consume all space up to this token
      result += self._space_between(prev_loc, tok.start)
      if tok.src == ')' and single_paren and encountered_paren:
        self.rewind()
        parsed_to_i = self._i
        parsed_to_loc = tok.start
        fmt.append(node, suffix_attr, result)
        break

      # Consume the token itself
      result += tok.src

      if tok.src == ')':
        # Close out the open scope
        encountered_paren = True
        self._scope_stack.pop()
        fmt.prepend(node, prefix_attr, self._parens.pop())
        fmt.append(node, suffix_attr, result)
        result = ''
        parsed_to_i = self._i
        parsed_to_loc = tok.end
        if not self._parens or node not in self._scope_stack[-1]:
          break
      prev_loc = tok.end

    # Reset back to the last place where we parsed anything
    self._i = parsed_to_i
    self._loc = parsed_to_loc

  def hint_open(self):
    """Indicates opening a group of parentheses or brackets."""
    self._hints += 1

  def hint_closed(self):
    """Indicates closing a group of parentheses or brackets."""
    self._hints -= 1
    if self._hints < 0:
      raise ValueError('Hint value negative')

  @contextlib.contextmanager
  def scope(self, node, attr=None, trailing_comma=False):
    """Context manager to handle a parenthesized scope."""
    self.open_scope(node, single_paren=(attr is not None))
    yield
    if attr:
      self.close_scope(node, prefix_attr=attr + '_prefix',
                       suffix_attr=attr + '_suffix',
                       trailing_comma=trailing_comma,
                       single_paren=True)
    else:
      self.close_scope(node, trailing_comma=trailing_comma)

  def is_in_scope(self):
    """Return True iff there is a scope open."""
    return self._parens or self._hints

  def str(self):
    """Parse a full string literal from the input."""
    def predicate(token):
      return (token.type in (TOKENS.STRING, TOKENS.COMMENT) or
              self.is_in_scope() and token.type in (TOKENS.NL, TOKENS.NEWLINE))

    return self.eat_tokens(predicate)

  def eat_tokens(self, predicate):
    """Parse input from tokens while a given condition is met."""
    content = ''
    prev_loc = self._loc
    tok = None
    for tok in self.takewhile(predicate, advance=False):
      content += self._space_between(prev_loc, tok.start)
      content += tok.src
      prev_loc = tok.end

    if tok:
      self._loc = tok.end
    return content

  def fstr(self):
    """Parses an fstring, including subexpressions.

    Returns:
      A generator function which, when repeatedly reads a chunk of the fstring
      up until the next subexpression and yields that chunk, plus a new token
      generator to use to parse the subexpression. The subexpressions in the
      original fstring data are replaced by placeholders to make it possible to
      fill them in with new values, if desired.
    """
    def fstr_parser():
      # Reads the whole fstring as a string, then parses it char by char
      if self.peek_non_whitespace().type == TOKENS.STRING:
        # Normal fstrings are one ore more STRING tokens, maybe mixed with
        # spaces, e.g.: f"Hello, {name}"
        str_content = self.str()
      else:
        # Format specifiers in fstrings are also JoinedStr nodes, but these are
        # arbitrary expressions, e.g. in: f"{value:{width}.{precision}}", the
        # format specifier is an fstring: "{width}.{precision}" but these are
        # not STRING tokens.
        def fstr_eater(tok):
          if tok.type == TOKENS.OP and tok.src == '}':
            if fstr_eater.level <= 0:
              return False
            fstr_eater.level -= 1
          if tok.type == TOKENS.OP and tok.src == '{':
            fstr_eater.level += 1
          return True
        fstr_eater.level = 0
        str_content = self.eat_tokens(fstr_eater)

      indexed_chars = enumerate(str_content)
      val_idx = 0
      i = -1
      result = ''
      while i < len(str_content) - 1:
        i, c = next(indexed_chars)
        result += c

        # When an open bracket is encountered, start parsing a subexpression
        if c == '{':
          # First check if this is part of an escape sequence
          # (f"{{" is used to escape a bracket literal)
          nexti, nextc = next(indexed_chars)
          if nextc == '{':
            result += c
            continue
          indexed_chars = itertools.chain([(nexti, nextc)], indexed_chars)

          # Add a placeholder onto the result
          result += fstring_utils.placeholder(val_idx) + '}'
          val_idx += 1

          # Yield a new token generator to parse the subexpression only
          tg = TokenGenerator(str_content[i+1:], ignore_error_token=True)
          yield (result, tg)
          result = ''

          # Skip the number of characters consumed by the subexpression
          for tg_i in range(tg.chars_consumed()):
            i, c = next(indexed_chars)

          # Eat up to and including the close bracket
          i, c = next(indexed_chars)
          while c != '}':
            i, c = next(indexed_chars)
      # Yield the rest of the fstring, when done
      yield (result, None)
    return fstr_parser

  def _space_between(self, start_loc, end_loc):
    """Parse the space between a location and the next token"""
    if start_loc > end_loc:
      raise ValueError('start_loc > end_loc', start_loc, end_loc)
    if start_loc[0] > len(self.lines):
      return ''

    prev_row, prev_col = start_loc
    end_row, end_col = end_loc
    if prev_row == end_row:
      return self.lines[prev_row - 1][prev_col:end_col]

    return ''.join(itertools.chain(
        (self.lines[prev_row - 1][prev_col:],),
        self.lines[prev_row:end_row - 1],
        (self.lines[end_row - 1][:end_col],) if end_col > 0 else '',
    ))

  def next_name(self):
    """Parse the next name token."""
    last_i = self._i
    def predicate(token):
      return token.type != TOKENS.NAME

    unused_tokens = list(self.takewhile(predicate, advance=False))
    result = self.next(advance=False)
    self._i = last_i
    return result

  def next_of_type(self, token_type):
    """Parse a token of the given type and return it."""
    token = self.next()
    if token.type != token_type:
      raise ValueError("Expected %r but found %r\nline %d: %s" % (
          tokenize.tok_name[token_type], token.src, token.start[0],
          self.lines[token.start[0] - 1]))
    return token

  def takewhile(self, condition, advance=True):
    """Parse tokens as long as a condition holds on the next token."""
    prev_loc = self._loc
    token = self.next(advance=advance)
    while token is not None and condition(token):
      yield token
      prev_loc = self._loc
      token = self.next(advance=advance)
    self.rewind()
    self._loc = prev_loc


def _scope_helper(node):
  """Get the closure of nodes that could begin a scope at this point.

  For instance, when encountering a `(` when parsing a BinOp node, this could
  indicate that the BinOp itself is parenthesized OR that the BinOp's left node
  could be parenthesized.

  E.g.: (a + b * c)   or   (a + b) * c   or   (a) + b * c
        ^                  ^                  ^

  Arguments:
    node: (ast.AST) Node encountered when opening a scope.

  Returns:
    A closure of nodes which that scope might apply to.
  """
  if isinstance(node, ast.Attribute):
    return (node,) + _scope_helper(node.value)
  if isinstance(node, ast.Subscript):
    return (node,) + _scope_helper(node.value)
  if isinstance(node, ast.Assign):
    return (node,) + _scope_helper(node.targets[0])
  if isinstance(node, ast.AugAssign):
    return (node,) + _scope_helper(node.target)
  if isinstance(node, ast.Expr):
    return (node,) + _scope_helper(node.value)
  if isinstance(node, ast.Compare):
    return (node,) + _scope_helper(node.left)
  if isinstance(node, ast.BoolOp):
    return (node,) + _scope_helper(node.values[0])
  if isinstance(node, ast.BinOp):
    return (node,) + _scope_helper(node.left)
  if isinstance(node, ast.Tuple) and node.elts:
    return (node,) + _scope_helper(node.elts[0])
  if isinstance(node, ast.Call):
    return (node,) + _scope_helper(node.func)
  if isinstance(node, ast.GeneratorExp):
    return (node,) + _scope_helper(node.elt)
  if isinstance(node, ast.IfExp):
    return (node,) + _scope_helper(node.body)
  return (node,)
   

def _generate_tokens(source, ignore_error_token=False):
  token_generator = tokenize.generate_tokens(StringIO(source).readline)
  try:
    for tok in token_generator:
      yield Token(*tok) 
  except tokenize.TokenError:
    if not ignore_error_token:
      raise
