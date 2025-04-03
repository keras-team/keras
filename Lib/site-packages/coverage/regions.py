# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Find functions and classes in Python code."""

from __future__ import annotations

import ast
import dataclasses

from typing import cast

from coverage.plugin import CodeRegion


@dataclasses.dataclass
class Context:
    """The nested named context of a function or class."""
    name: str
    kind: str
    lines: set[int]


class RegionFinder:
    """An ast visitor that will find and track regions of code.

    Functions and classes are tracked by name. Results are in the .regions
    attribute.

    """
    def __init__(self) -> None:
        self.regions: list[CodeRegion] = []
        self.context: list[Context] = []

    def parse_source(self, source: str) -> None:
        """Parse `source` and walk the ast to populate the .regions attribute."""
        self.handle_node(ast.parse(source))

    def fq_node_name(self) -> str:
        """Get the current fully qualified name we're processing."""
        return ".".join(c.name for c in self.context)

    def handle_node(self, node: ast.AST) -> None:
        """Recursively handle any node."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.handle_FunctionDef(node)
        elif isinstance(node, ast.ClassDef):
            self.handle_ClassDef(node)
        else:
            self.handle_node_body(node)

    def handle_node_body(self, node: ast.AST) -> None:
        """Recursively handle the nodes in this node's body, if any."""
        for body_node in getattr(node, "body", ()):
            self.handle_node(body_node)

    def handle_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Called for `def` or `async def`."""
        lines = set(range(node.body[0].lineno, cast(int, node.body[-1].end_lineno) + 1))
        if self.context and self.context[-1].kind == "class":
            # Function bodies are part of their enclosing class.
            self.context[-1].lines |= lines
        # Function bodies should be excluded from the nearest enclosing function.
        for ancestor in reversed(self.context):
            if ancestor.kind == "function":
                ancestor.lines -= lines
                break
        self.context.append(Context(node.name, "function", lines))
        self.regions.append(
            CodeRegion(
                kind="function",
                name=self.fq_node_name(),
                start=node.lineno,
                lines=lines,
            )
        )
        self.handle_node_body(node)
        self.context.pop()

    def handle_ClassDef(self, node: ast.ClassDef) -> None:
        """Called for `class`."""
        # The lines for a class are the lines in the methods of the class.
        # We start empty, and count on visit_FunctionDef to add the lines it
        # finds.
        lines: set[int] = set()
        self.context.append(Context(node.name, "class", lines))
        self.regions.append(
            CodeRegion(
                kind="class",
                name=self.fq_node_name(),
                start=node.lineno,
                lines=lines,
            )
        )
        self.handle_node_body(node)
        self.context.pop()
        # Class bodies should be excluded from the enclosing classes.
        for ancestor in reversed(self.context):
            if ancestor.kind == "class":
                ancestor.lines -= lines


def code_regions(source: str) -> list[CodeRegion]:
    """Find function and class regions in source code.

    Analyzes the code in `source`, and returns a list of :class:`CodeRegion`
    objects describing functions and classes as regions of the code::

        [
            CodeRegion(kind="function", name="func1", start=8, lines={10, 11, 12}),
            CodeRegion(kind="function", name="MyClass.method", start=30, lines={34, 35, 36}),
            CodeRegion(kind="class", name="MyClass", start=25, lines={34, 35, 36}),
        ]

    The line numbers will include comments and blank lines.  Later processing
    will need to ignore those lines as needed.

    Nested functions and classes are excluded from their enclosing region.  No
    line should be reported as being part of more than one function, or more
    than one class.  Lines in methods are reported as being in a function and
    in a class.

    """
    rf = RegionFinder()
    rf.parse_source(source)
    return rf.regions
