# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Bytecode analysis for coverage.py"""

from __future__ import annotations

import dis

from types import CodeType
from typing import Iterable, Optional
from collections.abc import Iterator

from coverage.types import TArc, TOffset


def code_objects(code: CodeType) -> Iterator[CodeType]:
    """Iterate over all the code objects in `code`."""
    stack = [code]
    while stack:
        # We're going to return the code object on the stack, but first
        # push its children for later returning.
        code = stack.pop()
        for c in code.co_consts:
            if isinstance(c, CodeType):
                stack.append(c)
        yield code


def op_set(*op_names: str) -> set[int]:
    """Make a set of opcodes from instruction names.

    The names might not exist in this version of Python, skip those if not.
    """
    return {op for name in op_names if (op := dis.opmap.get(name))}


# Opcodes that are unconditional jumps elsewhere.
ALWAYS_JUMPS = op_set(
    "JUMP_BACKWARD",
    "JUMP_BACKWARD_NO_INTERRUPT",
    "JUMP_FORWARD",
)

# Opcodes that exit from a function.
RETURNS = op_set("RETURN_VALUE", "RETURN_GENERATOR")


class InstructionWalker:
    """Utility to step through trails of instructions.

    We have two reasons to need sequences of instructions from a code object:
    First, in strict sequence to visit all the instructions in the object.
    This is `walk(follow_jumps=False)`.  Second, we want to follow jumps to
    understand how execution will flow: `walk(follow_jumps=True)`.

    """

    def __init__(self, code: CodeType) -> None:
        self.code = code
        self.insts: dict[TOffset, dis.Instruction] = {}

        inst = None
        for inst in dis.get_instructions(code):
            self.insts[inst.offset] = inst

        assert inst is not None
        self.max_offset = inst.offset

    def walk(
        self, *, start_at: TOffset = 0, follow_jumps: bool = True
    ) -> Iterable[dis.Instruction]:
        """
        Yield instructions starting from `start_at`.  Follow unconditional
        jumps if `follow_jumps` is true.
        """
        seen = set()
        offset = start_at
        while offset < self.max_offset + 1:
            if offset in seen:
                break
            seen.add(offset)
            if inst := self.insts.get(offset):
                yield inst
                if follow_jumps and inst.opcode in ALWAYS_JUMPS:
                    offset = inst.jump_target
                    continue
            offset += 2


TBranchTrail = tuple[set[TOffset], Optional[TArc]]
TBranchTrails = dict[TOffset, list[TBranchTrail]]


def branch_trails(code: CodeType) -> TBranchTrails:
    """
    Calculate branch trails for `code`.

    Instructions can have a jump_target, where they might jump to next.  Some
    instructions with a jump_target are unconditional jumps (ALWAYS_JUMPS), so
    they aren't interesting to us, since they aren't the start of a branch
    possibility.

    Instructions that might or might not jump somewhere else are branch
    possibilities.  For each of those, we track a trail of instructions.  These
    are lists of instruction offsets, the next instructions that can execute.
    We follow the trail until we get to a new source line.  That gives us the
    arc from the original instruction's line to the new source line.

    """
    the_trails: TBranchTrails = {}
    iwalker = InstructionWalker(code)
    for inst in iwalker.walk(follow_jumps=False):
        if not inst.jump_target:
            # We only care about instructions with jump targets.
            continue
        if inst.opcode in ALWAYS_JUMPS:
            # We don't care about unconditional jumps.
            continue

        from_line = inst.line_number
        if from_line is None:
            continue

        def walk_one_branch(start_at: TOffset) -> TBranchTrail:
            # pylint: disable=cell-var-from-loop
            inst_offsets: set[TOffset] = set()
            to_line = None
            for inst2 in iwalker.walk(start_at=start_at):
                inst_offsets.add(inst2.offset)
                if inst2.line_number and inst2.line_number != from_line:
                    to_line = inst2.line_number
                    break
                elif inst2.jump_target and (inst2.opcode not in ALWAYS_JUMPS):
                    break
                elif inst2.opcode in RETURNS:
                    to_line = -code.co_firstlineno
                    break
            if to_line is not None:
                return inst_offsets, (from_line, to_line)
            else:
                return set(), None

        # Calculate two trails: one from the next instruction, and one from the
        # jump_target instruction.
        trails = [
            walk_one_branch(start_at=inst.offset + 2),
            walk_one_branch(start_at=inst.jump_target),
        ]
        the_trails[inst.offset] = trails

        # Sometimes we get BRANCH_RIGHT or BRANCH_LEFT events from instructions
        # other than the original jump possibility instruction.  Register each
        # trail under all of their offsets so we can pick up in the middle of a
        # trail if need be.
        for trail in trails:
            for offset in trail[0]:
                if offset not in the_trails:
                    the_trails[offset] = []
                the_trails[offset].append(trail)

    return the_trails
