# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Results of coverage measurement."""

from __future__ import annotations

import collections
import dataclasses

from collections.abc import Container, Iterable
from typing import TYPE_CHECKING

from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo

if TYPE_CHECKING:
    from coverage.data import CoverageData
    from coverage.plugin import FileReporter


def analysis_from_file_reporter(
    data: CoverageData,
    precision: int,
    file_reporter: FileReporter,
    filename: str,
) -> Analysis:
    """Create an Analysis from a FileReporter."""
    has_arcs = data.has_arcs()
    statements = file_reporter.lines()
    excluded = file_reporter.excluded_lines()
    executed = file_reporter.translate_lines(data.lines(filename) or [])

    if has_arcs:
        arc_possibilities_set = file_reporter.arcs()
        arcs: Iterable[TArc] = data.arcs(filename) or []
        arcs = file_reporter.translate_arcs(arcs)

        # Reduce the set of arcs to the ones that could be branches.
        dests = collections.defaultdict(set)
        for fromno, tono in arc_possibilities_set:
            dests[fromno].add(tono)
        single_dests = {
            fromno: list(tonos)[0]
            for fromno, tonos in dests.items()
            if len(tonos) == 1
        }
        new_arcs = set()
        for fromno, tono in arcs:
            if fromno != tono:
                new_arcs.add((fromno, tono))
            else:
                if fromno in single_dests:
                    new_arcs.add((fromno, single_dests[fromno]))

        arcs_executed_set = file_reporter.translate_arcs(new_arcs)
        exit_counts = file_reporter.exit_counts()
        no_branch = file_reporter.no_branch_lines()
    else:
        arc_possibilities_set = set()
        arcs_executed_set = set()
        exit_counts = {}
        no_branch = set()

    return Analysis(
        precision=precision,
        filename=filename,
        has_arcs=has_arcs,
        statements=statements,
        excluded=excluded,
        executed=executed,
        arc_possibilities_set=arc_possibilities_set,
        arcs_executed_set=arcs_executed_set,
        exit_counts=exit_counts,
        no_branch=no_branch,
    )


@dataclasses.dataclass
class Analysis:
    """The results of analyzing a FileReporter."""

    precision: int
    filename: str
    has_arcs: bool
    statements: set[TLineNo]
    excluded: set[TLineNo]
    executed: set[TLineNo]
    arc_possibilities_set: set[TArc]
    arcs_executed_set: set[TArc]
    exit_counts: dict[TLineNo, int]
    no_branch: set[TLineNo]

    def __post_init__(self) -> None:
        self.arc_possibilities = sorted(self.arc_possibilities_set)
        self.arcs_executed = sorted(self.arcs_executed_set)
        self.missing = self.statements - self.executed

        if self.has_arcs:
            n_branches = self._total_branches()
            mba = self.missing_branch_arcs()
            n_partial_branches = sum(len(v) for k,v in mba.items() if k not in self.missing)
            n_missing_branches = sum(len(v) for k,v in mba.items())
        else:
            n_branches = n_partial_branches = n_missing_branches = 0

        self.numbers = Numbers(
            precision=self.precision,
            n_files=1,
            n_statements=len(self.statements),
            n_excluded=len(self.excluded),
            n_missing=len(self.missing),
            n_branches=n_branches,
            n_partial_branches=n_partial_branches,
            n_missing_branches=n_missing_branches,
        )

    def narrow(self, lines: Container[TLineNo]) -> Analysis:
        """Create a narrowed Analysis.

        The current analysis is copied to make a new one that only considers
        the lines in `lines`.
        """

        statements = {lno for lno in self.statements if lno in lines}
        excluded = {lno for lno in self.excluded if lno in lines}
        executed = {lno for lno in self.executed if lno in lines}

        if self.has_arcs:
            arc_possibilities_set = {
                (a, b) for a, b in self.arc_possibilities_set
                if a in lines or b in lines
            }
            arcs_executed_set = {
                (a, b) for a, b in self.arcs_executed_set
                if a in lines or b in lines
            }
            exit_counts = {
                lno: num for lno, num in self.exit_counts.items()
                if lno in lines
            }
            no_branch = {lno for lno in self.no_branch if lno in lines}
        else:
            arc_possibilities_set = set()
            arcs_executed_set = set()
            exit_counts = {}
            no_branch = set()

        return Analysis(
            precision=self.precision,
            filename=self.filename,
            has_arcs=self.has_arcs,
            statements=statements,
            excluded=excluded,
            executed=executed,
            arc_possibilities_set=arc_possibilities_set,
            arcs_executed_set=arcs_executed_set,
            exit_counts=exit_counts,
            no_branch=no_branch,
        )

    def missing_formatted(self, branches: bool = False) -> str:
        """The missing line numbers, formatted nicely.

        Returns a string like "1-2, 5-11, 13-14".

        If `branches` is true, includes the missing branch arcs also.

        """
        if branches and self.has_arcs:
            arcs = self.missing_branch_arcs().items()
        else:
            arcs = None

        return format_lines(self.statements, self.missing, arcs=arcs)

    def arcs_missing(self) -> list[TArc]:
        """Returns a sorted list of the un-executed arcs in the code."""
        missing = (
            p for p in self.arc_possibilities
                if p not in self.arcs_executed_set
                    and p[0] not in self.no_branch
                    and p[1] not in self.excluded
        )
        return sorted(missing)

    def _branch_lines(self) -> list[TLineNo]:
        """Returns a list of line numbers that have more than one exit."""
        return [l1 for l1,count in self.exit_counts.items() if count > 1]

    def _total_branches(self) -> int:
        """How many total branches are there?"""
        return sum(count for count in self.exit_counts.values() if count > 1)

    def missing_branch_arcs(self) -> dict[TLineNo, list[TLineNo]]:
        """Return arcs that weren't executed from branch lines.

        Returns {l1:[l2a,l2b,...], ...}

        """
        missing = self.arcs_missing()
        branch_lines = set(self._branch_lines())
        mba = collections.defaultdict(list)
        for l1, l2 in missing:
            assert l1 != l2, f"In {self.filename}, didn't expect {l1} == {l2}"
            if l1 in branch_lines:
                mba[l1].append(l2)
        return mba

    def executed_branch_arcs(self) -> dict[TLineNo, list[TLineNo]]:
        """Return arcs that were executed from branch lines.

        Only include ones that we considered possible.

        Returns {l1:[l2a,l2b,...], ...}

        """
        branch_lines = set(self._branch_lines())
        eba = collections.defaultdict(list)
        for l1, l2 in self.arcs_executed:
            assert l1 != l2, f"Oops: Didn't think this could happen: {l1 = }, {l2 = }"
            if (l1, l2) not in self.arc_possibilities_set:
                continue
            if l1 in branch_lines:
                eba[l1].append(l2)
        return eba

    def branch_stats(self) -> dict[TLineNo, tuple[int, int]]:
        """Get stats about branches.

        Returns a dict mapping line numbers to a tuple:
        (total_exits, taken_exits).

        """

        missing_arcs = self.missing_branch_arcs()
        stats = {}
        for lnum in self._branch_lines():
            exits = self.exit_counts[lnum]
            missing = len(missing_arcs[lnum])
            stats[lnum] = (exits, exits - missing)
        return stats


@dataclasses.dataclass
class Numbers:
    """The numerical results of measuring coverage.

    This holds the basic statistics from `Analysis`, and is used to roll
    up statistics across files.

    """

    precision: int = 0
    n_files: int = 0
    n_statements: int = 0
    n_excluded: int = 0
    n_missing: int = 0
    n_branches: int = 0
    n_partial_branches: int = 0
    n_missing_branches: int = 0

    @property
    def n_executed(self) -> int:
        """Returns the number of executed statements."""
        return self.n_statements - self.n_missing

    @property
    def n_executed_branches(self) -> int:
        """Returns the number of executed branches."""
        return self.n_branches - self.n_missing_branches

    @property
    def pc_covered(self) -> float:
        """Returns a single percentage value for coverage."""
        if self.n_statements > 0:
            numerator, denominator = self.ratio_covered
            pc_cov = (100.0 * numerator) / denominator
        else:
            pc_cov = 100.0
        return pc_cov

    @property
    def pc_covered_str(self) -> str:
        """Returns the percent covered, as a string, without a percent sign.

        Note that "0" is only returned when the value is truly zero, and "100"
        is only returned when the value is truly 100.  Rounding can never
        result in either "0" or "100".

        """
        return display_covered(self.pc_covered, self.precision)

    @property
    def ratio_covered(self) -> tuple[int, int]:
        """Return a numerator and denominator for the coverage ratio."""
        numerator = self.n_executed + self.n_executed_branches
        denominator = self.n_statements + self.n_branches
        return numerator, denominator

    def __add__(self, other: Numbers) -> Numbers:
        return Numbers(
            self.precision,
            self.n_files + other.n_files,
            self.n_statements + other.n_statements,
            self.n_excluded + other.n_excluded,
            self.n_missing + other.n_missing,
            self.n_branches + other.n_branches,
            self.n_partial_branches + other.n_partial_branches,
            self.n_missing_branches + other.n_missing_branches,
        )

    def __radd__(self, other: int) -> Numbers:
        # Implementing 0+Numbers allows us to sum() a list of Numbers.
        assert other == 0   # we only ever call it this way.
        return self


def display_covered(pc: float, precision: int) -> str:
    """Return a displayable total percentage, as a string.

    Note that "0" is only returned when the value is truly zero, and "100"
    is only returned when the value is truly 100.  Rounding can never
    result in either "0" or "100".

    """
    near0 = 1.0 / 10 ** precision
    if 0 < pc < near0:
        pc = near0
    elif (100.0 - near0) < pc < 100:
        pc = 100.0 - near0
    else:
        pc = round(pc, precision)
    return "%.*f" % (precision, pc)


def _line_ranges(
    statements: Iterable[TLineNo],
    lines: Iterable[TLineNo],
) -> list[tuple[TLineNo, TLineNo]]:
    """Produce a list of ranges for `format_lines`."""
    statements = sorted(statements)
    lines = sorted(lines)

    pairs = []
    start: TLineNo | None = None
    lidx = 0
    for stmt in statements:
        if lidx >= len(lines):
            break
        if stmt == lines[lidx]:
            lidx += 1
            if not start:
                start = stmt
            end = stmt
        elif start:
            pairs.append((start, end))
            start = None
    if start:
        pairs.append((start, end))
    return pairs


def format_lines(
    statements: Iterable[TLineNo],
    lines: Iterable[TLineNo],
    arcs: Iterable[tuple[TLineNo, list[TLineNo]]] | None = None,
) -> str:
    """Nicely format a list of line numbers.

    Format a list of line numbers for printing by coalescing groups of lines as
    long as the lines represent consecutive statements.  This will coalesce
    even if there are gaps between statements.

    For example, if `statements` is [1,2,3,4,5,10,11,12,13,14] and
    `lines` is [1,2,5,10,11,13,14] then the result will be "1-2, 5-11, 13-14".

    Both `lines` and `statements` can be any iterable. All of the elements of
    `lines` must be in `statements`, and all of the values must be positive
    integers.

    If `arcs` is provided, they are (start,[end,end,end]) pairs that will be
    included in the output as long as start isn't in `lines`.

    """
    line_items = [(pair[0], nice_pair(pair)) for pair in _line_ranges(statements, lines)]
    if arcs is not None:
        line_exits = sorted(arcs)
        for line, exits in line_exits:
            for ex in sorted(exits):
                if line not in lines and ex not in lines:
                    dest = (ex if ex > 0 else "exit")
                    line_items.append((line, f"{line}->{dest}"))

    ret = ", ".join(t[-1] for t in sorted(line_items))
    return ret


def should_fail_under(total: float, fail_under: float, precision: int) -> bool:
    """Determine if a total should fail due to fail-under.

    `total` is a float, the coverage measurement total. `fail_under` is the
    fail_under setting to compare with. `precision` is the number of digits
    to consider after the decimal point.

    Returns True if the total should fail.

    """
    # We can never achieve higher than 100% coverage, or less than zero.
    if not (0 <= fail_under <= 100.0):
        msg = f"fail_under={fail_under} is invalid. Must be between 0 and 100."
        raise ConfigError(msg)

    # Special case for fail_under=100, it must really be 100.
    if fail_under == 100.0 and total != 100.0:
        return True

    return round(total, precision) < fail_under
