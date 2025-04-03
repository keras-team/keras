# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""LCOV reporting for coverage.py."""

from __future__ import annotations

import base64
import hashlib
import sys

from typing import IO, TYPE_CHECKING
from collections.abc import Iterable

from coverage.plugin import FileReporter
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.types import TMorf

if TYPE_CHECKING:
    from coverage import Coverage


def line_hash(line: str) -> str:
    """Produce a hash of a source line for use in the LCOV file."""
    # The LCOV file format optionally allows each line to be MD5ed as a
    # fingerprint of the file.  This is not a security use.  Some security
    # scanners raise alarms about the use of MD5 here, but it is a false
    # positive.  This is not a security concern.
    # The unusual encoding of the MD5 hash, as a base64 sequence with the
    # trailing = signs stripped, is specified by the LCOV file format.
    hashed = hashlib.md5(line.encode("utf-8"), usedforsecurity=False).digest()
    return base64.b64encode(hashed).decode("ascii").rstrip("=")


def lcov_lines(
    analysis: Analysis,
    lines: list[int],
    source_lines: list[str],
    outfile: IO[str],
) -> None:
    """Emit line coverage records for an analyzed file."""
    hash_suffix = ""
    for line in lines:
        if source_lines:
            hash_suffix = "," + line_hash(source_lines[line-1])
        # Q: can we get info about the number of times a statement is
        # executed?  If so, that should be recorded here.
        hit = int(line not in analysis.missing)
        outfile.write(f"DA:{line},{hit}{hash_suffix}\n")

    if analysis.numbers.n_statements > 0:
        outfile.write(f"LF:{analysis.numbers.n_statements}\n")
        outfile.write(f"LH:{analysis.numbers.n_executed}\n")


def lcov_functions(
    fr: FileReporter,
    file_analysis: Analysis,
    outfile: IO[str],
) -> None:
    """Emit function coverage records for an analyzed file."""
    # lcov 2.2 introduces a new format for function coverage records.
    # We continue to generate the old format because we don't know what
    # version of the lcov tools will be used to read this report.

    # "and region.lines" below avoids a crash due to a bug in PyPy 3.8
    # where, for whatever reason, when collecting data in --branch mode,
    # top-level functions have an empty lines array.  Instead we just don't
    # emit function records for those.

    # suppressions because of https://github.com/pylint-dev/pylint/issues/9923
    functions = [
        (min(region.start, min(region.lines)), #pylint: disable=nested-min-max
         max(region.start, max(region.lines)), #pylint: disable=nested-min-max
         region)
        for region in fr.code_regions()
        if region.kind == "function" and region.lines
    ]
    if not functions:
        return

    functions.sort()
    functions_hit = 0
    for first_line, last_line, region in functions:
        # A function counts as having been executed if any of it has been
        # executed.
        analysis = file_analysis.narrow(region.lines)
        hit = int(analysis.numbers.n_executed > 0)
        functions_hit += hit

        outfile.write(f"FN:{first_line},{last_line},{region.name}\n")
        outfile.write(f"FNDA:{hit},{region.name}\n")

    outfile.write(f"FNF:{len(functions)}\n")
    outfile.write(f"FNH:{functions_hit}\n")


def lcov_arcs(
    fr: FileReporter,
    analysis: Analysis,
    lines: list[int],
    outfile: IO[str],
) -> None:
    """Emit branch coverage records for an analyzed file."""
    branch_stats = analysis.branch_stats()
    executed_arcs = analysis.executed_branch_arcs()
    missing_arcs = analysis.missing_branch_arcs()

    for line in lines:
        if line not in branch_stats:
            continue

        # This is only one of several possible ways to map our sets of executed
        # and not-executed arcs to BRDA codes.  It seems to produce reasonable
        # results when fed through genhtml.
        _, taken = branch_stats[line]

        if taken == 0:
            # When _none_ of the out arcs from 'line' were executed,
            # it can mean the line always raised an exception.
            assert len(executed_arcs[line]) == 0
            destinations = [
                (dst, "-") for dst in missing_arcs[line]
            ]
        else:
            # Q: can we get counts of the number of times each arc was executed?
            # branch_stats has "total" and "taken" counts for each branch,
            # but it doesn't have "taken" broken down by destination.
            destinations = [
                (dst, "1") for dst in executed_arcs[line]
            ]
            destinations.extend(
                (dst, "0") for dst in missing_arcs[line]
            )

        # Sort exit arcs after normal arcs.  Exit arcs typically come from
        # an if statement, at the end of a function, with no else clause.
        # This structure reads like you're jumping to the end of the function
        # when the conditional expression is false, so it should be presented
        # as the second alternative for the branch, after the alternative that
        # enters the if clause.
        destinations.sort(key=lambda d: (d[0] < 0, d))

        for dst, hit in destinations:
            branch = fr.arc_description(line, dst)
            outfile.write(f"BRDA:{line},0,{branch},{hit}\n")

    # Summary of the branch coverage.
    brf = sum(t for t, k in branch_stats.values())
    brh = brf - sum(t - k for t, k in branch_stats.values())
    if brf > 0:
        outfile.write(f"BRF:{brf}\n")
        outfile.write(f"BRH:{brh}\n")


class LcovReporter:
    """A reporter for writing LCOV coverage reports."""

    report_type = "LCOV report"

    def __init__(self, coverage: Coverage) -> None:
        self.coverage = coverage
        self.config = coverage.config
        self.total = Numbers(self.coverage.config.precision)

    def report(self, morfs: Iterable[TMorf] | None, outfile: IO[str]) -> float:
        """Renders the full lcov report.

        `morfs` is a list of modules or filenames

        outfile is the file object to write the file into.
        """

        self.coverage.get_data()
        outfile = outfile or sys.stdout

        # ensure file records are sorted by the _relative_ filename, not the full path
        to_report = [
            (fr.relative_filename(), fr, analysis)
            for fr, analysis in get_analysis_to_report(self.coverage, morfs)
        ]
        to_report.sort()

        for fname, fr, analysis in to_report:
            self.total += analysis.numbers
            self.lcov_file(fname, fr, analysis, outfile)

        return self.total.n_statements and self.total.pc_covered

    def lcov_file(
        self,
        rel_fname: str,
        fr: FileReporter,
        analysis: Analysis,
        outfile: IO[str],
    ) -> None:
        """Produces the lcov data for a single file.

        This currently supports both line and branch coverage,
        however function coverage is not supported.
        """

        if analysis.numbers.n_statements == 0:
            if self.config.skip_empty:
                return

        outfile.write(f"SF:{rel_fname}\n")

        lines = sorted(analysis.statements)
        if self.config.lcov_line_checksums:
            source_lines = fr.source().splitlines()
        else:
            source_lines = []

        lcov_lines(analysis, lines, source_lines, outfile)
        lcov_functions(fr, analysis, outfile)
        if analysis.has_arcs:
            lcov_arcs(fr, analysis, lines, outfile)

        outfile.write("end_of_record\n")
