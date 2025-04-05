# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt

"""Json reporting for coverage.py"""

from __future__ import annotations

import datetime
import json
import sys

from collections.abc import Iterable
from typing import Any, IO, TYPE_CHECKING

from coverage import __version__
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.types import TMorf, TLineNo

if TYPE_CHECKING:
    from coverage import Coverage
    from coverage.data import CoverageData
    from coverage.plugin import FileReporter


# A type for data that can be JSON-serialized.
JsonObj = dict[str, Any]

# "Version 1" had no format number at all.
# 2: add the meta.format field.
# 3: add region information (functions, classes)
FORMAT_VERSION = 3

class JsonReporter:
    """A reporter for writing JSON coverage results."""

    report_type = "JSON report"

    def __init__(self, coverage: Coverage) -> None:
        self.coverage = coverage
        self.config = self.coverage.config
        self.total = Numbers(self.config.precision)
        self.report_data: JsonObj = {}

    def make_summary(self, nums: Numbers) -> JsonObj:
        """Create a dict summarizing `nums`."""
        return {
            "covered_lines": nums.n_executed,
            "num_statements": nums.n_statements,
            "percent_covered": nums.pc_covered,
            "percent_covered_display": nums.pc_covered_str,
            "missing_lines": nums.n_missing,
            "excluded_lines": nums.n_excluded,
        }

    def make_branch_summary(self, nums: Numbers) -> JsonObj:
        """Create a dict summarizing the branch info in `nums`."""
        return {
            "num_branches": nums.n_branches,
            "num_partial_branches": nums.n_partial_branches,
            "covered_branches": nums.n_executed_branches,
            "missing_branches": nums.n_missing_branches,
        }

    def report(self, morfs: Iterable[TMorf] | None, outfile: IO[str]) -> float:
        """Generate a json report for `morfs`.

        `morfs` is a list of modules or file names.

        `outfile` is a file object to write the json to.

        """
        outfile = outfile or sys.stdout
        coverage_data = self.coverage.get_data()
        coverage_data.set_query_contexts(self.config.report_contexts)
        self.report_data["meta"] = {
            "format": FORMAT_VERSION,
            "version": __version__,
            "timestamp": datetime.datetime.now().isoformat(),
            "branch_coverage": coverage_data.has_arcs(),
            "show_contexts": self.config.json_show_contexts,
        }

        measured_files = {}
        for file_reporter, analysis in get_analysis_to_report(self.coverage, morfs):
            measured_files[file_reporter.relative_filename()] = self.report_one_file(
                coverage_data,
                analysis,
                file_reporter,
            )

        self.report_data["files"] = measured_files
        self.report_data["totals"] = self.make_summary(self.total)

        if coverage_data.has_arcs():
            self.report_data["totals"].update(self.make_branch_summary(self.total))

        json.dump(
            self.report_data,
            outfile,
            indent=(4 if self.config.json_pretty_print else None),
        )

        return self.total.n_statements and self.total.pc_covered

    def report_one_file(
        self, coverage_data: CoverageData, analysis: Analysis, file_reporter: FileReporter
    ) -> JsonObj:
        """Extract the relevant report data for a single file."""
        nums = analysis.numbers
        self.total += nums
        summary = self.make_summary(nums)
        reported_file: JsonObj = {
            "executed_lines": sorted(analysis.executed),
            "summary": summary,
            "missing_lines": sorted(analysis.missing),
            "excluded_lines": sorted(analysis.excluded),
        }
        if self.config.json_show_contexts:
            reported_file["contexts"] = coverage_data.contexts_by_lineno(analysis.filename)
        if coverage_data.has_arcs():
            summary.update(self.make_branch_summary(nums))
            reported_file["executed_branches"] = list(
                _convert_branch_arcs(analysis.executed_branch_arcs()),
            )
            reported_file["missing_branches"] = list(
                _convert_branch_arcs(analysis.missing_branch_arcs()),
            )

        num_lines = len(file_reporter.source().splitlines())
        for noun, plural in file_reporter.code_region_kinds():
            reported_file[plural] = region_data = {}
            outside_lines = set(range(1, num_lines + 1))
            for region in file_reporter.code_regions():
                if region.kind != noun:
                    continue
                outside_lines -= region.lines
                region_data[region.name] = self.make_region_data(
                    coverage_data,
                    analysis.narrow(region.lines),
                )

            region_data[""] = self.make_region_data(
                coverage_data,
                analysis.narrow(outside_lines),
            )
        return reported_file

    def make_region_data(self, coverage_data: CoverageData, narrowed_analysis: Analysis) -> JsonObj:
        """Create the data object for one region of a file."""
        narrowed_nums = narrowed_analysis.numbers
        narrowed_summary = self.make_summary(narrowed_nums)
        this_region = {
            "executed_lines": sorted(narrowed_analysis.executed),
            "summary": narrowed_summary,
            "missing_lines": sorted(narrowed_analysis.missing),
            "excluded_lines": sorted(narrowed_analysis.excluded),
        }
        if self.config.json_show_contexts:
            contexts = coverage_data.contexts_by_lineno(narrowed_analysis.filename)
            this_region["contexts"] = contexts
        if coverage_data.has_arcs():
            narrowed_summary.update(self.make_branch_summary(narrowed_nums))
            this_region["executed_branches"] = list(
                _convert_branch_arcs(narrowed_analysis.executed_branch_arcs()),
            )
            this_region["missing_branches"] = list(
                _convert_branch_arcs(narrowed_analysis.missing_branch_arcs()),
            )
        return this_region


def _convert_branch_arcs(
    branch_arcs: dict[TLineNo, list[TLineNo]],
) -> Iterable[tuple[TLineNo, TLineNo]]:
    """Convert branch arcs to a list of two-element tuples."""
    for source, targets in branch_arcs.items():
        for target in targets:
            yield source, target
