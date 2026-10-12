"""Console reporting for tests made of checks (`subTest(check=...)`).

Registered from the root `conftest.py`. With pytest 9 (native subtests):

* A check report prints nothing of its own, so a test with checks is one
  console line again instead of two.
* The test's status word carries its check count and turns red when a
  check failed: `PASSED (12 checks)`, `FAILED (1/4 checks)`.
* The terminal summary lists failed checks per test with their message,
  and skipped checks grouped by reason.

pytest's own categories are kept for every report, so the final tally,
`-x`, exit codes and JUnit output are unchanged. On pytest < 9 the plugin
is a no-op. Disable with `-p no:check-console`.
"""

import collections

import pytest

_SubtestReport = getattr(pytest, "SubtestReport", ())


def _check_id(report):
    """`symbolic_partial[(None, 3)]` from the `subTest` params."""
    params = dict(report.context.kwargs)
    check = params.pop("check", report.context.msg or "check")
    extra = ", ".join(str(v) for v in params.values())
    return f"{check}[{extra}]" if extra else check


def _message(report):
    """The skip reason, or the first informative line of the failure."""
    longrepr = report.longrepr
    if isinstance(longrepr, tuple):  # Skip: (path, lineno, reason).
        return longrepr[2].removeprefix("Skipped: ")
    crash = getattr(longrepr, "reprcrash", None)
    lines = str(crash.message if crash else longrepr).splitlines()
    lines = [line.strip() for line in lines]
    return next((ln for ln in lines if ln and ln != "AssertionError:"), "")


class CheckConsole:
    def __init__(self):
        self.counts = collections.Counter()  # nodeid -> checks so far
        self.failed = collections.defaultdict(list)  # nodeid -> [(id, msg)]
        self.skipped = collections.Counter()  # reason -> checks
        self.totals = collections.Counter()  # outcome -> checks

    @pytest.hookimpl(tryfirst=True)
    def pytest_report_teststatus(self, report):
        if report.when != "call":
            return None
        if isinstance(report, _SubtestReport):
            self.counts[report.nodeid] += 1
            self.totals[report.outcome] += 1
            if report.failed:
                self.failed[report.nodeid].append(
                    (_check_id(report), _message(report))
                )
            elif report.skipped:
                self.skipped[_message(report)] += 1
            # pytest's category, no letter or word: counted, not printed.
            category = "subtests passed" if report.passed else report.outcome
            return category, "", ""
        n_failed = len(self.failed.get(report.nodeid, ()))
        n = self.counts.pop(report.nodeid, 0)
        if n_failed:
            word = f"FAILED ({n_failed}/{n} checks)"
            return report.outcome, "F", (word, {"red": True})
        if n and report.passed:
            return "passed", ".", (f"PASSED ({n} checks)", {"green": True})
        return None

    def pytest_terminal_summary(self, terminalreporter):
        tr = terminalreporter
        if self.failed:
            n = self.totals["failed"]
            tr.write_sep("=", f"{n} failed checks", red=True)
            for nodeid, checks in self.failed.items():
                tr.write_line(nodeid.split("::", 1)[-1], bold=True)
                for check, message in checks:
                    tr.write(f"  {check:<32}", red=True)
                    tr.write_line(message)
        if self.skipped:
            n = self.totals["skipped"]
            tr.write_sep("=", f"{n} skipped checks", yellow=True)
            reasons = self.skipped.most_common()
            tr.write_line(", ".join(f"{n} x {r!r}" for r, n in reasons))
        if self.totals:
            totals = ", ".join(
                f"{n} {k}" for k, n in sorted(self.totals.items())
            )
            tr.write_sep("=", f"checks: {totals}")


def pytest_configure(config):
    config.pluginmanager.register(CheckConsole(), "check-console")
