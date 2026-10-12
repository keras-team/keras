import pytest

pytest_plugins = "pytester"

CHECKS_TEST = """
import unittest


class ChecksTest(unittest.TestCase):
    def test_all_pass(self):
        for check in ("eager", "symbolic_static", "symbolic_dynamic"):
            with self.subTest(check=check):
                pass

    def test_one_fails(self):
        for check in ("eager", "symbolic_static", "symbolic_dynamic"):
            with self.subTest(check=check):
                if check == "symbolic_static":
                    self.fail("Output rank differs: symbolic=(2, 3, 1)")
                if check == "symbolic_dynamic":
                    self.skipTest("symbolic_static failed")

    def test_partial_mask_fails(self):
        with self.subTest(check="symbolic_partial"):
            with self.subTest(shapes="(None, 3)"):
                raise ValueError("needs static shape")

    def test_no_checks(self):
        pass
"""


@pytest.mark.skipif(
    not hasattr(pytest, "SubtestReport"), reason="Needs pytest >= 9"
)
class CheckConsoleTest:
    def test_console_output(self, pytester):
        pytester.makepyfile(CHECKS_TEST)
        result = pytester.runpytest(
            "-v",
            "-p",
            "keras.src.testing.check_console",
            "-p",
            "no:cacheprovider",
        )
        result.stdout.fnmatch_lines(
            [
                "*::test_all_pass PASSED (3 checks)*",
                "*::test_one_fails FAILED (1/3 checks)*",
                "*::test_partial_mask_fails FAILED (1/2 checks)*",
                "*::test_no_checks PASSED *",
                "*2 failed checks*",
                "ChecksTest::test_one_fails",
                "  symbolic_static*Output rank differs: symbolic=(2, 3, 1)",
                "ChecksTest::test_partial_mask_fails",
                "  symbolic_partial[[](None, 3)[]]*needs static shape",
                "*1 skipped checks*",
                "1 x 'symbolic_static failed'",
                "*checks: 2 failed, 5 passed, 1 skipped*",
            ]
        )
        # No dangling node-id line before the result line (pytest 9 prints
        # one when the first check reports).
        result.stdout.no_fnmatch_line("*::test_all_pass ")
        # pytest's own tally still counts the failed checks.
        result.assert_outcomes(passed=4, failed=2, skipped=1)
