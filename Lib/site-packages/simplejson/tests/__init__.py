from __future__ import absolute_import
import unittest
import sys
import os


class NoExtensionTestSuite(unittest.TestSuite):
    def run(self, result):
        import simplejson

        simplejson._toggle_speedups(False)
        result = unittest.TestSuite.run(self, result)
        simplejson._toggle_speedups(True)
        return result


class TestMissingSpeedups(unittest.TestCase):
    def runTest(self):
        if hasattr(sys, "pypy_translation_info"):
            "PyPy doesn't need speedups! :)"
        elif hasattr(self, "skipTest"):
            self.skipTest("_speedups.so is missing!")


def additional_tests(suite=None, project_dir=None):
    import simplejson
    import simplejson.encoder
    import simplejson.decoder

    if suite is None:
        suite = unittest.TestSuite()
    try:
        import doctest
    except ImportError:
        if sys.version_info < (2, 7):
            # doctests in 2.6 depends on cStringIO
            return suite
        raise
    for mod in (simplejson, simplejson.encoder, simplejson.decoder):
        suite.addTest(doctest.DocTestSuite(mod))
    if project_dir is not None:
        suite.addTest(
            doctest.DocFileSuite(
                os.path.join(project_dir, "index.rst"), module_relative=False
            )
        )
    return suite


def all_tests_suite(project_dir=None):
    def get_suite():
        suite_names = [
            "simplejson.tests.%s" % (os.path.splitext(f)[0],)
            for f in os.listdir(os.path.dirname(__file__))
            if f.startswith("test_") and f.endswith(".py")
        ]
        return additional_tests(
            suite=unittest.TestLoader().loadTestsFromNames(suite_names),
            project_dir=project_dir,
        )

    suite = get_suite()
    import simplejson

    if simplejson._import_c_make_encoder() is None:
        suite.addTest(TestMissingSpeedups())
    else:
        suite = unittest.TestSuite(
            [
                suite,
                NoExtensionTestSuite([get_suite()]),
            ]
        )
    return suite


def main(project_dir=None):
    runner = unittest.TextTestRunner(verbosity=1 + sys.argv.count("-v"))
    suite = all_tests_suite(project_dir=project_dir)
    raise SystemExit(not runner.run(suite).wasSuccessful())


if __name__ == "__main__":
    import os
    import sys

    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.insert(0, project_dir)
    main(project_dir=project_dir)
