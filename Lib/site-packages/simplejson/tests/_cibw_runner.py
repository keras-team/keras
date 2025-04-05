"""Internal module for running tests from cibuildwheel"""

import sys
import simplejson.tests

if __name__ == '__main__':
    simplejson.tests.main(project_dir=sys.argv[1])
