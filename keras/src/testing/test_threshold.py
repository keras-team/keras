from keras.src.metrics.metrics_utils import assert_thresholds_range, printBranchCoverage
  

print("-------------BRANCH COVERAGE FOR assert_thresholds_range-----------------")
# set env
thresholds = None
invalid_thresholds = None

print("Coverage before tests: ")
printBranchCoverage()

#test1
thresholds = [0,1] # just a list so threshold is not None
assert_thresholds_range(thresholds)
print("Coverage after test #1: ")
printBranchCoverage()

#test2
invalid_thresholds = [0.5, -0.1, 1.2]  # a list with invalid values (-0.1 and 1.2)
try:
    assert_thresholds_range(invalid_thresholds)
except ValueError as e:
    print(e)
    print("Coverage after test #2: ")
    printBranchCoverage()
