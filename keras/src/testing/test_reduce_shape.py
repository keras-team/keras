from keras.src.ops.operation_utils import printBranchCoverage, reduce_shape

print("-------------BRANCH COVERAGE FOR reduce_shape-----------------")

# set env
shape = (3, 4, 5)

print("Coverage before tests: ")
printBranchCoverage()

#test1 axis is None and keepdims is True 
axis = None
keepdims = True
result = reduce_shape(shape, axis, keepdims)
print("Coverage after test #1: ")
printBranchCoverage()

#test2 axis is None and keepdims is False
axis = None
keepdims = False
result = reduce_shape(shape, axis, keepdims)
print("Coverage after test #2: ")
printBranchCoverage()

#test3 axis is not None and keepdims is True
axis = [1]
keepdims = True
result = reduce_shape(shape, axis, keepdims)
print("Coverage after test #3: ")
printBranchCoverage()

# test4 axis is not None and keepdims is False 
axis = [1]
keepdims = False
result = reduce_shape(shape, axis, keepdims)
print("Coverage after test #4: ")
printBranchCoverage()
