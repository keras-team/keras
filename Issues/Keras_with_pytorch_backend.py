import keras
from keras import ops

keras.config.set_backend("torch")

def tensor_operations_example():
    # Create tensors
    x = ops.array([[1, 2, 3], [4, 5, 6]])
    y = ops.array([[10, 20, 30], [40, 50, 60]])

    # Perform elementwise operations
    z_add = ops.add(x, y)
    z_mul = ops.multiply(x, y)
    z_mean = ops.mean(z_mul)
    z_norm = ops.sqrt(ops.sum(ops.square(x)))

    print("x + y =\n", z_add)
    print("x * y =\n", z_mul)
    print("Mean(x * y) =", z_mean)
    print("L2 norm of x =", z_norm)

tensor_operations_example()
