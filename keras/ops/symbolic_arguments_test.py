import tree

from keras import testing
from keras.backend import KerasTensor
from keras.ops.symbolic_arguments import SymbolicArguments


class SymbolicArgumentsTest(testing.TestCase):
    # Testing multiple args and empty kwargs
    def test_args(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        b = KerasTensor(shape=shape)
        args = SymbolicArguments(
            (
                a,
                b,
            ),
            {},
        )

        self.assertEqual(args.keras_tensors, [a, b])
        self.assertEqual(args._flat_arguments, [a, b])
        self.assertEqual(args._single_positional_tensor, None)

    # Testing single arg and single position tensor
    def test_args_single_arg(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        args = SymbolicArguments((a))

        self.assertEqual(args.keras_tensors, [a])
        self.assertEqual(args._flat_arguments, [a])
        self.assertEqual(len(args.kwargs), 0)
        self.assertEqual(isinstance(args.args[0], KerasTensor), True)
        self.assertEqual(args._single_positional_tensor, a)

    # Testing kwargs
    def test_kwargs(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        b = KerasTensor(shape=shape)
        c = KerasTensor(shape=shape)
        args = SymbolicArguments(
            (
                a,
                b,
            ),
            {1: c},
        )

        self.assertEqual(args.keras_tensors, [a, b, c])
        self.assertEqual(args._flat_arguments, [a, b, c])
        self.assertEqual(args._single_positional_tensor, None)

    # Testing conversion function with args and kwargs
    def test_conversion_fn(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        b = KerasTensor(shape=shape)
        c = KerasTensor(shape=shape)
        sym_args = SymbolicArguments(
            (
                a,
                b,
            ),
            {1: c},
        )

        (value, _) = sym_args.convert(lambda x: x**2)
        args1 = value[0][0]

        self.assertIsInstance(args1, KerasTensor)

        mapped_value = tree.map_structure(lambda x: x**2, a)
        self.assertEqual(mapped_value.shape, args1.shape)
        self.assertEqual(mapped_value.dtype, args1.dtype)

    # Testing fill in function with single args only
    def test_fill_in_single_arg(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)

        tensor_dict = {id(a): 3}
        sym_args = SymbolicArguments((a))

        # Call the method to be tested
        result, _ = sym_args.fill_in(tensor_dict)

        self.assertEqual(result, (3,))

    # Testing fill in function with multiple args
    def test_fill_in_multiple_arg(self):
        shape = (2, 3, 4)
        a = KerasTensor(shape=shape)
        b = KerasTensor(shape=shape)

        tensor_dict = {id(b): 2}
        sym_args = SymbolicArguments((a, b))

        # Call the method to be tested
        result, _ = sym_args.fill_in(tensor_dict)

        self.assertEqual(result, ((a, 2),))

    # Testing fill in function for args and kwargs
    def test_fill_in(self):
        shape1 = (2, 3, 4)
        shape2 = (3, 2, 4)
        a = KerasTensor(shape=shape1)
        b = KerasTensor(shape=shape2)
        c = KerasTensor(shape=shape2)
        dictionary = {id(a): 3, id(c): 2}
        sym_args = SymbolicArguments(
            (
                a,
                b,
            ),
            {1: c},
        )

        (values, _) = sym_args.fill_in(dictionary)

        self.assertEqual(values, ((3, b), {1: 2}))
