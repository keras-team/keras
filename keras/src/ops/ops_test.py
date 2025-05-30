import inspect

from absl.testing import parameterized

from keras.api import ops as api_ops_root
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.ops.operation import Operation
from keras.src.testing.test_utils import named_product
from keras.src.utils.naming import to_snake_case

OPS_MODULES = ("core", "image", "linalg", "math", "nn", "numpy")


def op_functions_and_classes(ops_module):
    """Enumerate pairs of op function and op classes in a module.

    Will return for instance `(ExpandDims, expand_dims)`, `(Sum, sum)`, ...

    Args:
        ops_module: the module to explore.

    Returns:
        iterable returning tuples with function and class pairs.
    """
    # Go through all symbols.
    for op_class_name in dir(ops_module):
        op_class = getattr(ops_module, op_class_name)
        # Find the ones that are classes that extend `Operation`.
        if isinstance(op_class, type) and Operation in op_class.__mro__:
            # Infer what the corresponding op function name should be.
            op_function_name = to_snake_case(op_class_name)
            # With some exceptions.
            op_function_name = {
                "batch_norm": "batch_normalization",
                "rms_norm": "rms_normalization",
                "search_sorted": "searchsorted",
            }.get(op_function_name, op_function_name)
            # Check if that function exist. Some classes are abstract super
            # classes for multiple operations and should be ignored.
            op_function = getattr(ops_module, op_function_name, None)
            if op_function is not None:
                # We have a pair, return it.
                yield op_function, op_class


class OperationTest(testing.TestCase):
    @parameterized.named_parameters(named_product(module_name=OPS_MODULES))
    def test_class_function_consistency(self, module_name):
        ops_module = getattr(ops, module_name)
        if module_name in ("core", "math"):
            # `core` and `math` are not exported as their own module.
            api_ops_module = None
        else:
            api_ops_module = getattr(api_ops_root, module_name)

        for op_function, op_class in op_functions_and_classes(ops_module):
            name = op_function.__name__

            # ==== Check exports ====
            # - op should be exported as e.g. `keras.ops.numpy.sum`
            # - op should also be exported as e.g. `keras.ops.sum`

            if module_name != "image":
                # `image` ops are not exported at the top-level.
                self.assertIsNotNone(
                    getattr(api_ops_root, name, None),
                    f"Not exported as `keras.ops.{name}`",
                )
            if api_ops_module is not None:
                # `core` and `math` are not exported as their own module.
                self.assertIsNotNone(
                    getattr(api_ops_module, name, None),
                    f"Not exported as `keras.ops.{module_name}.{name}`",
                )

            # ==== Check static parameters ====
            # Static parameters are declared in the class' `__init__`.
            # Dynamic parameters are declared in the class' `call` method.
            # - they should all appear in the op signature with the same name
            # - they should have the same default value
            # - they should appear in the same order and usually with the
            #   dynamic parameters first, and the static parameters last.

            dynamic_parameters = list(
                inspect.signature(op_class.call).parameters.values()
            )[1:]  # Remove self

            if op_class.__init__ is Operation.__init__:
                # This op class has no static parameters. Do not use the `name`
                # and `dtype` parameters from the `__init__` of `Operation`.
                static_parameters = []
            else:
                class_init_signature = inspect.signature(op_class.__init__)
                static_parameters = list(
                    class_init_signature.parameters.values()
                )[1:]  # Remove self

            op_signature = inspect.signature(op_function)

            for p in dynamic_parameters + static_parameters:
                # Check the same name appeas in the op signature
                self.assertIn(
                    p.name,
                    op_signature.parameters,
                    f"Op function `{name}` is missing a parameter that is in "
                    f"op class `{op_class.__name__}`",
                )
                # Check default values are the same
                self.assertEqual(
                    p.default,
                    op_signature.parameters[p.name].default,
                    f"Default mismatch for parameter `{p.name}` between op "
                    f"function `{name}` and op class `{op_class.__name__}`",
                )

            # Check order of parameters.
            dynamic_parameter_names = [p.name for p in dynamic_parameters]
            static_parameter_names = [p.name for p in static_parameters]

            if name in (
                "fori_loop",
                "while_loop",
                "batch_normalization",
                "dot_product_attention",
                "average",
                "einsum",
                "pad",
            ):
                # Loose case:
                # order of of parameters is preserved but they are interspersed.
                op_dynamic_parameter_names = [
                    name
                    for name in op_signature.parameters.keys()
                    if name in dynamic_parameter_names
                ]
                self.assertEqual(
                    op_dynamic_parameter_names,
                    dynamic_parameter_names,
                    "Inconsistent dynamic parameter order for op "
                    f"function `{name}` and op class `{op_class.__name__}`",
                )
                op_static_parameter_names = [
                    name
                    for name in op_signature.parameters.keys()
                    if name in static_parameter_names
                ]
                self.assertEqual(
                    op_static_parameter_names,
                    static_parameter_names,
                    "Inconsistent static parameter order for op "
                    f"function `{name}` and op class `{op_class.__name__}`",
                )
            else:
                # Strict case:
                # dynamic parameters first and static parameters at the end.
                self.assertEqual(
                    list(op_signature.parameters.keys()),
                    dynamic_parameter_names + static_parameter_names,
                    "Inconsistent static parameter position for op "
                    f"function `{name}` and op class `{op_class.__name__}`",
                )

    @parameterized.named_parameters(named_product(module_name=OPS_MODULES))
    def test_backend_consistency(self, module_name):
        ops_module = getattr(ops, module_name)
        backend_ops_module = getattr(backend, module_name)

        for op_function, _ in op_functions_and_classes(ops_module):
            name = op_function.__name__

            if hasattr(ops_module, "_" + name):
                # For an op function `foo`, if there is a function named `_foo`,
                # that means we have a backend independent implementation.
                continue
            if name in ("view_as_complex", "view_as_real", "get_item"):
                # These ops have an inlined backend independent implementation.
                continue

            # ==== Check backend implementation ====
            # - op should have an implementation in every backend
            # - op implementation should have the same signature (same
            #   parameters, same order, same defaults)

            backend_op_function = getattr(backend_ops_module, name, None)

            if backend.backend() == "openvino" and backend_op_function is None:
                # Openvino is still missing a number of ops.
                continue

            self.assertIsNotNone(backend_op_function, f"Missing op `{name}`")

            if name == "multi_hot":
                # multi_hot has code to massage the input parameters before
                # calling the backend implementation, so the signature is
                # different on purpose.
                continue

            # Signature should match in every way.
            self.assertEqual(
                inspect.signature(backend_op_function),
                inspect.signature(op_function),
                f"Signature mismatch for `{name}`",
            )
