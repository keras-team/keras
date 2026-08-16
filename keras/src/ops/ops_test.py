import inspect

from absl.testing import parameterized

try:
    from keras.api import ops as api_ops_root
except ImportError:
    from keras import ops as api_ops_root

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.ops.operation import Operation
from keras.src.testing.test_utils import named_product
from keras.src.utils.naming import to_snake_case

OPS_MODULES = ("core", "image", "linalg", "math", "nn", "numpy")

SELF_PARAMETER = inspect.Parameter(
    "self", inspect.Parameter.POSITIONAL_OR_KEYWORD
)
NAME_PARAMETER = inspect.Parameter(
    "name", inspect.Parameter.KEYWORD_ONLY, default=None
)

# Parameters with these names are known to always be static (non-tensors).
STATIC_PARAMETER_NAMES = frozenset(
    {"axis", "axes", "dtype", "shape", "newshape", "sparse", "ragged"}
)


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

            # ==== Check handling of name in __init__ ====
            # - op class `__init__` should have a `name` parameter at the end,
            #   which should be keyword only and with a default value of `None`
            # - op class `__init__` should call `super().__init__(name=name)`

            if op_class.__init__ is Operation.__init__:
                # `name` is not keyword only in `Operation`, use this instead.
                class_init_signature = inspect.Signature(
                    [SELF_PARAMETER, NAME_PARAMETER]
                )
            else:
                class_init_signature = inspect.signature(op_class.__init__)

                # Check call to super.
                self.assertContainsSubsequence(
                    inspect.getsource(op_class.__init__),
                    "super().__init__(name=name)",
                    f"`{op_class.__name__}.__init__` is not calling "
                    "`super().__init__(name=name)`",
                )

            static_parameters = list(class_init_signature.parameters.values())
            # Remove `self`.
            static_parameters = static_parameters[1:]
            name_index = -1
            if static_parameters[-1].kind == inspect.Parameter.VAR_KEYWORD:
                # When there is a `**kwargs`, `name` appears before.
                name_index = -2
            # Verify `name` parameter is as expected.
            self.assertEqual(
                static_parameters[name_index],
                NAME_PARAMETER,
                f"The last parameter of `{op_class.__name__}.__init__` "
                "should be `name`, should be a keyword only, and should "
                "have a default value of `None`",
            )
            # Remove `name`, it's not part of the op signature.
            static_parameters.pop(name_index)

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

            op_signature = inspect.signature(op_function)

            for p in dynamic_parameters + static_parameters:
                # Check the same name appears in the op signature
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

            dynamic_parameter_names = [p.name for p in dynamic_parameters]
            static_parameter_names = [p.name for p in static_parameters]

            # Check for obvious mistakes in parameters that were made dynamic
            # but should be static.
            for p in dynamic_parameters:
                self.assertNotIn(
                    p.name,
                    STATIC_PARAMETER_NAMES,
                    f"`{p.name}` should not be a dynamic parameter in op class "
                    f"`{op_class.__name__}` based on its name.",
                )
                self.assertNotIsInstance(
                    p.default,
                    (bool, str),
                    f"`{p.name}` should not be a dynamic parameter in op class "
                    f"`{op_class.__name__}` based on default `{p.default}`.",
                )

            # Check order of parameters.
            if name in (
                "fori_loop",
                "vectorized_map",
                "while_loop",
                "batch_normalization",
                "dot_product_attention",
                "average",
                "einsum",
                "full",
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

            # ==== Check compute_output_spec is implement ====
            # - op class should override Operation's `compute_output_spec`
            self.assertTrue(
                hasattr(op_class, "compute_output_spec")
                and op_class.compute_output_spec
                is not Operation.compute_output_spec,
                f"Op class `{op_class.__name__}` should override "
                "`compute_output_spec`",
            )

    @parameterized.named_parameters(named_product(module_name=OPS_MODULES))
    def test_backend_consistency(self, module_name):
        ops_module = getattr(ops, module_name)
        backend_ops_module = getattr(backend, module_name)

        for op_function, _ in op_functions_and_classes(ops_module):
            name = op_function.__name__

            if hasattr(ops_module, f"_{name}"):
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
