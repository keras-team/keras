import copy
import warnings

import torch

from keras.src import backend
from keras.src import ops
from keras.src import tree
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.module_utils import torch_xla


class TorchExportArchive:
    def track(self, resource):
        raise NotImplementedError(
            "`track` is not implemented in the torch backend. Use"
            "`track_and_add_endpoint` instead."
        )

    def add_endpoint(self, name, fn, input_signature, **kwargs):
        raise NotImplementedError(
            "`add_endpoint` is not implemented in the torch backend. Use"
            "`track_and_add_endpoint` instead."
        )

    def track_and_add_endpoint(self, name, resource, input_signature, **kwargs):
        # Disable false alarms related to lifting parameters.
        warnings.filterwarnings("ignore", message=".*created when tracing.*")
        warnings.filterwarnings(
            "ignore", message=".*Unable to find the path of the module.*"
        )

        if not isinstance(resource, torch.nn.Module):
            raise TypeError(
                "`resource` must be an instance of `torch.nn.Module`. "
                f"Received: resource={resource} (of type {type(resource)})"
            )

        def _check_input_signature(input_spec):
            for s in tree.flatten(input_spec.shape):
                if s is None:
                    raise ValueError(
                        "The shape in the `input_spec` must be fully "
                        f"specified. Received: input_spec={input_spec}"
                    )

        def _to_torch_tensor(x, replace_none_number=1):
            shape = backend.standardize_shape(x.shape)
            shape = tuple(
                s if s is not None else replace_none_number for s in shape
            )
            return ops.ones(shape, x.dtype)

        tree.map_structure(_check_input_signature, input_signature)
        sample_inputs = tree.map_structure(_to_torch_tensor, input_signature)
        sample_inputs = tuple(sample_inputs)

        # Ref: torch_xla.tf_saved_model_integration
        # TODO: Utilize `dynamic_shapes`
        exported = torch.export.export(
            resource, sample_inputs, dynamic_shapes=None, strict=False
        )
        options = torch_xla.stablehlo.StableHLOExportOptions(
            override_tracing_arguments=sample_inputs
        )
        stablehlo_model = torch_xla.stablehlo.exported_program_to_stablehlo(
            exported, options
        )
        state_dict_keys = list(stablehlo_model._bundle.state_dict.keys())

        # Remove unused variables.
        for k in state_dict_keys:
            if "lifted" not in k:
                stablehlo_model._bundle.state_dict.pop(k)

        bundle = copy.deepcopy(stablehlo_model._bundle)
        bundle.state_dict = {
            k: tf.Variable(v, trainable=False, name=k)
            for k, v in bundle.state_dict.items()
        }
        bundle.additional_constants = [
            tf.Variable(v, trainable=False) for v in bundle.additional_constants
        ]

        # Track variables in `bundle` for `write_out`.
        self._tf_trackable.variables += (
            list(bundle.state_dict.values()) + bundle.additional_constants
        )

        # Ref: torch_xla.tf_saved_model_integration.save_stablehlo_graph_as_tf
        def make_tf_function(func, bundle):
            from tensorflow.compiler.tf2xla.python import xla as tfxla

            def _get_shape_with_dynamic(signature):
                shape = copy.copy(signature.shape)
                for i in signature.dynamic_dims:
                    shape[i] = None
                return shape

            def _extract_call_parameters(args, meta, bundle):
                call_args = []
                if meta.input_pytree_spec is not None:
                    args = tree.flatten(args)
                for loc in meta.input_locations:
                    if loc.type_ == torch_xla.stablehlo.VariableType.PARAMETER:
                        call_args.append(bundle.state_dict[loc.name])
                    elif loc.type_ == torch_xla.stablehlo.VariableType.CONSTANT:
                        call_args.append(
                            bundle.additional_constants[loc.position]
                        )
                    else:
                        call_args.append(args[loc.position])
                return call_args

            def inner(*args):
                Touts = [sig.dtype for sig in func.meta.output_signature]
                Souts = [
                    _get_shape_with_dynamic(sig)
                    for sig in func.meta.output_signature
                ]
                call_args = _extract_call_parameters(args, func.meta, bundle)
                results = tfxla.call_module(
                    tuple(call_args),
                    version=5,
                    Tout=Touts,  # dtype information
                    Sout=Souts,  # Shape information
                    function_list=[],
                    module=func.bytecode,
                )
                if len(Souts) == 1:
                    results = results[0]
                return results

            return inner

        decorated_fn = tf.function(
            make_tf_function(
                stablehlo_model._bundle.stablehlo_funcs[0], bundle
            ),
            input_signature=input_signature,
        )
        return decorated_fn
