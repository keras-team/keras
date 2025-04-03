# SPDX-License-Identifier: Apache-2.0


"""Opset registry."""

import collections
import inspect

from tf2onnx import constants

# pylint: disable=unused-argument,missing-docstring,invalid-name


class tf_op:
    """Class to implement the decorator to register handlers that map tf to onnx."""

    # Maps domains (string) to lists (idx represents opset) of dicts (key = op to handle, value = handler)
    _OPSETS = collections.OrderedDict()
    # Cache of mapping for current domain and opset. Maps op names to handlers [(func, kwargs) tuple]
    _MAPPING = None
    # Cache of mapping from domain to map of op name to handlers. Used to fetch handlers from different domains
    _DOMAIN_MAPPING = None

    def __init__(self, name, domain=constants.ONNX_DOMAIN, **kwargs):
        """Called decorator from decorator.

        :param name: The name (or list of names) of the tensorflow operator.
        :param domain: The domain the handler requires, defaults to onnx.
        :param kwargs: Dictionary that are passed to the handler. A key 'onnx_op' will change the operator name.
        """
        if not isinstance(name, list):
            name = [name]
        self.names = name
        self.domain = domain
        self.kwargs = kwargs

    def __call__(self, func):
        for k, v in inspect.getmembers(func, inspect.ismethod):
            if k.startswith("version_"):
                version = int(k.replace("version_", ""))
                tf_op.register_handler(v, version, self.names, self.domain, self.kwargs)
        return func

    def register_compat_handler(self, func, version):
        """Register old style custom handler.

        :param func: The handler.
        :param version: The version of the handler.
        """
        tf_op.register_handler(func, version, self.names, self.domain, self.kwargs)

    @staticmethod
    def register_handler(func, version, names, domain, kwargs):
        """Register handler.

        :param func: The handler.
        :param version: (int) The opset of onnx (or other domain) required for the handler.
        :param names: List of names of the operators to convert.
        :param domain: The domain the handler requires, defaults to onnx.

        """
        opset = tf_op._OPSETS.get(domain)
        if not opset:
            opset = []
            tf_op._OPSETS[domain] = opset
        while version >= len(opset):
            opset.append({})
        opset_dict = opset[version]
        for name in names:
            opset_dict[name] = (func, kwargs)

    @staticmethod
    def get_opsets():
        return tf_op._OPSETS

    @staticmethod
    def create_mapping(max_onnx_opset_version, extra_opsets):
        """Create the final mapping dictionary by stacking domains and opset versions.

        :param max_onnx_opset_version: The highest onnx opset the resulting graph may use.
        :param extra_opsets: Extra opsets the resulting graph may use.
        """
        mapping = {constants.ONNX_DOMAIN: max_onnx_opset_version}
        if extra_opsets:
            for extra_opset in extra_opsets:
                mapping[extra_opset.domain] = extra_opset.version
        ops_mapping = {}
        domain_to_ops_mapping = collections.defaultdict(dict)
        for domain, opsets in tf_op.get_opsets().items():
            for target_opset, op_map in enumerate(opsets):
                m = mapping.get(domain)
                if m:
                    if target_opset <= m and op_map:
                        domain_to_ops_mapping[domain].update(ops_mapping)
                        ops_mapping.update(op_map)

        tf_op._MAPPING = ops_mapping
        tf_op._DOMAIN_MAPPING = domain_to_ops_mapping
        return ops_mapping

    @staticmethod
    def find_effective_op(name, domain=None):
        """Find the effective version of an op create_mapping.
           This is used if we need to compose ops from other ops where we'd need to find the
           op that is going to be used in the final graph, for example there is a custom op
           that overrides a onnx op ...

        :param name: The operator name.
        :param domain: The domain to use (optional).
        """
        if domain is None:
            map_info = tf_op._MAPPING.get(name)
        else:
            map_info = tf_op._DOMAIN_MAPPING[domain].get(name)
        if map_info is None:
            return None
        return map_info


class tfl_op:
    """Class to implement the decorator to register handlers that map tflite to tf or onnx."""

    def __init__(self, name, domain=constants.ONNX_DOMAIN, **kwargs):
        """Called decorator from decorator.

        :param name: The name (or list of names) of the tflite operator.
        :param domain: The domain the operator belongs to, defaults to onnx. Use 'com.google.tensorflow' for tflite->tf
        :param kwargs: Dictionary that are passed to the handler. A key 'onnx_op' will change the operator name.
                       'tf_op' will convert the op to tf during a tflite to tf conversion pass.
        """
        if not isinstance(name, list):
            name = [name]
        self.names = name
        self.domain = domain
        self.kwargs = kwargs

    def __call__(self, func):
        # Register any handlers of the form 'version_#'
        tf_op(self.names, self.domain, **self.kwargs)(func)
        # TFLite to TF handlers have the function name 'to_tf' which takes the optional 'tf_op' kwarg
        if hasattr(func, 'to_tf'):
            tf_op.register_handler(func.to_tf, 0, self.names, 'com.google.tensorflow', self.kwargs)
        return func

    @staticmethod
    def create_tfl_to_tf_mapping():
        return tf_op.get_opsets()['com.google.tensorflow'][0]
