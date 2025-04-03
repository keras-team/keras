# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.schema
"""

import logging
import copy
from collections import defaultdict, OrderedDict
from onnx import defs, helper, TensorProto, OperatorSetIdProto, shape_inference

from . import constants
from . import utils

logger = logging.getLogger(__name__)


class OnnxOpSchema(object):
    """Wrapper for Onnx schema."""

    def __init__(self, name, domain, since_version, attributes):
        """Create a Onnx schema
        Args:
            name (str): op name
            attributes (List[str]): valid attributes
            domain (str): default value "" means it's Onnx domain
            since_version (int): opset version, default is 1
        """
        self._name = name
        self._domain = domain
        self._attributes = attributes
        self._since_version = since_version

    @property
    def attributes(self):
        return self._attributes

    @property
    def domain(self):
        return self._domain

    @property
    def name(self):
        return self._name

    @property
    def since_version(self):
        return self._since_version

    @staticmethod
    def from_onnx_schema(onnx_schema):
        name = onnx_schema.name
        domain = onnx_schema.domain
        since_version = int(onnx_schema.since_version)
        attributes = onnx_schema.attributes
        return OnnxOpSchema(name, domain, since_version, attributes)

    def has_attribute(self, attr):
        return attr in self.attributes


def _register_all_schemas_with_history():
    """Register all schemas with history"""
    onnx_schemas = defs.get_all_schemas_with_history()
    name_domain_version_schema_map = defaultdict(lambda: defaultdict(dict))
    for s in onnx_schemas:
        schema = OnnxOpSchema.from_onnx_schema(s)
        name_domain_version_schema_map[schema.name][schema.domain][schema.since_version] = schema

    ordered_map = defaultdict(lambda: defaultdict(OrderedDict))
    for name, domain_version_schema_map in name_domain_version_schema_map.items():
        for domain, version_schema_map in domain_version_schema_map.items():
            ordered_map[name][domain] = OrderedDict(
                sorted(version_schema_map.items(), key=lambda x: -x[0])
            )
    return ordered_map


def _parse_domain_opset_versions(schemas):
    """ Get max opset version among all schemas within each domain. """
    domain_opset_versions = dict()
    for domain_version_schema_map in schemas.values():
        for domain, version_schema_map in domain_version_schema_map.items():
            # version_schema_map is sorted by since_version in descend order
            max_version = next(iter(version_schema_map))
            if domain not in domain_opset_versions:
                domain_opset_versions[domain] = int(max_version)
            else:
                domain_opset_versions[domain] = max(domain_opset_versions[domain], int(max_version))
    return domain_opset_versions


# format is <OpName, <Domain, <SinceVersion, OpSchema>>>
# SinceVersion is sorted from high to low
_schemas = _register_all_schemas_with_history()

_domain_opset_versions = _parse_domain_opset_versions(_schemas)


def get_schema(name, max_inclusive_opset_version, domain=None):
    """Get schema by name within specific version."""
    domain = domain or constants.ONNX_DOMAIN
    domain_version_schema_map = _schemas[name]
    version_schema_map = domain_version_schema_map[domain]
    for version, schema in version_schema_map.items():
        if version <= max_inclusive_opset_version:
            return schema
    return None


def get_max_supported_opset_version(domain=None):
    """Get max supported opset version by current onnx package given a domain."""
    domain = domain or constants.ONNX_DOMAIN
    return _domain_opset_versions.get(domain, None)


def infer_onnx_shape_dtype(node, opset_version, input_shapes, input_dtypes, initializers=None):
    """
    Infer shapes and dtypes for outputs of the node.
    Sometimes, shape inference needs the values of node's inputs, so initializers are used.
    """

    def build_onnx_op(node):
        """Build onnx op"""
        onnx_node = helper.make_node(node.type, node.input, node.output, name=node.name)
        # deal with attributes
        attr = []
        attr_graphs = node.get_body_graphs()
        if attr_graphs:
            for attr_name, sub_graph in attr_graphs.items():
                copied_sub_graph = copy.deepcopy(sub_graph)
                graph_proto = copied_sub_graph.make_graph("graph for " + node.name + " " + attr_name)
                attr.append(helper.make_attribute(attr_name, graph_proto))
        attr.extend(node.get_onnx_attrs().values())
        if attr:
            onnx_node.attribute.extend(attr)
        return onnx_node

    inputs = []
    outputs = []
    for inp, shape, dtype in zip(node.input, input_shapes, input_dtypes):
        inputs.append(utils.make_onnx_inputs_outputs(inp, dtype, shape))
    for output in node.output:
        outputs.append(utils.make_onnx_inputs_outputs(output, TensorProto.UNDEFINED, None))
    graph_proto = helper.make_graph([build_onnx_op(node)], "infer-graph", inputs, outputs, initializer=initializers)
    imp = OperatorSetIdProto()
    imp.version = opset_version
    model_proto = helper.make_model(graph_proto, opset_imports=[imp])

    inferred_model = None
    try:
        try:
            inferred_model = shape_inference.infer_shapes(model_proto, strict_mode=False)
        except TypeError:
            # strict_mode arg doesn't exist in old onnx packages
            inferred_model = shape_inference.infer_shapes(model_proto)
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "ONNX Failed to infer shapes and dtypes for [%s, type: %s]",
            node.name, node.type, exc_info=1
        )
        return None, None

    shapes = {}
    dtypes = {}
    for output in inferred_model.graph.output:
        tensor_type = output.type.tensor_type
        if tensor_type.HasField("elem_type"):
            dtypes[output.name] = tensor_type.elem_type
        else:
            dtypes[output.name] = TensorProto.UNDEFINED
        # Missing dim_value in shapes of onnx means unknown which is -1 in our convertor
        if tensor_type.HasField("shape"):
            shapes[output.name] = [
                dim.dim_value if dim.HasField("dim_value") else -1 for dim in tensor_type.shape.dim
            ]
        else:
            shapes[output.name] = None
    output_shapes = []
    output_dtypes = []
    for output in node.output:
        if output in shapes:
            output_shapes.append(shapes[output])
        else:
            output_shapes.append(None)
        if output in dtypes:
            output_dtypes.append(dtypes[output])
        else:
            output_dtypes.append(TensorProto.UNDEFINED)
    return output_shapes, output_dtypes
