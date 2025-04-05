# SPDX-License-Identifier: Apache-2.0

"""Rewrites operator einsum into simple ONNX operators.
"""
import math
from itertools import permutations
import numpy as np
from onnx import helper, numpy_helper, TensorProto, AttributeProto
from .. import utils
from ..constants import OPSET_TO_IR_VERSION, PREFERRED_OPSET
from .optimizer_base import GraphOptimizerBase


class OnnxMicroRuntime:
    """
    Implements a micro runtime for ONNX graphs.
    It does not implements all the operator types.
    This runtime is used to infer shape. `shape_inference`
    from `onnx` does not return all shapes when the onnx graph
    includes an operator *Reshape*.

    :param model_onnx: ONNX model
    """

    def __init__(self, model_onnx):
        if not hasattr(model_onnx, 'graph'):
            raise TypeError(
                "model_onnx is not an ONNX graph but %r." % type(model_onnx))
        self.model_onnx = model_onnx

    def run(self, inputs):
        """
        Computes the outputs of the graph.

        :param inputs: dictionary
        :return: all intermediates results and output as a dictionary
        """
        def _get_dtype(onnx_type):
            if onnx_type == 1:
                return np.float32
            if onnx_type == 7:
                return np.int64
            raise ValueError("Unable to guess dtype from ONNX type %r." % onnx_type)

        def _extract_numpy_array(v):
            return np.frombuffer(v.raw_data, dtype=_get_dtype(v.data_type))

        if not isinstance(inputs, dict):
            raise TypeError(
                "inputs must be a dictionary not %r." % type(inputs))
        results = inputs.copy()

        for init in self.model_onnx.graph.initializer:
            name = init.name
            mat = _extract_numpy_array(init)
            results[name] = mat

        for node in self.model_onnx.graph.node:
            op_type = node.op_type
            inp = [results[n] for n in node.input]
            meth_name = "_op_%s" % op_type.lower()
            if not hasattr(self, meth_name):
                raise NotImplementedError(
                    "OnnxMicroRuntime does not implement operator %r." % op_type)
            kwargs = {}
            for at in node.attribute:
                kwargs[at.name] = at
            out = getattr(self, meth_name)(*inp, **kwargs)
            for n, o in zip(node.output, out):
                results[n] = o

        return results

    def _op_add(self, x, y):
        "Runtime for operator."
        return (x + y,)

    def _op_concat(self, *args, axis=None):
        "Runtime for operator."
        if axis is not None:
            axis = axis.i

        def _preprocess(a, axis):
            if axis >= len(a.shape):
                new_shape = a.shape + (1,) * (axis + 1 - len(a.shape))
                return a.reshape(new_shape)
            return a

        targs = tuple(_preprocess(a, axis) for a in args)
        return (np.concatenate(targs, axis),)

    def _op_gemm(self, a, b, c=None, alpha=None, beta=None,  # pylint: disable=C0103
                 transA=None, transB=None):  # pylint: disable=C0103
        "Runtime for operator."
        if alpha is not None:
            alpha = alpha.f
        if beta is not None:
            beta = beta.f
        if transA is None:
            transA = False
        else:
            transA = transA.i
        if transB is None:
            transB = False
        else:
            transB = transB.i

        def _gemm00(a, b, c, alpha, beta):
            o = np.dot(a, b) * alpha
            if beta != 0:
                o += c * beta
            return o

        def _gemm01(a, b, c, alpha, beta):
            o = np.dot(a, b.T) * alpha
            if beta != 0:
                o += c * beta
            return o

        def _gemm10(a, b, c, alpha, beta):
            o = np.dot(a.T, b) * alpha
            if beta != 0:
                o += c * beta
            return o

        def _gemm11(a, b, c, alpha, beta):
            o = np.dot(a.T, b.T) * alpha
            if beta != 0:
                o += c * beta
            return o

        if transA:
            fct = _gemm11 if transB else _gemm10
        else:
            fct = _gemm01 if transB else _gemm00
        return (fct(a, b, c, alpha=alpha, beta=beta),)

    def _op_gather(self, x, indices, axis=None):
        "Runtime for operator."
        if not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x)
        if not indices.flags['C_CONTIGUOUS']:
            indices = indices.ascontiguousarray()
        if axis is not None:
            axis = axis.i
        return (np.take(x, indices, axis=axis),)

    def _op_identity(self, x):
        "Runtime for operator."
        return (x,)

    def _op_matmul(self, x, y):
        "Runtime for operator."
        return (np.matmul(x, y),)

    def _op_max(self, *x):
        "Runtime for operator."
        return (np.maximum(*x),)  #pylint: disable=E1120

    def _op_mul(self, x, y):
        "Runtime for operator."
        return (x * y,)

    def _op_reduceprod(self, data, axes=None, keepdims=None):
        "Runtime for operator :epkg:`Op:ReduceProd`."
        if keepdims is not None:
            keepdims = keepdims.i
        if axes is not None and not isinstance(axes, int):
            if isinstance(axes, np.ndarray) and len(axes.shape) == 0:
                axes = int(axes)
            else:
                axes = tuple(axes) if len(axes) > 0 else None
        return (np.prod(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)

    def _op_reducesum(self, data, axes, keepdims=None, noop_with_empty_axes=None):
        "Runtime for operator."
        if keepdims is not None:
            keepdims = keepdims.i
        if noop_with_empty_axes is not None:
            noop_with_empty_axes = noop_with_empty_axes.i
        if axes is None and self.noop_with_empty_axes is not None:
            return (data,)
        if axes is not None and not isinstance(axes, int):
            if isinstance(axes, np.ndarray) and len(axes.shape) == 0:
                axes = int(axes)
            else:
                axes = tuple(axes) if len(axes) > 0 else None
        return (np.sum(data, axis=axes, keepdims=keepdims, dtype=data.dtype),)

    def _op_reshape(self, x, shape):
        "Runtime for operator."
        return (x.reshape(shape),)

    def _op_shape(self, x):
        "Runtime for operator."
        return (np.array(list(x.shape), dtype=np.int64),)

    def _op_squeeze(self, x, axes=None):
        "Runtime for operator."
        if axes is None:
            return (x,)
        if isinstance(axes, AttributeProto):
            axes = list(axes.ints)
        if hasattr(axes, '__iter__'):
            return (np.squeeze(x, axis=tuple(axes)),)
        return (np.squeeze(x, axis=axes),)

    def _op_transpose(self, x, perm=None):
        "Runtime for operator."
        if perm is not None:
            perm = tuple(perm.ints)
        return (np.transpose(x, perm),)

    def _op_unsqueeze(self, x, axes=None):
        "Runtime for operator."
        if axes is None:
            return (x,)
        if isinstance(axes, AttributeProto):
            axes = list(axes.ints)
        if hasattr(axes, '__iter__'):
            return (np.expand_dims(x, axis=tuple(axes)),)
        return (np.expand_dims(x, axis=axes),)


def single_axes(axes):
    """
    *axes* contains positive values, then it is the position
    of this axis in the original matrix, otherwise it is -1
    meaning this axis is an added single dimension to align
    all the dimensions based on the einsum equation.

    :param axes: axes described above
    :return: list of integer in set `{1, 2}`, 1 for
        a single axis, 2 otherwise
    """
    if axes is None:
        return axes
    return [(1 if a == -1 else 2) for a in axes]


class EinsumSubOp:
    """
    Defines a sub operation used in Einsum decomposition.

    :param name: name (reshape, transpose, reduce_sum, matmul, id,
        squeeze, diagonal, mul, batch_dot)
    :param inputs: inputs
    :param kwargs: arguments

    Operator suffixed by `_mm` (*transpose_mm*, *reduce_sum_mm*)
    are equivalent to the same operator without the suffix
    but takes two inputs and only changes the first one.

    Attributes `_info` summarizes the known information
    about dimensions. Many of them are empty because inserted.
    Value `1` means it was the case, `2` means it is a plain dimension.
    """
    _allowed = {'expand_dims', 'transpose', 'reduce_sum', 'matmul', 'id',
                'squeeze', 'diagonal', 'mul', 'batch_dot',
                'transpose_mm', 'reduce_sum_mm'}

    def __init__(self, full_dim, name, *inputs, **kwargs):
        self.full_dim = full_dim
        self.name = name
        self.inputs = inputs
        self.kwargs = kwargs
        self._info = {}
        if name not in EinsumSubOp._allowed:
            raise ValueError(
                "Unexpected name %r. It should be in %r."
                "" % (name, EinsumSubOp._allowed))
        if len(inputs) not in (1, 2):
            raise RuntimeError(
                "Inputs must contains 1 or 2 inputs not %d." % len(inputs))
        if name == 'matmul' and len(inputs) != 2:
            raise RuntimeError(
                "Inputs must contains 2 inputs not %d for operator 'matmul'."
                "" % len(inputs))
        for i, inp in enumerate(inputs):
            if not isinstance(inp, (int, EinsumSubOp)):
                raise TypeError(
                    "Input %d has type %r, int or EinsumSubOp is expected."
                    "" % (i, type(inp)))
        self._check_()

    def _check_(self):
        "Checks for wrong values."
        if self.name == 'transpose':
            self._check_arg_('perm', tuple)
            perm = self.kwargs['perm']
            if len(perm) != len(set(perm)):
                raise RuntimeError(
                    "perm has duplicated values %r (name=%r)."
                    "" % (perm, self.name))
            if list(perm) == list(range(len(perm))):
                raise ValueError(
                    "Transpose = identity perm=%r. It must be removed."
                    "" % perm)
        elif self.name == 'matmul':
            self._check_arg_('axes', tuple)
            self._check_arg_('left', tuple)
            self._check_arg_('right', tuple)
            axes = self.kwargs['axes']
            left = self.kwargs['left']
            right = self.kwargs['right']
            for a in axes:
                if a in left and a in right:
                    raise RuntimeError(
                        "One axis belongs to every set (axes, left, right). "
                        "axes=%r, left=%r, right=%r." % (axes, left, right))

    def __repr__(self):
        inps = ", ".join(map(str, self.inputs))
        kw = ", ".join("%s=%r" % (k, w) for k, w in self.kwargs.items())
        m = "%s(%r, %s, %s)" % (
            self.__class__.__name__, self.name, inps, kw)
        return m

    def _check_arg_(self, name, typ, empty=False):
        if name not in self.kwargs:
            raise RuntimeError(
                "Parameter %r not found for operator %r." % (name, self.name))
        if empty and self.kwargs[name] is None:
            return
        if not isinstance(self.kwargs[name], typ):
            raise TypeError(
                "Unexpected type %r for parameter %r and parameter %r."
                "" % (type(self.kwargs[name]), name, self.name))

    def _check_row_(self, row, inp=False):
        """
        Checks input or output is valid.
        """

    def _compute_output_row_id(self, row, row2=None, ab=False):
        "compute shape after operator id"
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True)
        row[:] = row2[:]
        self._check_row_(row)

    def _compute_output_row_transpose(self, row, row2=None, ab=False):
        "compute shape after operator transpose"
        if ab:
            self._compute_output_row_transpose(row2)
            return
        self._check_row_(row, True)
        self._check_arg_('perm', tuple)
        if len(self.kwargs['perm']) != len(row):
            raise RuntimeError(
                "Unexpected permutation %r (row=%r)."
                "" % (self.kwargs['perm'], row))
        perm = self.kwargs['perm']
        cpy = row.copy()
        for i, p in enumerate(perm):
            row[i] = cpy[p]
        self._check_row_(row)

    def _compute_output_row_transpose_mm(self, row, row2=None, ab=False):
        "compute shape after operator transpose"
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True)
        if row2 is None:
            raise RuntimeError("transpose_mm expects a second input.")
        self._compute_output_row_transpose(row, row2=None)

    def _compute_output_row_expand_dims(self, row, row2=None, ab=False):
        "compute shape after operator expand_dims"
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True)
        self._check_arg_('axes', tuple)
        axes = self.kwargs['axes']
        for axis in axes:
            if not isinstance(axis, tuple):
                raise TypeError(
                    "Parameter axes of expand_dims should be a tuple of "
                    "tuple, axes=%r." % axes)
            if row[axis[1]] != -1:
                raise RuntimeError(
                    "Dimension should be -1 in row %r axis=%r." % (
                        row, self.kwargs['axis']))
        self._check_row_(row)

    def _compute_output_row_reduce_sum(self, row, row2=None, ab=False):
        "compute shape after operator reduce_sum"
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True)
        self._check_arg_('axes', tuple)
        for a in self.kwargs['axes']:
            row[a] = -1
        self._check_row_(row)

    def _compute_output_row_reduce_sum_mm(self, row, row2=None, ab=False):
        "compute shape after operator reduce_sum"
        if not ab:
            raise RuntimeError("ab must be true.")
        self._check_row_(row2, True)
        if row2 is None:
            raise RuntimeError("reduce_sum_mm expects a second input.")
        self._compute_output_row_reduce_sum(row, row2=None)

    def _compute_output_row_squeeze(self, row, row2=None, ab=False):
        "compute shape after operator squeeze"
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True)
        self._check_arg_('axes', tuple)
        for a in self.kwargs['axes']:
            row[a] = -1
        self._check_row_(row)

    def _compute_output_row_diagonal(self, row, row2=None, ab=False):
        "compute shape after operator diagonal"
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True)
        self._check_arg_('diag', list)
        to_remove = []
        for choice, choices in self.kwargs['diag']:
            for ch in choices:
                if ch != choice:
                    to_remove.append(ch)
            for i in range(len(row)):  # pylint: disable=C0200
                if row[i] in choices:
                    if row[i] != choice:
                        row[i] = choice
        to_remove.sort()
        for r in to_remove:
            for i in range(len(row)):  # pylint: disable=C0200
                if row[i] == r:
                    raise RuntimeError(
                        "Unexpected result r=%r row=%r to_remove=%r "
                        "diag=%r." % (
                            r, row, to_remove, self.kwargs['diag']))
                if row[i] > r:
                    row[i] -= 1
        self._check_row_(row)

    def _compute_output_row_matmul(self, row, row2=None, ab=False):
        "compute shape after operator matmul"
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True)
        self._check_row_(row2, True)
        self._check_arg_('axes', tuple)
        self._check_arg_('left', tuple)
        self._check_arg_('right', tuple)
        self._check_arg_('ndim', int)
        if row2 is None:
            raise RuntimeError("matmul expects two inputs.")
        row2[:] = np.maximum(row, row2)
        for a in self.kwargs['axes']:
            if a not in self.kwargs['right']:
                row2[a] = -1
        self._check_row_(row2)

    def _compute_output_row_batch_dot(self, row, row2=None, ab=False):
        "compute shape after operator batch_dot"
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True)
        self._check_row_(row2, True)
        self._check_arg_('batch_axes', tuple)
        self._check_arg_('keep_axes', tuple, empty=True)
        self._check_arg_('sum_axes', tuple)
        self._check_arg_('left', tuple)
        self._check_arg_('right', tuple)
        self._check_arg_('ndim', int)
        if row2 is None:
            raise RuntimeError("batch_dot expects two inputs.")
        row2[:] = np.maximum(row, row2)
        for a in self.kwargs['sum_axes']:
            if a not in self.kwargs['right']:
                row2[a] = -1
        self._check_row_(row2)

    def _compute_output_row_mul(self, row, row2=None, ab=False):
        "compute shape after operator mul"
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True)
        self._check_row_(row2, True)
        if row2 is None:
            raise RuntimeError("mul expects two inputs.")
        row2[:] = np.maximum(row, row2)
        self._check_row_(row2)

    def compute_output_row(self, row, row2=None, ab=False):
        """
        Updates *row* based on the operator.
        """
        method_name = "_compute_output_row_%s" % self.name
        meth = getattr(self, method_name, None)
        if meth is None:
            raise NotImplementedError(
                "compute_output_row not implemented for %r." % self.name)
        self.add_info(i_row=single_axes(row), i_row2=single_axes(row2))
        meth(row, row2=row2, ab=ab)
        self.add_info(o_row=single_axes(row), o_row2=single_axes(row2))

    def add_info(self, **kwargs):
        """
        Adds information to the node.

        :param kwargs: dictionary
        """
        for k, v in kwargs.items():
            if k in self._info:
                raise KeyError(
                    "Key %r already added (operator %r)." % (k, self.name))
            self._info[k] = v

    def _check_inputs_(self, n_expected, check_dim=False):
        if len(self.inputs) != n_expected:
            raise RuntimeError(
                "Number of inputs must be %d not %d for operator %r."
                "" % (n_expected, len(self.inputs), self.name))

    def _check_shape_(self, m):
        if len(m.shape) != self.full_dim:
            raise RuntimeError(
                "Number of dimensions %r is different from expected value "
                "%d." % (m.shape, self.full_dim))

    def _get_data(self, data, key, as_str=False):
        "Returns data[key] or raises an exception if not here."
        if isinstance(key, int):
            if key not in data:
                raise RuntimeError(
                    "Unable to find key %d in %r." % (
                        key, list(sorted(data))))
            value = data[key]
        elif isinstance(key, EinsumSubOp):
            if id(key) not in data:
                raise RuntimeError(
                    "Unable to find key %d in %r." % (
                        id(key), list(sorted(data))))
            value = data[id(key)]
        else:
            raise TypeError(
                "Unexpected input type %r." % type(key))
        if as_str:
            if isinstance(value, str):
                return value
            if hasattr(value, 'output') and len(value.output) == 1:
                return value.output[0]
            if hasattr(value, list) and len(value) == 1:
                return value[0]
            raise RuntimeError(
                "Unable to guess what to return in that case %r - %r"
                "." % (type(value), value))
        return value

    def _onnx_name(self):
        return 'einsum%d_%s' % (id(self), self.name[:2])

    def _check_onnx_opset_(self, opset, limit):
        if opset is not None and opset < limit:
            raise RuntimeError(
                "Opset (%r) must be >= %r for operator %r."
                "" % (opset, limit, self.name))

    def _to_onnx_id(self, names, opset):  # pylint: disable=W0613
        self._check_inputs_(1)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        yield helper.make_node('Identity', [name], [self._onnx_name()])

    def _to_onnx_expand_dims(self, names, opset):
        "insert node unsqueeze"
        self._check_inputs_(1)
        self._check_onnx_opset_(opset, 11)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        axes = self.kwargs['axes']
        name_axes = name + '_axes'
        if opset >= 13:
            yield numpy_helper.from_array(
                np.array([a[1] for a in axes], dtype=np.int64), name=name_axes)
            yield helper.make_node(
                'Unsqueeze', [name, name_axes], [self._onnx_name()])
        else:
            yield helper.make_node(
                'Unsqueeze', [name], [self._onnx_name()], axes=[a[1] for a in axes])

    def _to_onnx_squeeze(self, names, opset):
        "insert node squeeze"
        self._check_inputs_(1)
        self._check_onnx_opset_(opset, 11)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        axes = self.kwargs['axes']
        name_axes = name + '_axes'
        if opset >= 13:
            yield numpy_helper.from_array(
                np.array(axes, dtype=np.int64), name=name_axes)
            yield helper.make_node(
                'Squeeze', [name, name_axes], [self._onnx_name()])
        else:
            yield helper.make_node(
                'Squeeze', [name], [self._onnx_name()], axes=axes)

    def _to_onnx_transpose(self, names, opset):  # pylint: disable=W0613
        self._check_inputs_(1)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        perm = self.kwargs['perm']
        yield helper.make_node(
            'Transpose', [name], [self._onnx_name()], perm=perm)

    def _to_onnx_reduce_sum(self, names, opset):
        self._check_inputs_(1)
        self._check_onnx_opset_(opset, 11)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        axes = self.kwargs['axes']
        name_axes = self._onnx_name() + '_axes'
        yield numpy_helper.from_array(
            np.array(axes, dtype=np.int64), name=name_axes)
        yield helper.make_node(
            'ReduceSum', [name, name_axes], [self._onnx_name()], keepdims=1)

    def _to_onnx_mul(self, data, opset):
        self._check_inputs_(2)
        self._check_onnx_opset_(opset, 1)
        inp1 = self.inputs[0]
        inp2 = self.inputs[1]
        m1 = self._get_data(data, inp1)
        m2 = self._get_data(data, inp2)
        yield helper.make_node('Mul', [m1, m2], [self._onnx_name()])

    def _to_onnx_batch_dot(self, names, opset):
        "convert the graph to dot"
        self._check_inputs_(2)
        self._check_onnx_opset_(opset, 13)
        inp1, inp2 = self.inputs[:2]
        name1 = self._get_data(names, inp1)
        name2 = self._get_data(names, inp2)

        batch_axes = self.kwargs['batch_axes']
        keep_axes = self.kwargs['keep_axes']
        sum_axes = self.kwargs['sum_axes']
        left = self.kwargs['left']
        right = self.kwargs['right']
        root = self._onnx_name()

        def return_name_one():
            name_one = root + "_1"
            return name_one, numpy_helper.from_array(
                np.array([1], dtype=np.int64), name=name_one)

        name_one = None
        name_shape1 = root + "_shape1"
        name_shape2 = root + "_shape2"
        concat_left = []
        concat_right = []
        yield helper.make_node('Shape', [name1], [name_shape1])
        yield helper.make_node('Shape', [name2], [name_shape2])

        if len(batch_axes) > 0:
            name_batch_axes = root + "_batch_axes"
            yield numpy_helper.from_array(
                np.array(batch_axes, dtype=np.int64), name=name_batch_axes)

        if len(sum_axes) > 0:
            name_sum_axes = root + "_sum_axes"
            yield numpy_helper.from_array(
                np.array(sum_axes, dtype=np.int64), name=name_sum_axes)

        # dim0 = int(np.prod([m1.shape[i] for i in batch_axes]))
        # dim0b = int(np.prod([m2.shape[i] for i in batch_axes]))
        if len(batch_axes) > 1:
            name_dim0 = root + "_dim0"
            name_dim0b = root + "_dim0b"
            name_dim0g = name_dim0 + 'g'
            name_dim0bg = name_dim0b + 'g'
            concat_left.append(name_dim0)
            concat_right.append(name_dim0b)
            yield helper.make_node(
                'Gather', [name_shape1, name_batch_axes], [name_dim0g])
            yield helper.make_node(
                'Gather', [name_shape2, name_batch_axes], [name_dim0bg])
            yield helper.make_node(
                'ReduceProd', [name_dim0g], [name_dim0], keepdims=1)
            yield helper.make_node(
                'ReduceProd', [name_dim0bg], [name_dim0b], keepdims=1)
        elif len(batch_axes) == 1:
            name_dim0g = root + "_dim0g"
            name_dim0bg = root + "_dim0bg"
            name_dim0 = name_dim0g
            name_dim0b = name_dim0bg
            concat_left.append(name_dim0)
            concat_right.append(name_dim0b)
            yield helper.make_node(
                'Gather', [name_shape1, name_batch_axes], [name_dim0g])
            yield helper.make_node(
                'Gather', [name_shape2, name_batch_axes], [name_dim0bg])
        else:
            if name_one is None:
                name_one, cst_init = return_name_one()
                yield cst_init
            name_dim0 = name_one
            name_dim0b = name_one
            concat_left.append(name_dim0)
            concat_right.append(name_dim0b)

        # dimb = int(-1 if keep_axes is None else np.prod(
        #     [m1.shape[i] for i in keep_axes]))
        if keep_axes in (-1, None) or len(keep_axes) == 0:
            name_dimb = root + "__1"
            concat_left.append(name_dimb)
            concat_right.append(name_dimb)
            yield numpy_helper.from_array(
                np.array([-1], dtype=np.int64), name=name_dimb)
        elif len(keep_axes) == 1:
            name_keep_axes = root + "_keep_axes"
            name_dimb = root + "_dimb"
            name_dimbg = name_dimb
            concat_left.append(name_dimb)
            concat_right.append(name_dimb)
            yield numpy_helper.from_array(
                np.array(keep_axes, dtype=np.int64), name=name_keep_axes)
            yield helper.make_node(
                'Gather', [name_shape1, name_keep_axes], [name_dimbg])
        else:
            name_keep_axes = root + "_keep_axes"
            name_dimb = root + "_dimb"
            name_dimbg = name_dimb + 'g'
            concat_left.append(name_dimb)
            concat_right.append(name_dimb)
            yield numpy_helper.from_array(
                np.array(keep_axes, dtype=np.int64), name=name_keep_axes)
            yield helper.make_node(
                'Gather', [name_shape1, name_keep_axes], [name_dimbg])
            yield helper.make_node(
                'ReduceProd', [name_dimbg], [name_dimb], keepdims=1)

        # dim1 = int(np.prod([m1.shape[i] for i in sum_axes]))
        # dim2 = int(np.prod([m2.shape[i] for i in sum_axes]))

        if len(sum_axes) == 0:
            if name_one is None:
                name_one, cst_init = return_name_one()
                yield cst_init
            name_dim1 = name_one
            name_dim2 = name_one
            concat_left.append(name_dim1)
            concat_right.append(name_dim2)
        elif len(sum_axes) == 1:
            name_dim1 = root + "_dim1"
            name_dim2 = root + "_dim2"
            name_dim1g = name_dim1
            name_dim2g = name_dim2
            concat_left.append(name_dim1)
            concat_right.append(name_dim2)
            yield helper.make_node(
                'Gather', [name_shape1, name_sum_axes], [name_dim1g])
            yield helper.make_node(
                'Gather', [name_shape2, name_sum_axes], [name_dim2g])
        else:
            name_dim1 = root + "_dim1"
            name_dim2 = root + "_dim2"
            name_dim1g = name_dim1 + 'g'
            name_dim2g = name_dim2 + 'g'
            concat_left.append(name_dim1)
            concat_right.append(name_dim2)
            yield helper.make_node(
                'Gather', [name_shape1, name_sum_axes], [name_dim1g])
            yield helper.make_node(
                'Gather', [name_shape2, name_sum_axes], [name_dim2g])
            yield helper.make_node(
                'ReduceProd', [name_dim1g], [name_dim1], keepdims=1)
            yield helper.make_node(
                'ReduceProd', [name_dim2g], [name_dim2], keepdims=1)

        batch_kind = self.get_dot_kind()
        if batch_kind in ('11', 'N1', 'N1'):
            # *shape1, *shape2
            name_minus_one = root + "__01"
            yield numpy_helper.from_array(
                np.array([-1], dtype=np.int64), name=name_minus_one)
            name_agg_shape1_2 = root + "_resh1_%s" % batch_kind
            name_agg_shape2_2 = root + "_resh2_%s" % batch_kind
            yield helper.make_node(
                'Concat', [name_minus_one, name_dim1], [name_agg_shape1_2], axis=0)
            yield helper.make_node(
                'Concat', [name_minus_one, name_dim2], [name_agg_shape2_2], axis=0)

            # m1sh = m1.reshape((-1, dim1))
            # m2sh = m2.reshape((-1, dim2))
            name_agg1_2 = root + "_aresh1"
            name_agg2_2 = root + "_aresh2"
            yield helper.make_node('Reshape', [name1, name_agg_shape1_2], [name_agg1_2])
            yield helper.make_node('Reshape', [name2, name_agg_shape2_2], [name_agg2_2])

            # dot = gemm(m1sh, m2sh, False, True)
            name_dot = root + "_gemm"
            yield helper.make_node(
                'Gemm', [name_agg1_2, name_agg2_2], [name_dot],
                alpha=1., beta=0., transA=0, transB=1)
        else:
            # *shape1, *shape2
            name_agg_shape1 = root + "_resh1"
            name_agg_shape2 = root + "_resh2"
            yield helper.make_node(
                'Concat', concat_left, [name_agg_shape1], axis=0)
            yield helper.make_node(
                'Concat', concat_right, [name_agg_shape2], axis=0)

            # m1sh = m1.reshape((dim0, dimb, dim1))
            # m2sh = m2.reshape((dim0b, dimb, dim2))
            name_agg1 = root + "_aresh1"
            name_agg2 = root + "_aresh2"
            yield helper.make_node('Reshape', [name1, name_agg_shape1], [name_agg1])
            yield helper.make_node('Reshape', [name2, name_agg_shape2], [name_agg2])

            # dot = m1sh @ np.transpose(m2sh, (0, 2, 1))
            name_agg2_tr = root + "_aresh2_tr"
            yield helper.make_node(
                'Transpose', [name_agg2], [name_agg2_tr], perm=[0, 2, 1])

            name_dot = root + "_dot"
            yield helper.make_node(
                'MatMul', [name_agg1, name_agg2_tr], [name_dot])

        # new_shape = ([max(m1.shape[i], m2.shape[i]) for i in batch_axes] +
        #      [m1.shape[i] for i in left if i not in batch_axes] +
        #      [m2.shape[i] for i in right if i not in batch_axes])
        concat_final = []
        if len(batch_axes) > 0:
            name_max_dim = root + "_max_dim"
            concat_final.append(name_max_dim)
            yield helper.make_node(
                'Max', [name_dim0g, name_dim0bg], [name_max_dim])

        left_set = list(sorted(set(left) - (set(batch_axes) & set(left))))
        if len(left_set) > 0:
            name_left_dim = root + "_left_dim"
            name_left_set = root + "_left_set"
            yield numpy_helper.from_array(
                np.array(left_set, dtype=np.int64), name=name_left_set)
            yield helper.make_node(
                'Gather', [name_shape1, name_left_set], [name_left_dim])
            concat_final.append(name_left_dim)

        right_set = list(sorted(set(right) - (set(batch_axes) & set(right))))
        if len(right_set) > 0:
            name_right_dim = root + "_right_dim"
            name_right_set = root + "_right_set"
            yield numpy_helper.from_array(
                np.array(right_set, dtype=np.int64), name=name_right_set)
            yield helper.make_node(
                'Gather', [name_shape2, name_right_set], [name_right_dim])
            concat_final.append(name_right_dim)

        name_new_shape = root + '_new_shape'
        diff = (
            self.full_dim -
            (len(batch_axes) + len(left_set) + len(right_set)))
        if diff > 0:
            names_ones = root + "_ones"
            yield numpy_helper.from_array(
                np.array([1 for i in range(diff)], dtype=np.int64),
                name=names_ones)
            concat_final.append(names_ones)

        yield helper.make_node(
            'Concat', concat_final, [name_new_shape], axis=0)

        name_final = root + '_final'
        yield helper.make_node(
            'Reshape', [name_dot, name_new_shape], [name_final])

    def to_onnx(self, names, opset=None, **kwargs):
        """
        Converts this node into ONNX. Enumerates all ONNX node
        which participate to the conversion. The last one
        is the final output.

        :param names: dictionary where to find already converted operators
        :param opset: opset
        :param kwargs: additional parameter for the conversion
        :return: output
        """
        if opset is None:
            opset = PREFERRED_OPSET
        method_name = "_to_onnx_%s" % self.name
        meth = getattr(self, method_name, None)
        if meth is None:
            if self.name.endswith("_mm"):
                raise NotImplementedError(
                    "to_onnx not implemented for %r."
                    "You should call method simplify_mm_nodes "
                    "to remove it." % self.name)
            raise NotImplementedError(
                "to_onnx not implemented for %r." % self.name)
        for ni, node in enumerate(meth(names, opset=opset, **kwargs)):
            if hasattr(node, 'output'):
                names[id(self)] = node.output[0]
                node.name = "OPT%s_%d_%d" % (method_name, ni, id(self))
            yield node

    def get_dot_kind(self):
        """
        Every matrix multiplication can be either:
        * a simple multiplication (`M`) (undetected)
        * a 2D matrix multiplication (`11`)
        * a broadcasted matrix multiplication (`N1` or `1N`)
        * a batch matrix multiplication (`NN`)
        This method returns which kind it is.
        """
        batch_axes = self.kwargs['batch_axes']
        # keep_axes = self.kwargs['keep_axes']
        # sum_axes = self.kwargs['sum_axes']
        # left = self.kwargs['left']
        # right = self.kwargs['right']
        info = self._info
        row_left = info['i_row']
        row_right = info['i_row2']

        batch_left = [row_left[k] for k in batch_axes]
        batch_right = [row_right[k] for k in batch_axes]
        n_left = len(batch_left) > 0 and max(batch_left) == 2
        n_right = len(batch_right) > 0 and max(batch_right) == 2
        return "%s%s" % ('N' if n_left else '1', 'N' if n_right else '1')


class GraphEinsumSubOp:
    """
    Class gathering all nodes produced to explicit einsum
    operators.

    :param letters: list of distinct letters
    :param mat: matrix, see *analyse_einsum_equation*
    :param lengths: lengths of every input
    :param duplicates: see *analyse_einsum_equation*
    """

    def __init__(self, letters, mat, lengths, duplicates):
        self._nodes = {}
        self._mark = {}
        self._ops = []
        self._inputs = {}
        self.last_op = None
        self.last_added_op = None
        self.metadata = dict(
            letters=letters, mat=mat, lengths=lengths,
            mat0=mat.copy(), duplicates=duplicates)

    def append(self, op):
        """
        Adds one input or result.

        :param op: integer (an input) or an instance of class *EinsumSubOp*.
        :return: op or None if op is an integer
        """
        if isinstance(op, int):
            if op in self._nodes:
                raise RuntimeError("Key %d already added." % op)
            self._nodes[op] = op
            self.last_added_op = op
            self._inputs[op] = op
            return None
        if isinstance(op, EinsumSubOp):
            if op in self._nodes:
                raise RuntimeError(
                    "Key %d already added, op=%r." % (id(op), op))
            self._nodes[id(op)] = op
            self._ops.append(op)
            self.last_added_op = op
            return op
        raise TypeError("Unexpected type %r." % type(op))

    def mark_last_node(self):
        """
        Marks the last node as the final output.
        """
        if self.last_added_op is None:
            raise RuntimeError("last_added_op is None.")
        self.mark(-1, self.last_added_op)

    def mark(self, i, op):
        """
        Marks one input or result as an intermediate result
        after a full einsum step.

        :param op: integer (an input) or an instance of class *EinsumSubOp*.
        """
        if not isinstance(i, int):
            raise TypeError("i must an integer not %r." % type(i))
        if i != -1 and i not in self._inputs:
            raise RuntimeError(
                "Input %d was not registered in %r." % (i, self._inputs))
        if isinstance(op, EinsumSubOp):
            if id(op) not in self._nodes:
                raise RuntimeError(
                    "Key %d not found, op=%r." % (id(op), op))
            self._mark[i] = op
            self._mark[id(op)] = i
            self.last_op = op
        else:
            raise TypeError("Unexpected type %r." % type(i))

    def __iter__(self):
        "Iterates on nodes."
        for op in self._ops:
            yield op

    def to_dot(self, **kwargs):
        """
        Produces a graph in *dot*.

        :param kwargs: additional graph option
        :return: string
        """
        options = {
            'orientation': 'portrait',
            'ranksep': '0.25',
            'nodesep': '0.05',
            'width': '0.5',
            'height': '0.1',
            'size': '5',
            'node': '[shape=record]',
        }
        options.update(kwargs)

        def d2s(d):
            it = []
            for k, v in sorted(d.items()):
                it.append("%s=%s" % (k, v))
            return " ".join(it)

        def d2sd(d):
            it = []
            for k, v in sorted(d.items()):
                if len(v) > 1:
                    it.append("%s=%s" % (k, ",".join(map(str, v))))
            return " ".join(it)

        rows = ["digraph{"]
        for k, v in options.items():
            if isinstance(v, str) and "[" in v:
                rows.append("{} {};".format(k, v))
            else:
                rows.append("{}={};".format(k, v))
        for k, v in self._nodes.items():
            if isinstance(v, int):
                let = [(r, self.metadata['letters'][i])
                       for i, r in enumerate(self.metadata['mat0'][v])
                       if r != -1]
                dup = self.metadata['duplicates'][v]
                if dup is None:
                    dup = ""
                else:
                    dup = " - %s" % d2sd(dup)
                let.sort()
                letters = "".join(_[1] for _ in let)
                lab = "input %d\\\\n%s\\\\n%s%s" % (
                    v, letters, str(self.metadata['mat0'][v]), dup)
                sk = v
            else:
                lab = "%s\\\\n%s" % (v.name, d2s(v.kwargs))
                sk = id(v)

            if sk in self._mark and isinstance(self._mark[sk], int):
                la = self._mark[sk]
                lab = lab.replace("\\\\n", " - I%d\\\\n" % la)
                s = ('%d [label="%s" style=filled fillcolor=red];' % (k, lab))
            else:
                s = '%d [label="%s"];' % (k, lab)
            rows.append(s)
            if not hasattr(v, 'inputs'):
                continue
            for i in v.inputs:
                vid = i if isinstance(i, int) else id(i)
                s = "%d -> %d;" % (vid, k)
                rows.append(s)
        rows.append("}")
        return "\n".join(rows)

    def clean_unused_nodes(self):
        """
        Cleans nodes with unused outputs.
        """

        def iteration(_):
            # Walks through all nodes.
            is_used = {}
            for node in self._ops:
                if not isinstance(node, EinsumSubOp):
                    continue
                if id(node) not in is_used:
                    is_used[id(node)] = []
                for inp in node.inputs:
                    if not isinstance(inp, EinsumSubOp):
                        continue
                    idn = id(inp)
                    if idn not in is_used:
                        is_used[idn] = []
                    is_used[idn].append(id(node))

            # Remove unused nodes.
            removed = []
            for k, v in is_used.items():
                if len(v) == 0:
                    removed.append(k)
            removed = set(removed)
            i_rem = []
            for i, op in enumerate(self._ops):
                if not isinstance(op, EinsumSubOp):
                    continue
                if id(op) in removed and id(op) not in self._mark:
                    i_rem.append((i, id(op)))
            for i, idn in reversed(i_rem):
                del self._ops[i]
                del self._nodes[idn]
            return len(i_rem) > 0

        it = 1
        while iteration(it):
            it += 1

        self.last_op = None
        self.last_added_op = None

    def simplify_mm_nodes(self):
        """
        Node name suffixed by `mm` are an artifact to keep
        the graph consistent while building it. They can
        now be replaced by the equivalent node without suffix `mm`.
        """
        for op in self:
            if not isinstance(op, EinsumSubOp):
                continue
            if op.name.endswith('_mm'):
                if len(op.inputs) != 2:
                    raise RuntimeError(
                        "Expecting 2 inputs for node %r not %r id=%r." % (
                            op.name, len(op.inputs), id(op)))
                op.name = op.name[:-3]
                op.inputs = op.inputs[:1]

    def _get_forward_nodes(self):
        """
        Returns the forward nodes.
        """
        forward = {}
        for op in self:
            if isinstance(op, int):
                continue
            for inp in op.inputs:
                key = inp if isinstance(inp, int) else id(inp)
                if key in forward:
                    forward[key].append(op)
                else:
                    forward[key] = [op]
        return forward

    def _replace_node_sequence(self, added, deleted):
        """
        Removes a sequence of nodes. The method does not check
        that the graph remains consistent.
        """
        forward = self._get_forward_nodes()
        key = id(deleted[-1])
        if key not in forward:
            raise RuntimeError(
                "key %r missing in all forward nodes." % key)

        # deletion
        mark_input = None
        for d in deleted:
            del self._nodes[id(d)]
            if id(d) in self._mark:
                del self._mark[id(d)]
                dels = []
                for k, v in self._mark.items():
                    if id(v) == id(d):
                        mark_input = k
                        dels.append(k)
                if len(dels) != 1:
                    raise RuntimeError(
                        "Input %d has more than one marked operator "
                        "(%r)." % (id(d), dels))
                del self._mark[dels[0]]

        dels = set(id(o) for o in deleted)
        rem = []
        for i, op in enumerate(self._ops):
            if id(op) in dels:
                rem.append(i)
        if len(rem) != len(deleted):
            raise RuntimeError(
                "Mismatched length %r, %r, len=%r." % (
                    rem, dels, len(deleted)))
        for i in reversed(rem):
            del self._ops[i]
        self.last_add_op = None

        # insertion
        if added is not None:
            self._ops.insert(rem[0], added)
            self._nodes[id(added)] = added
            for op in forward[key]:
                new_inputs = list(op.inputs)
                for i in range(len(op.inputs)):
                    if id(op.inputs[i]) == key:
                        new_inputs[i] = added
                op.inputs = tuple(new_inputs)
            if mark_input is not None:
                self.mark(mark_input, added)
        else:
            inps = deleted[0].inputs
            if len(inps) != 1:
                raise RuntimeError(
                    "More than one input. Call another method.")
            inp = inps[0]
            for op in forward[key]:
                new_inputs = list(op.inputs)
                for i in range(len(op.inputs)):
                    if id(op.inputs[i]) == key:
                        new_inputs[i] = inp
                op.inputs = tuple(new_inputs)
            if mark_input is not None:
                self.mark(mark_input, inp)

    def remove_duplicate_transpose(self, verbose=False):
        """
        Removes consecutive transpose by merging them.
        :param verbose: display intermediate information
        """
        modif = 1
        while modif > 0:
            modif = 0
            candidates = []
            forward = self._get_forward_nodes()
            for op in self:
                if op.name == "transpose":
                    inp = op.inputs[0]
                    if (isinstance(inp, EinsumSubOp) and
                            inp.name == 'transpose' and
                            len(forward[id(inp)]) == 1):
                        candidates.append(op)

            if len(candidates) > 0:
                modif = 1
                # Not efficient to take the first one and to
                # start again but the graph should not be too big.
                cand = candidates[0]
                op2 = cand
                op1 = cand.inputs[0]
                perm1 = op1.kwargs['perm']
                perm2 = op2.kwargs['perm']
                if len(perm1) != len(perm2):
                    raise RuntimeError(
                        "Transposition should have the same length "
                        "%r, %r." % (perm1, perm2))
                perm = list(perm1)
                for i in range(len(perm)):  # pylint: disable=C0200
                    perm[i] = perm1[perm2[i]]
                if list(range(len(perm))) == perm:
                    # identity, everything needs to be removed
                    new_op = None
                else:
                    new_op = op2.__class__(
                        op2.full_dim, op2.name, op1.inputs[0],
                        perm=tuple(perm))
                self._replace_node_sequence(new_op, [op1, op2])
                if verbose:
                    print("[GraphEinsumSubOp.remove_duplicate_transpose] remove nodes %r"
                          " - id=%d,%d + %d perm1=%r perm2=%r -> perm=%r" % (
                              op2.name, id(op1), id(op2),
                              id(new_op) if new_op is not None else -1,
                              perm1, perm2, perm))

    def to_onnx(self, output, *inputs, proto_type=None, opset=None, **kwargs):
        """
        Converts the graph into ONNX.

        :param output: output name
        :param inputs: input names
        :param proto_type: type used for all operators
        :param opset: desired opset, None for the last one
        :param kwargs: additional parameter to use when building
            the ONNX graph, list of supported parameters:
            *name*, *ir_version*, *producer_name*,
            *producer_version*, *initializer*
        :return: ONNX graph
        """
        # inputs
        if opset is None:
            opset = PREFERRED_OPSET
        onx_inputs = []
        if proto_type is None:
            proto_type = TensorProto.FLOAT
        lengths = self.metadata['lengths']
        for inp, le in zip(inputs, lengths):
            onx_inputs.append(helper.make_tensor_value_info(
                inp, proto_type, [None for i in range(le)]))

        # output
        onx_output = helper.make_tensor_value_info(
            output, proto_type, [None for i in range(lengths[-1])])

        # nodes
        names = dict(enumerate(inputs))
        nodes = []
        inits = []
        if "initializer" in kwargs:
            inits.extend(kwargs['initializer'])
        for op in self:
            for onx_node in op.to_onnx(names, opset=opset):
                if hasattr(onx_node, 'output'):
                    nodes.append(onx_node)
                else:
                    inits.append(onx_node)

        # last node
        last_node = nodes[-1]
        nodes.append(helper.make_node(
            'Identity', [last_node.output[0]], [output]))

        # Builds the graph
        model = helper.make_model(
            opset_imports=[helper.make_operatorsetid('', opset)],
            ir_version=kwargs.get('ir_version', OPSET_TO_IR_VERSION.get(opset, 6)),
            producer_name=kwargs.get('producer_name', 'tensorflow-onnx'),
            producer_version=kwargs.get('producer_version', "0.0.dev"),
            graph=helper.make_graph(
                name=kwargs.get('name', 'einsum'),
                inputs=onx_inputs, outputs=[onx_output],
                initializer=inits, nodes=nodes))
        return model

    def to_tf2onnx(self, ctx, node):
        """
        Converts this node into ONNX. Enumerates all ONNX node
        which participate to the conversion. The last one
        is the final output.

        :param ctx: context
        :param node: einsum node to replace
        :return: output
        """
        onx = self.to_onnx(node.output[0], *node.input, dtype=np.float32, opset=ctx.opset)
        new_names = {k.name: v for k, v in zip(onx.graph.input, node.input)}
        for init in onx.graph.initializer:
            np_val = numpy_helper.to_array(init)
            new_init = ctx.make_const(utils.make_name(init.name), np_val)
            new_names[init.name] = new_init.name
            yield new_init
        for op in onx.graph.node:
            kwargs = {p.name: p for p in op.attribute}
            new_node = ctx.make_node(
                op.op_type, [new_names[i] for i in op.input], attr=kwargs)
            yield new_node
            new_names[op.output[0]] = new_node.output[0]


def analyse_einsum_equation(equation):
    """
    Analyses an einsum equation.

    :param equation: `numpy.einsum` equation
    :return: three results, list of letters,
        a matrix (see below), lengths of each components,
        duplicates

    The returned a matrix is defined as follows:

    .. math::

        m_{ij}=\\left\\{\\begin{array}{ll}-1 &
        \\text{if letter j is involved in input i} \\\\
        p & \\text{p is position of letter j in equation i}
        \\end{array}\\right.
    """
    spl = equation.strip(' ,').split("->")
    if len(spl) != 2 or len(spl[1]) == 0 or len(spl[0]) == 0:
        raise NotImplementedError(
            "The function only implements the case when there are "
            "two sides in the equation: %r." % equation)
    inputs = list(map(lambda s: s.strip(), spl[0].split(',')))
    output = spl[1]
    all_letters = set(inputs[0])

    # Set of letters
    for inp in inputs[1:]:
        all_letters |= set(inp)
    letters = list(sorted(all_letters))
    for c in letters:
        if not(('a' <= c <= 'z') or ('A' <= c <= 'Z')):
            raise ValueError(
                "Equation %r must only contain lower or upper letters "
                "but %r is not." % (equation, c))

    rev = {c: i for i, c in enumerate(letters)}
    for c in output:
        if c not in letters:
            raise ValueError(
                "Output contains one unexpected letter %r in "
                "equation %r." % (c, equation))
    mat = np.full((len(inputs) + 1, len(letters)), -1, dtype=np.int8)
    for i, inp in enumerate(inputs):
        for k, c in enumerate(inp):
            mat[i, rev[c]] = k
    for k, c in enumerate(output):
        mat[len(inputs), rev[c]] = k
    lengths = [len(inp) for inp in inputs]
    lengths.append(len(output))

    # Look for duplicates
    duplicates = []
    for inp in inputs + [output]:
        if len(inp) == len(set(inp)):
            duplicates.append(None)
            continue
        # There is some duplicates.
        counts = {}
        for i, c in enumerate(inp):
            if c in counts:
                counts[c].append(i)
            else:
                counts[c] = [i]
        duplicates.append(counts)

    return "".join(letters), mat, lengths, duplicates


def decompose_einsum_equation(equation, *shapes):
    """
    Decomposes an equation used in `numpy.einsum` knowing
    the input shapes. It returns a sequence of operations
    to do to compute the results.

    :param equation: a string
    :param shapes: sequence of input shapes
    :return: instance of class *GraphEinsumSubOp*

    Available operations: *expand_dims*, *transpose*, *matmul*, *reduce_sum*,
    *id*, *squeeze*, *diagonal*. It analyses an equation and produces a graph
    where node are instance of class *EinsumSubOp*.
    One example:

    ::

        import onnx
        from onnx import helper, numpy_helper
        import numpy as np
        import onnxruntime as ort
        import time
        from tf2onnx.optimizer.einsum_optimizer import decompose_einsum_equation


        def make_model(op_name, initializer, attrs):
            model = helper.make_model(
                opset_imports=[helper.make_operatorsetid('', 13)],
                ir_version=6,
                producer_name='einsum_test',
                producer_version='1.6.0',
                graph=helper.make_graph(
                    name='einsum_test',
                    inputs=[helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, None)],
                    outputs=[helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, None)],
                    initializer=[numpy_helper.from_array(initializer, name="Y")],
                    nodes=[
                        helper.make_node(op_name, ["X", "Y"], ["Z"], **attrs)
                    ]
                )
            )
            return model


        def make_model2(equation, initializer):
            initializer=[numpy_helper.from_array(initializer, name="Y")]
            seq = decompose_einsum_equation(equation)
            onx = seq.to_onnx("Z", "X", "Y", initializer=initializer)
            return onx


        def main():
            inp1 = np.random.uniform(size=[100, 20, 768]).astype(np.float32)
            inp2 = np.random.uniform(size=[768, 32000]).astype(np.float32)
            model_matmul = make_model("MatMul", inp2, {})
            sess_matmul = ort.InferenceSession(model_matmul.SerializeToString())

            start = time.time()
            for i in range(20):
                res_matmul = sess_matmul.run(["Z"], {"X": inp1})[0]
            print("matmul time:", time.time() - start)

            # Both equations are equivalent but they don't produce
            # the same graph as matrices are aligned the same dimensions
            # ordered by alphabetical order. In the second equation,
            # the last two dimensions are switched.
            for eq in ['bid,nd->bin', 'bdn,in->bdi']:
                eq_name = eq.replace(",", "_").replace("->", "_")

                model_einsum = make_model("Einsum", inp2.transpose(), {'equation': eq})
                with open("model_einsum_%s.onnx" % eq_name, "wb") as f:
                    f.write(model_einsum.SerializeToString())

                model_decompose = make_model2(eq, inp2.transpose())
                with open("model_decompose_%s.onnx" % eq_name, "wb") as f:
                    f.write(model_decompose.SerializeToString())

                sess_einsum = ort.InferenceSession(model_einsum.SerializeToString())
                sess_dec = ort.InferenceSession(model_decompose.SerializeToString())

                start = time.time()
                for i in range(20):
                    res_einsum = sess_einsum.run(["Z"], {"X": inp1})[0]
                print("%s einsum time:" % eq, time.time() - start)
                np.testing.assert_allclose(res_matmul, res_einsum, 1e-5)
                print("Results match")

                start = time.time()
                for i in range(20):
                    res_dec = sess_dec.run(["Z"], {"X": inp1})[0]
                print("%s decompose time:" % eq, time.time() - start)
                np.testing.assert_allclose(res_matmul, res_dec, 1e-5)
                print("Results match")


        main()

    It produces the following output:

    ::

        matmul time: 20.827451944351196
        bid,nd->bin einsum time: 22.50755500793457
        Results match
        bid,nd->bin decompose time: 21.672495365142822
        Results match
        bdn,in->bdi einsum time: 22.701029539108276
        Results match
        bdn,in->bdi decompose time: 20.90113353729248
        Results match
    """
    graph = _decompose_einsum_equation(
        equation, *shapes, op_matmul='batch_dot')

    # Last step: clean unused nodes.
    last_node = graph.last_added_op
    graph.append(EinsumSubOp(last_node.full_dim, 'id', last_node))
    graph.mark_last_node()
    graph.simplify_mm_nodes()
    graph.clean_unused_nodes()
    graph.remove_duplicate_transpose()
    return graph


def is_transpose_identity(perm):
    """
    Tells if the permutation *perm* does nothing (itentity).

    :param perm: permutation
    :return: boolean
    """
    return list(perm) == list(range(len(perm)))


def _basic_verification(lengths, shapes, equation):
    if len(lengths) - 1 != len(shapes):
        raise ValueError(
            "Equation %r has %d inputs but %d shapes are given."
            "" % (equation, len(lengths), len(shapes)))
    for i, (le, sh) in enumerate(zip(lengths, shapes)):
        if le != len(sh):
            raise ValueError(
                "Inputs %d has %d dimensions but shapes %r has %d "
                " in equation %r." % (i, le, sh, len(sh), equation))


def _apply_transpose_reshape(op, row):
    """
    Put all dimensions in the same order.

    :param op: integer (for one input) or an operator
    :param row: letter involved in this input (as a vector of binaries)
    :return: last created operator
    """
    axes = []
    p = 0
    perm = []
    for i, r in enumerate(row):
        if r == -1:
            axes.append((p, i))
        else:
            p += 1
            perm.append((r, i))
    op = EinsumSubOp(len(row), 'expand_dims', op, axes=tuple(axes))
    yield op
    perm.sort()
    p = 0
    new_perm = np.arange(len(row))
    for i, r in enumerate(row):
        if r == -1:
            continue
        new_perm[perm[p][1]] = i
        p += 1
    if not is_transpose_identity(new_perm):
        op = EinsumSubOp(len(row), 'transpose', op, perm=tuple(new_perm))
        yield op


def _apply_squeeze_transpose(op, row_last, row_output):
    """
    Puts output dimension in the expected order.
    """
    perm = []
    sq = []
    for i, d in enumerate(row_output):
        if d == -1:
            sq.append(i)
        else:
            perm.append((d, i))
    perm.sort()
    new_perm = np.arange(len(row_last))
    p = 0
    for i, d in enumerate(row_output):
        if d == -1:
            continue
        new_perm[i] = perm[p][1]
        p += 1
    if not is_transpose_identity(new_perm):
        op = EinsumSubOp(len(row_last), 'transpose', op,
                         perm=tuple(new_perm))
        yield op
    if len(sq) > 0:
        op = EinsumSubOp(len(row_last), 'squeeze', op, axes=tuple(sq))
        yield op


def _apply_einsum_matmul(fd, op1, op2, axes, left, right, ndim,
                         op_matmul, row1, row2):
    """
    Decomposes the generic matrix multiplication into numpy operations
    depending on the operator to use for matrix multiplication
    *op_matmul* (see *decompose_einsum_equation*).
    """
    allowed = {'matmul', 'batch_dot', 'dot'}
    if op_matmul not in allowed:
        raise ValueError(
            "Unknown operator op_matmul=%r not in %r." % (op_matmul, allowed))
    if op_matmul == 'matmul':
        yield EinsumSubOp(fd, 'matmul', op1, op2,
                          axes=axes, left=left, right=right, ndim=ndim)

    elif len(axes) == 0 and len(set(left) & set(right)) == 0:
        yield EinsumSubOp(fd, 'mul', op1, op2)

    elif len(set(axes) & set(left)) == 0 and len(set(axes) & set(right)) == 0:

        # No intersection between axes and right: matrix multiplication
        all_axes = set(left) | set(right) | set(axes)
        common_axes = list(set(left) & set(right))
        for i in range(ndim):
            if i not in all_axes:
                common_axes.append(i)
        common_axes.sort()

        # ReduceSum*
        has_dim = set(i for i in range(len(row1)) if row1[i] >= 0)
        right_no_left = (set(right) & has_dim) - (set(right) & (set(left) | set(axes)))
        if right_no_left:
            op1 = EinsumSubOp(fd, 'reduce_sum_mm', op1, op2,
                              axes=tuple(sorted(right_no_left)))
            yield op1

        has_dim = set(i for i in range(len(row2)) if row2[i] >= 0)
        left_no_right = (set(left) & has_dim) - (set(left) & (set(right) | set(axes)))
        if left_no_right:
            op2 = EinsumSubOp(fd, 'reduce_sum', op2,
                              axes=tuple(sorted(left_no_right)))
            yield op2

        # Transpose
        i_axes = [(-1 if i in common_axes
                   else (1 if i in axes else 0), i)
                  for i in range(ndim)]
        i_axes.sort()
        perm = [_[1] for _ in i_axes]
        perm_left = [i for i in range(len(perm)) if perm[i] in left]
        perm_right = [i for i in range(len(perm)) if perm[i] in right]
        if not is_transpose_identity(perm):
            op1 = EinsumSubOp(fd, 'transpose_mm', op1, op2, perm=tuple(perm))
            yield op1
            op2 = EinsumSubOp(fd, 'transpose', op2, perm=tuple(perm))
            yield op2

        # Reshape
        all_axes = list(range(0, ndim))
        new_axes = all_axes[-len(axes):] if len(axes) > 0 else []
        new_common_axes = all_axes[:len(common_axes)]
        not_in_both = []
        for i in range(0, ndim):
            if i not in left and i not in right and i not in common_axes:
                not_in_both.append(i)

        op = EinsumSubOp(fd, 'batch_dot', op1, op2,
                         batch_axes=tuple(new_common_axes),
                         keep_axes=None, sum_axes=tuple(new_axes),
                         left=tuple(perm_left), right=tuple(perm_right),
                         ndim=ndim)
        yield op

        # Transpose again
        ordered_axes = (common_axes +
                        list(i for i in left if i not in right) +
                        list(i for i in right if i not in left) +
                        not_in_both)
        rev_perm = [(a, i) for i, a in enumerate(ordered_axes)]
        rev_perm.sort()
        rev_perm = [p[1] for p in rev_perm]

        if not is_transpose_identity(rev_perm):
            op_unused = EinsumSubOp(fd, 'transpose_mm', op1,
                                    op, perm=tuple(rev_perm))
            yield op_unused
            op = EinsumSubOp(fd, 'transpose', op, perm=tuple(rev_perm))
            yield op
    else:
        raise NotImplementedError(
            "axes and right or left have axes in common, "
            "axes=%r left=%r right=%r ndim=%r." % (
                axes, left, right, ndim))


def _decompose_einsum_equation(equation, *shapes, op_matmul='batch_dot'):
    """
    :param op_matmul: which operator to use for matrix multiplication,
        a single operator *matmul*, or *batch_dot* with *transposes*,
        *reduce_sum*, or just *dot*
    """
    letters, mat, lengths, duplicates = analyse_einsum_equation(equation)
    if len(letters) != mat.shape[1]:
        raise RuntimeError(  # pragma: no cover
            "Unexpected number of letters %r, shape=%r." % (
                letters, mat.shape))
    if len(shapes) > 0:
        _basic_verification(lengths, shapes, equation)
    else:
        shapes = [(2,) * le for le in lengths[:-1]]

    # last_row, current_row (row = shape)
    rows = np.full((2, mat.shape[1]), -1)
    graph = GraphEinsumSubOp(letters, mat, lengths, duplicates)
    fd = mat.shape[1]
    for i in range(len(shapes)):
        graph.append(i)

        # Input matrix aligned to the same dimensions.
        op = EinsumSubOp(fd, 'id', i)
        op.compute_output_row(rows[1, :], mat[i, :])
        marked = graph.append(op)

        duplicate = duplicates[i]
        if duplicate is not None:
            # Diagonal
            diag = []
            for _, v in duplicate.items():
                if len(v) == 1:
                    continue
                diag.append((v[0], tuple(v)))
            op = EinsumSubOp(fd, 'diagonal', op, diag=diag)
            op.compute_output_row(rows[1, :], mat[i, :])
            tr_row = rows[1, :]
            marked = graph.append(op)
        else:
            diag = None
            tr_row = mat[i]

        for op in _apply_transpose_reshape(op, tr_row):
            op.compute_output_row(rows[1, :])
            marked = graph.append(op)

        # Reduction? (a dimension not used later)
        red = []
        for d in range(0, mat.shape[1]):
            if (mat[i + 1:, d].max() == -1 and rows[1, d] != -1 and
                    rows[0, d] == -1):
                red.append(d)
        if len(red) > 0:
            op = EinsumSubOp(fd, 'reduce_sum',
                             graph.last_added_op, axes=tuple(red))
            op.compute_output_row(rows[1, :])
            marked = graph.append(op)

        if graph.last_op is not None:
            # Matrix multiplication?
            common_dims = []
            left = []
            right = []
            for d in range(0, mat.shape[1]):
                if rows[:, d].min() >= 0:
                    if mat[i + 1:, d].max() >= 0:
                        left.append(d)
                        right.append(d)
                    else:
                        common_dims.append(d)
                else:
                    if rows[0, d] >= 0:
                        left.append(d)
                    if rows[1, d] >= 0:
                        right.append(d)
            for iop in _apply_einsum_matmul(
                    fd, graph.last_op, op, axes=tuple(common_dims),
                    left=tuple(left), right=tuple(right),
                    ndim=rows.shape[1], op_matmul=op_matmul,
                    row1=rows[0, :], row2=rows[1, :]):
                op = iop
                op.compute_output_row(rows[0, :], rows[1, :], ab=True)
                marked = graph.append(op)

        # End
        graph.mark(i, marked)
        rows[0, :] = rows[1, :]

    # Final output
    if mat[len(shapes), :].max() >= 0:
        rows[1, :] = mat[len(shapes), :]
        red = []
        for d in range(0, mat.shape[1]):
            if rows[0, d] > 0 and rows[1, d] == -1:
                red.append(d)
            elif rows[0, d] == -1 and rows[1, d] >= 0:
                raise RuntimeError(
                    "Issue in equation %r, variable %d, last_result is %r, "
                    "output is %r." % (equation, d, rows[0, :], rows[1, :]))
        if len(red) > 0:
            op = EinsumSubOp(fd, 'reduce_sum', op, axes=tuple(red))
            graph.append(op)
            op.compute_output_row(rows[1, :])

        # Removes empty axes.
        for op in _apply_squeeze_transpose(op, rows[1, :], mat[len(shapes), :]):
            op.compute_output_row(rows[1, :])
            graph.append(op)
    return graph


_ml_transpose_coefs = {
    'CST_': 0.4720163707200312,
    'begin': 0.0,
    'dbegin': 0.0,
    'dend': 0.0,
    'dim': 0.0,
    'discont': 0.0180766756730043,
    'edit': 0.06940318842803926,
    'end': 0.0,
    'end16': 0.0,
    'end32': 0.0,
    'ibegin16': 0.0,
    'ibegin2': 0.0,
    'ibegin32': 0.0,
    'ibegin4': 0.0,
    'ibegin64': 0.0,
    'ibegin8': 0.04389296884016416,
    'iend16': 0.5316238365817172,
    'iend2': 0.16287259236456927,
    'iend32': 0.0,
    'iend4': 0.0,
    'iend64': 0.0,
    'iend8': 0.0,
    'middle': 1.3381940773605624e-06,
    'rbegin': 0.0,
    'rdiscont': 0.0,
    'redit': 0.18604684802855143,
    'rend': 0.0,
    'rend16': 0.0,
    'rend32': 0.0,
    'rev': 0.42909943168149206,
    'rmiddle': 0.0,
    'rot': 0.22272566615803094,
    'size': 2.8663794075460607e-06}


def _edit_distance(seq1, seq2):
    "Computes the edit distance between two sequences."
    dist = {(-1, -1): 0}
    if len(seq1) == 0:
        for j, d in enumerate(seq2):
            dist[-1, j] = dist[-1, j - 1] + 1
            dist[j, -1] = dist[j - 1, -1] + 1
    for i, c in enumerate(seq1):
        dist[i, -1] = dist[i - 1, -1] + 1
        dist[-1, i] = dist[-1, i - 1] + 1
        for j, d in enumerate(seq2):
            opt = []
            if (i - 1, j) in dist:
                x = dist[i - 1, j] + 1
                opt.append((x, (i - 1, j)))
            if (i, j - 1) in dist:
                x = dist[i, j - 1] + 1
                opt.append((x, (i, j - 1)))
            if (i - 1, j - 1) in dist:
                x = dist[i - 1, j - 1] + (1 if c != d else 0)
                opt.append((x, (i - 1, j - 1)))
            mi = min(opt)
            dist[i, j] = mi[0]

    return dist[len(seq1) - 1, len(seq2) - 1]


def _is_rotation(perm):
    t = tuple(perm)
    c = list(range(len(perm)))
    for i in range(len(c)):
        for k in range(len(c)):  # pylint: disable=C0200
            c[k] = (k + i) % len(c)
        if t == tuple(c):
            return True
    return False


def _relu(x, origin=0):
    return origin if x < origin else x


def compute_transposition_features(shape, perm):
    """
    Given a shape and a permutation, computes many features
    used to predict the cost of the transposition.

    :param shape: shape
    :param perm: permutation
    :return: dictionary of features
    """
    total = np.prod(np.array(shape, dtype=np.int64))

    begin = 1
    dbegin = 0
    for i, p in enumerate(perm):
        if p != i:
            break
        dbegin += 1
        begin *= shape[i]

    end = 1
    dend = 0
    for i in range(len(perm) - 1, -1, -1):
        if perm[i] != i:
            break
        dend += 1
        end *= shape[i]

    dis_cont = 0
    for i in range(1, len(shape)):
        if perm[i] != perm[i - 1] + 1:
            dis_cont += 1

    middle = max(1, int(total / (end * begin)))
    feat = dict(size=total, begin=begin, end=end, middle=middle,
                dim=len(shape), discont=dis_cont)

    for c in [16, 32]:
        feat["end%d" % c] = _relu(end, c)

    keys = list(feat)
    for k in keys:
        if k in {'dim', 'cpu', 'size'}:
            continue
        feat['r%s' % k] = float(feat[k] / total)

    for c in [2, 4, 8, 16, 32, 64]:
        feat["iend%d" % c] = float(end >= c)
        feat["ibegin%d" % c] = float(begin >= c)

    # feat['CST'] = 1
    feat['CST_'] = -1
    feat['dbegin'] = - dbegin
    feat['dend'] = - dend

    keys = list(feat)
    for k in keys:
        if k.startswith('end') or k.startswith('begin'):
            feat[k] = - feat[k]
        elif k.startswith('rend') or k.startswith('rbegin'):
            feat[k] = - feat[k]
        elif k.startswith('iend') or k.startswith('ibegin'):
            feat[k] = - feat[k]
        elif k == "rdiscont":
            feat[k] = - feat[k]

    idp = list(range(len(perm)))
    feat["rot"] = -1 if _is_rotation(perm) else 0
    feat["rev"] = 1 if perm == tuple(idp[::-1]) else 0
    feat["edit"] = _edit_distance(idp, perm)
    feat["redit"] = feat["edit"] / len(idp)
    return feat


def predict_transposition_cost(shape, perm, coefs=None):
    """
    Given a shape and a permutation, predicts the cost of the
    transposition.

    :param shape: shape
    :param perm: permutation
    :param coefs: trained coefficients or None to get
        the default ones
    :return: dictionary of features
    """
    if coefs is None:
        coefs = _ml_transpose_coefs
    feat = compute_transposition_features(shape, perm)
    res = 0
    for k, v in feat.items():
        res += v * coefs[k]
    return max(0., res / 1000)


class CachedEinsum:
    """
    Stores all the necessary information to cache the preprocessing
    of a an einsum equation. The optimization only works it
    the best permutation of letters is chosen.

    :param equation: numpy equation
    :param opset: ONNX opset
    :param optimize: finds the best letter permutation
    :param dtype: dtype
    :param decompose: to decompose Einsum operator or to keep it as is
    :param strategy: optimization strategy (only `ml` is implemented)
    :param key: key used to cache this class

    The class creates the following attributes:
    * `equation_` corresponding to the best equivalent equation
    * `graph_`: the corresponding graph returned by function
        :func:`decompose_einsum_equation
        <mlprodict.testing.einsum.einsum_impl.decompose_einsum_equation> `
    * `onnx_`: if a conversion to onnx is used, stores the onnx graph
    * `runtime_`: a function used by `__call__`, calls the runtime
    """
    einsum_cache = {}


    def __init__(self, equation, opset=None, optimize=False,
                 dtype=np.float32, decompose=True, strategy="ml", key=None):
        self.equation = equation
        self.opset = opset
        self.optimize = optimize
        self.dtype = dtype
        self.decompose = decompose
        self.strategy = strategy
        self.key = key

    def default_inputs(self, n=None):
        """
        Returns default inputs (reshaped np.arange + 0.7i).

        :param n: dimension (all dimension have the same size)

        If *N is None*, N is given a size depending on the number of letters
        to avoid spending too much time on optimization.
        """
        if n is None:
            letters = set(c for c in self.equation
                          if "a" <= c <= "z" or "A" <= c <= "Z")
            nn = math.factorial(len(letters))
            n = max(int(2 ** 11 / nn), 4)
            n = min(n, 15)
        inps = self.equation.split('->')[0].split(',')
        lens = [len(s) for s in inps]
        inputs = [np.arange(n ** d).reshape((n,) * d) for d in lens]
        inputs = [(i + 0.7 * ii).astype(self.dtype)
                  for ii, i in enumerate(inputs)]
        return inputs

    def build(self):
        """
        Preprocesses the equation builds whatever is necessary
        to compute the result of the einsum equation.
        """
        if not self.optimize and not hasattr(self, 'equation_'):
            self.equation_ = self.equation
        elif self.strategy == 'ml':
            self.equation_ = self._build_optimize_ml()
        else:
            raise ValueError(  # pragma error
                "Unknown strategy %r." % self.strategy)
        self.build_runtime()

    def _build_optimize_ml(self):
        """
        Selection the best equation by evaluation the sum of of the cost
        of all permutations involved in the replacement of einsum operator.
        """
        # loops over all permutations
        if self.equation.lower() != self.equation:
            raise RuntimeError(
                "Only lower equation can be optimized, %r is not." % self.equation)
        letters = list(
            sorted(set(c for c in self.equation if "a" <= c <= "z")))
        subset = list(permutations(letters))
        subset.insert(0, letters)
        best = []
        confs = []
        inputs = None
        for perm in subset:
            replace = {d: c for c, d in zip(letters, perm)}
            eq = self.equation
            for k, v in replace.items():
                eq = eq.replace(k, v.upper())
            eq = eq.lower()
            inst = CachedEinsum(eq, opset=self.opset,
                                optimize=False, dtype=self.dtype,
                                decompose=self.decompose)
            inst.build()
            if inputs is None:
                inputs = inst.default_inputs()
            if hasattr(inst, 'onnx_'):
                onx = inst.onnx_
            else:
                inits = [
                    ('X%d' % i, FloatTensorType(list(inputs[i].shape)))
                    for i in range(len(inputs))]
                onx = inst.graph_.to_onnx('Y', *inits, opset=self.opset)

            rt = OnnxMicroRuntime(onx)
            dict_inputs = {'X%d' % i: inp for i, inp in enumerate(inputs)}
            out = rt.run(dict_inputs)

            transposes = []
            for node in onx.graph.node:  # pylint: disable=E1101
                if node.op_type == 'Transpose':
                    shape = [(d * 10 if d > 1 else d)
                             for d in out[node.input[0]].shape]
                    transposes.append(
                        [shape, list(node.attribute[0].ints)])

            delta = sum(max(0, predict_transposition_cost(*v))
                        for v in transposes)

            confs.append((delta, eq))
            if len(best) < 10:
                best.append((delta, eq))
                best.sort()
            elif delta < best[-1][0]:
                best[-1] = (delta, eq)
                best.sort()
        self.optimized_ = best
        self.timed_permutations_ = confs
        return best[0][1]

    def build_onnx_einsum(self, input_names):
        """
        Builds an ONNX graph with a single einsum operator.
        """
        opset = (self.opset if self.opset is not None
                 else PREFERRED_OPSET)
        ir_version = OPSET_TO_IR_VERSION.get(opset, max(OPSET_TO_IR_VERSION.values()))
        proto_type = TensorProto.FLOAT

        model = helper.make_model(
            opset_imports=[helper.make_operatorsetid('', opset)],
            ir_version=ir_version,
            producer_name='tensorflow-onnx',
            producer_version='0.0.1',
            graph=helper.make_graph(
                name='einsum',
                inputs=[helper.make_tensor_value_info(n, proto_type, None)
                        for n in input_names],
                outputs=[helper.make_tensor_value_info("Y", proto_type, None)],
                nodes=[
                    helper.make_node(
                        'Einsum', input_names, ["Y"], equation=self.equation_)]))
        return model

    def build_runtime(self):
        """
        Builds the runtime associated to the
        equation `self.equation_`. It requires onnxunruntime.
        """
        # The conversion should not fail if onnxruntime is not here.
        # Delayed import.
        # from onnxruntime import InferenceSession  # pylint: disable=C0415
        if self.decompose:
            self.graph_ = decompose_einsum_equation(self.equation_)

            n_inputs = len(self.graph_.metadata['lengths']) - 1
            input_names = ['X%d' % i for i in range(n_inputs)]
            self.onnx_names_ = input_names
            onx = self.graph_.to_onnx(
                'Y', *input_names, opset=self.opset, dtype=self.dtype)
            self.onnx_ = onx
        else:
            n_inputs = len(self.equation.split('->')[0].split(','))
            input_names = ['X%d' % i for i in range(n_inputs)]
            self.onnx_ = self.build_onnx_einsum(input_names)
            self.onnx_names_ = input_names
        # self.sess_ = InferenceSession(self.onnx_.SerializeToString())
        # self.runtime_ = lambda *inputs: self.sess_.run(
        #     None, dict(zip(self.onnx_names_, inputs)))[0]

    def __call__(self, *inputs):
        """
        Calls the runtime `self.runtime_`.
        """
        if not hasattr(self, 'runtime_'):
            raise RuntimeError(
                "Method build_runtime was not called.")
        return self.runtime_(*inputs)

    @staticmethod
    def build_einsum(equation, opset, optimize, dtype,
                     decompose=True, strategy='ml', key=None):
        """
        Creates an instance of *CachedEinsum*.
        """
        inst = CachedEinsum(equation, opset=opset,
                            optimize=optimize, dtype=dtype,
                            decompose=decompose, strategy=strategy,
                            key=key)
        inst.build()
        return inst


def optimize_einsum(equation, dtype, optimize=True,
                    cache=True, opset=None, decompose=True,
                    strategy='ml'):
    """
    This function returns an instance of CachedEinsum.
    It has an attribute `equation_` which holds a new equation
    equal to the first one but with permutated letter.
    This new equation has a smaller computation time
    when executed with onnxruntime (optimized on the machine CPU).
    Attribute `equation_` returns an optimized equation,
    `timed_permutations_` returns the execution time for
    every permutation. Results are cached based on the function
    arguments. It saves time if the same einsum equation
    appears twice.
    """
    einsum_cache = CachedEinsum.einsum_cache
    cached = None
    if cache:
        key = equation, opset, optimize, dtype, decompose
        cached = einsum_cache.get(key, None)
    if cached is None:
        cached = CachedEinsum.build_einsum(
            equation, opset, optimize, dtype, decompose=decompose,
            key=key, strategy=strategy)
    else:
        cache = False
    if cache:
        einsum_cache[key] = cached
    return cached


class EinsumOptimizer(GraphOptimizerBase):
    """Remove einsum operators and replace them by a combination of
    Transpose, ReduceSum, Reshape, MatMul, Gemm, Squeeze, Unsqueeze, Mul.
    It does not handle equation with `...`, square indices (`ii->i`),
    and undefined output shape (`ab,bc`).

    The optimizer may change the einsum equation for an equivalent
    one but more efficient for onnxruntime. All matrices are aligned
    to the same dimensions ordered by alphabetical order of the letters
    in the equation. Depending on that order, the transposing
    and the matrix computation may be more or less efficient.
    The worst order may be 4, 5 times slower than the best order.

    :param decompose: keep the operator einsum or replace it
        by a graph combining operators listed above
    """

    def __init__(self, decompose=True):  # pylint: disable=useless-super-delegation
        super(EinsumOptimizer, self).__init__()
        self._decompose = decompose
        self._strategy = 'ml'

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        graph_changed = True
        while graph_changed:
            graph_changed = False
            ops = graph.get_nodes()
            for op in ops:
                if op.type == "Einsum" and self._optimize_einsum(op, graph):
                    graph_changed = True
                    self.graph_been_opt = True
        return graph

    def _optimize_einsum(self, node, graph):
        """Apply the optimizer."""
        inp_shape = graph.get_shape(node.input[0])
        if inp_shape is None:
            # The rank must be known
            return False

        equation = node.attr['equation'].s.decode('ascii')

        # optimize the equation? decompose=True, False
        new_equation_obj = optimize_einsum(
            equation, decompose=self._decompose, dtype=np.float32, opset=graph.opset,
            strategy=self._strategy)
        if self._decompose:
            seq = decompose_einsum_equation(new_equation_obj.equation_)
            new_nodes = list(seq.to_tf2onnx(graph, node))

            if len(new_nodes) > 0:
                # optimisation was made, node should be removed.
                last_node = new_nodes[-1]
                self.logger.info(
                    "replacing einsum node %r by its decomposed version, name of the last "
                    "node %r.", node.name, last_node.name)
                graph.replace_all_inputs(node.output[0], last_node.output[0])
                graph.safe_remove_nodes([node])
                if node.output[0] in graph.outputs:
                    graph.make_node(
                        'Identity', [last_node.output[0]], outputs=[node.output[0]],
                        name="%s_final" % node.name)
                return True
        elif equation != new_equation_obj.equation_:
            node.attr['equation'].s = new_equation_obj.equation_.encode('ascii')
            self.logger.info(
                "replacing einsum equation %r by %r",
                equation, new_equation_obj.equation_)
            return True
        return False
