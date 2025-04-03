# SPDX-License-Identifier: Apache-2.0


"""
signal
"""

import logging

import numpy as np
from onnx import onnx_pb, helper
from onnx.numpy_helper import to_array
from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring

def make_dft_constant(length, dtype, fft_length):
    utils.make_sure(fft_length > 0, "fft_length must be strictly positive but is %r.", fft_length)
    new_length = max(length, fft_length // 2 + 1)
    n = np.arange(new_length)
    k = n.reshape((new_length, 1)).astype(np.float64)
    mat = np.exp(-2j * np.pi * k * n / fft_length)
    both = np.empty((2,) + mat.shape, dtype=dtype)
    both[0, :, :] = np.real(mat)
    both[1, :, :] = np.imag(mat)
    return both


class CommonFFTOp:
    supported_dtypes = [
        onnx_pb.TensorProto.FLOAT,
        onnx_pb.TensorProto.FLOAT16,
        onnx_pb.TensorProto.DOUBLE,
        onnx_pb.TensorProto.COMPLEX64,
        onnx_pb.TensorProto.COMPLEX128,
    ]

    @classmethod
    def any_version(cls, const_length, opset, ctx, node, axis=None,
                    fft_length=None, dim=None, onnx_dtype=None, shape=None,
                    input_name=None, **kwargs):
        """
        Inspired from `Python implementation of RFFT
        <https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/>`_.

        Complex version:

        ::

            import numpy as np

            def _DFT_cst(N, fft_length):
                n = np.arange(N)
                k = n.reshape((N, 1)).astype(np.float64)
                M = np.exp(-2j * np.pi * k * n / fft_length)
                return M

            def DFT(x, fft_length=None):
                if len(x.shape) == 1:
                    x = x.reshape((-1, 1))
                else:
                    x = x.T
                if fft_length is None:
                    fft_length = x.shape[0]
                cst = _DFT_cst(x.shape[0], fft_length)
                size = fft_length // 2 + 1
                return np.dot(cst[:, :fft_length], x[:fft_length]).T[:, :size]

        Real version, first axis is (real, imag) part:

        ::

            def _DFT_real_cst(N, fft_length):
                n = np.arange(N)
                k = n.reshape((N, 1)).astype(np.float64)
                M = np.exp(-2j * np.pi * k * n / fft_length)
                both = np.empty((2,) + M.shape)
                both[0, :, :] = np.real(M)
                both[1, :, :] = np.imag(M)
                return both

            def DFT_real(x, fft_length=None):
                if len(x.shape) == 1:
                    x = x.reshape((-1, 1))
                else:
                    x = x.T
                if fft_length is None:
                    fft_length = x.shape[0]
                size = fft_length // 2 + 1
                cst = _DFT_real_cst(x.shape[0], fft_length)
                res = np.dot(cst[:, :, :fft_length], x[:fft_length])[:, :size, :]
                return np.transpose(res, (0, 2, 1))

        `tf.signal.rfft` also works for tensors with 3+ dimensions.
        The strategy here is to reshape the matrix into a 2D matrix
        then apply FFT, then reshape the matrix into dimension (2, ...).
        The first dimension is still real/imaginary part.
        """
        if input_name is None:
            input_name = node.input[0]
            node_name = node.name
        else:
            node_name = input_name.split(':')[0]

        minus_one = ctx.make_const(name=utils.make_name('FFT_minus_one'),
                                   np_val=np.array([-1], dtype=np.int64))
        zero = ctx.make_const(name=utils.make_name('FFT_zero'),
                              np_val=np.array([0], dtype=np.int64))

        if axis is None:
            consumers = ctx.find_output_consumers(node.output[0])
            consumer_types = set(op.type for op in consumers)
            utils.make_sure(
                axis == 0 or consumer_types == {'ComplexAbs'},
                "Current implementation of RFFT or FFT only allows ComplexAbs as consumer not %r",
                consumer_types)

            onnx_dtype = ctx.get_dtype(input_name)
            utils.make_sure(onnx_dtype in CommonFFTOp.supported_dtypes, "Unsupported input type.")
            shape = ctx.get_shape(node.input[0])
            if shape is None or len(shape) > 2:
                if shape is None or min(shape) <= 0:
                    two = ctx.make_const(name=utils.make_name('FFT_two'),
                                         np_val=np.array([2], dtype=np.int64))
                    current_shape = ctx.make_node(
                        'Shape', [input_name], name=utils.make_name('FFT_' + node_name + '_shape'))
                    last_dim = ctx.make_node('Gather', [current_shape.output[0], minus_one.name],
                                             name=utils.make_name('FFT_' + node_name + '_shape'))
                    new_shape = ctx.make_node('Concat', [minus_one.name, last_dim.output[0]],
                                              name=utils.make_name('FFT_' + node_name + '_concat'),
                                              attr={'axis': 0}).output[0]
                    if opset >= 10:
                        belly = ctx.make_node(
                            "Slice", inputs=[current_shape.output[0], zero.name, minus_one.name, zero.name],
                            name=utils.make_name('FFT_fshape_' + node_name + 'rfft'))
                    else:
                        belly = ctx.make_node(
                            "Slice", inputs=[current_shape.output[0]], attr=dict(starts=[0], ends=[-1], axes=[0]),
                            name=utils.make_name('FFT_fshape_' + node_name + 'rfft'))
                    reshape_final = ctx.make_node(
                        'Concat', [two.name, belly.output[0], minus_one.name],
                        name=utils.make_name('FFT_' + node_name + '_concatshape'),
                        attr={'axis': 0}).output[0]
                    if shape is not None and shape[-1] > 0:
                        shape_n = shape[-1]
                    else:
                        shape_n = None
                else:
                    new_shape = ctx.make_const(name=utils.make_name('FFT_new_shape'),
                                               np_val=np.array([-1, shape[-1]], dtype=np.int64)).name
                    reshape_final = ctx.make_const(
                        name=utils.make_name('FFT_final_shape'),
                        np_val=np.array([2] + list(shape[:-1]) + [-1], dtype=np.int64)).name
                    shape_n = shape[-1]
                reshaped_input = ctx.make_node('Reshape', [input_name, new_shape],
                                               name=utils.make_name('FFT_' + node_name + '_reshape'))
                input_name = reshaped_input.output[0]
                shape = None
            else:
                reshape_final = None
                shape_n = shape[-1]

            if onnx_dtype in (onnx_pb.TensorProto.COMPLEX64, onnx_pb.TensorProto.COMPLEX128):
                parent = ctx.get_node_by_output_in_current_graph(input_name)
                utils.make_sure(
                    parent.type == 'Cast' and parent.get_attr_value('to') == onnx_dtype,
                    "Current implementation of FFT or RFFT assumes the input is real or complex produced "
                    "by a node Cast just before this one.")
                input_name = parent.input[0]
                onnx_dtype = ctx.get_dtype(input_name)

            np_dtype = utils.map_onnx_to_numpy_type(onnx_dtype)
        else:
            shape_n = dim
            utils.make_sure(shape is not None, "shape must be known.")
            np_dtype = utils.map_onnx_to_numpy_type(onnx_dtype)
            reshape_final = None

        if np_dtype == np.float16:
            res_onnx_dtype = utils.map_numpy_to_onnx_dtype(np.float16)
            np_dtype = np.float16
        elif np_dtype in (np.float32, np.complex64):
            res_onnx_dtype = utils.map_numpy_to_onnx_dtype(np.float32)
            np_dtype = np.float32
        else:
            res_onnx_dtype = utils.map_numpy_to_onnx_dtype(np.float64)
            np_dtype = np.float64

        if const_length:
            # RFFT: length of FFT is known, some computation
            # (see function make_dft_constant)
            # can be done at conversion time and stored as constant
            utils.make_sure(axis is not None or len(node.input) == 2,
                            "Two inputs expected not %r", len(node.input) if axis is None else '?')

            # This input should be a constant.
            if fft_length is None:
                fft_length_name = node.input[1]
                node_fft_length = ctx.get_node_by_output(fft_length_name, search_in_parent_graphs=True)
                utils.make_sure(node_fft_length.type == 'Const',
                                "fft_length should be a constant, the other case is not implemented yet.")
                value = node_fft_length.get_attr("value")
                value_array = to_array(value.t)
                if axis is None:
                    utils.make_sure(
                        value_array.shape == (1,), "Unexpected shape for fft_length (%r)", value_array.shape)
                    fft_length = value_array[0]
                else:
                    utils.make_sure(
                        axis < len(value_array), "Inconsistent axis %r incompatible with fft_length=%r",
                        axis, value_array)
                    fft_length = value_array[axis]
                utils.make_sure(shape_n is None or fft_length <= shape_n,
                                "Case fft_length=%r > shape[1]=%r (shape=%r) is not implemented.",
                                fft_length, shape_n, shape)

            if axis is not None or (shape_n is not None and fft_length < shape_n):
                real_imag_part = make_dft_constant(shape_n, np_dtype, fft_length)[:, :, :fft_length]

                if axis != 0:
                    if opset >= 10:
                        cst_length = ctx.make_const(
                            name=utils.make_name('CPLX_cstl'), np_val=np.array([fft_length], dtype=np.int64))
                        sliced_input = ctx.make_node(
                            "Slice", inputs=[input_name, zero.name, cst_length.name, minus_one.name],
                            name=utils.make_name('CPLX_S1_' + node_name + 'rfft'))
                    else:
                        sliced_input = ctx.make_node(
                            "Slice", inputs=[input_name], attr=dict(starts=[0], ends=[fft_length], axes=[-1]),
                            name=utils.make_name('CPLX_S1_' + node_name + 'rfft'))
                    input_name = sliced_input.output[0]
            else:
                size = fft_length // 2 + 1
                real_imag_part = make_dft_constant(shape_n, np_dtype, fft_length)[:, :size, :fft_length]

            onx_real_imag_part = ctx.make_const(
                name=utils.make_name('cst_rfft_%d' % shape_n), np_val=real_imag_part)
            onx_real_imag_part_name = onx_real_imag_part.name
        else:
            # FFT: length of FFT is unknown at conversion time, the matrix
            # created by function make_dft_constant must be
            # done in ONNX.
            utils.make_sure(axis is None, "Dynamic version of FFT is not implemented when axis != None.")
            dyn_shape_all = ctx.make_node("Shape", inputs=[input_name],
                                          name=utils.make_name('CPLX_' + node_name + 'shape'))
            m1_cst = ctx.make_const(name=utils.make_name('CPLX_m1'), np_val=np.array([-1], dtype=np.int64))
            dyn_shape = ctx.make_node('Gather', inputs=[dyn_shape_all.output[0], m1_cst.name])
            one_tensor = helper.make_tensor("value", res_onnx_dtype, dims=[1], vals=[1])
            cst_1 = ctx.make_node("ConstantOfShape", inputs=[dyn_shape.output[0]], attr={"value": one_tensor})
            just_0 = ctx.make_const(name=utils.make_name('CPLX1'), np_val=np.array([0], dtype=np.int64))
            rng1 = ctx.make_node("CumSum", inputs=[cst_1.output[0], just_0.name],
                                 name=utils.make_name('CPLX_' + node_name + 'range'))
            p1_cst = ctx.make_const(name=utils.make_name('CPLX_p1'), np_val=np.array([1], dtype=np_dtype))
            rng = ctx.make_node("Sub", inputs=[rng1.output[0], p1_cst.name],
                                name=utils.make_name('CPLX_' + node_name + 'range'))
            resh_cst = ctx.make_const(name=utils.make_name('CPLX_reshape'), np_val=np.array([1, -1], dtype=np.int64))
            rng_tr1 = ctx.make_node("Reshape", inputs=[rng.output[0], resh_cst.name],
                                    name=utils.make_name('CPLX_' + node_name + 'range'))
            resh_cst = ctx.make_const(name=utils.make_name('CPLX_reshape'), np_val=np.array([-1, 1], dtype=np.int64))
            rng_tr2 = ctx.make_node("Reshape", inputs=[rng.output[0], resh_cst.name],
                                    name=utils.make_name('CPLX_' + node_name + 'range'))
            rng_mat = ctx.make_node('MatMul', inputs=[rng_tr2.output[0], rng_tr1.output[0]],
                                    name=utils.make_name('CPLX_' + node_name + 'range2'))
            pi_cst = ctx.make_const(name=utils.make_name('CPLX_pi'), np_val=np.array([np.pi * 2], dtype=np_dtype))
            angle_pi = ctx.make_node("Mul", inputs=[rng_mat.output[0], pi_cst.name],
                                     name=utils.make_name('CPLX_' + node_name + 'angle_pi'))
            shape_cast = ctx.make_node('Cast', inputs=[dyn_shape.output[0]], attr={'to': res_onnx_dtype})
            angle_pibn = ctx.make_node("Div", inputs=[angle_pi.output[0], shape_cast.output[0]],
                                       name=utils.make_name('CPLX_' + node_name + 'angle'))
            if opset >= 13:
                angle = ctx.make_node("Unsqueeze", inputs=[angle_pibn.output[0], just_0.name],
                                      name=utils.make_name('CPLX_' + node_name + 'angles'))
            else:
                angle = ctx.make_node("Unsqueeze", inputs=[angle_pibn.output[0]],
                                      name=utils.make_name('CPLX_' + node_name + 'angles'),
                                      attr={'axes': [0]})
            rng_cos = ctx.make_node("Cos", inputs=[angle.output[0]],
                                    name=utils.make_name('CPLX_' + node_name + 'cos'))
            rng_sin = ctx.make_node("Sin", inputs=[angle.output[0]],
                                    name=utils.make_name('CPLX_' + node_name + 'sin'))
            onx_real_imag_part = ctx.make_node("Concat", inputs=[rng_cos.output[0], rng_sin.output[0]],
                                               name=utils.make_name('CPLX_' + node_name + '_cst_fft'),
                                               attr={'axis': 0})
            onx_real_imag_part_name = onx_real_imag_part.output[0]
            fft_length = None

        if axis != 0:
            perm = [1, 0]
            trx = ctx.make_node(
                "Transpose", inputs=[input_name], attr=dict(perm=perm),
                name=utils.make_name(node_name + '_T_')).output[0]
        else:
            trx = input_name

        if axis is None:
            ctx.remove_node(node_name)
        mult = ctx.make_node(
            "MatMul", inputs=[onx_real_imag_part_name, trx],
            name=utils.make_name('CPLX_M_' + node_name + 'rfft'))

        if not const_length or (axis == 1 or (fft_length < shape_n and axis != 0)):
            size = fft_length // 2 + 1
            if opset >= 10:
                cst_axis = ctx.make_const(
                    name=utils.make_name('CPLX_csta'), np_val=np.array([-2], dtype=np.int64))
                cst_length = ctx.make_const(
                    name=utils.make_name('CPLX_cstl'), np_val=np.array([size], dtype=np.int64))
                sliced_mult = ctx.make_node(
                    "Slice", inputs=[mult.output[0], zero.name, cst_length.name, cst_axis.name],
                    name=utils.make_name('CPLX_S2_' + node_name + 'rfft'))
            else:
                sliced_mult = ctx.make_node(
                    "Slice", inputs=[mult.output[0]], attr=dict(starts=[0], ends=[size], axes=[-2]),
                    name=utils.make_name('CPLX_S2_' + node_name + 'rfft'))
        else:
            sliced_mult = mult

        if axis in (None, 1):
            perm = [0, 2, 1]
            last_node = ctx.make_node(
                "Transpose", inputs=[sliced_mult.output[0]], attr=dict(perm=perm),
                name=utils.make_name('CPLX_T_' + node_name + 'rfft'))
        else:
            last_node = sliced_mult

        if reshape_final is not None:
            reshaped_last_node = ctx.make_node(
                "Reshape", inputs=[last_node.output[0], reshape_final],
                name=utils.make_name('CPLX_Reshape_' + node_name + 'rfft'))
        else:
            reshaped_last_node = last_node

        if axis is None:
            ctx.replace_all_inputs(node.output[0], reshaped_last_node.output[0])  # ops=ctx.get_nodes()
        return last_node


@tf_op("RFFT")
class RFFTOp(CommonFFTOp):
    # support more dtype

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        return cls.any_version(True, 1, ctx, node, **kwargs)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # Slice changed in opset 10.
        return cls.any_version(True, 10, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Unsqueeze changed in opset 13.
        return cls.any_version(True, 13, ctx, node, **kwargs)


@tf_op("FFT")
class FFTOp(CommonFFTOp):
    # support more dtype

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        return cls.any_version(False, 1, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        return cls.any_version(False, 13, ctx, node, **kwargs)


class CommonFFT2DOp(CommonFFTOp):

    @classmethod
    def any_version_2d(cls, const_length, opset, ctx, node, **kwargs):
        """
        Python code equivalent to FF2D (assuming `fft_length[i] <= input.shape[i]` for all i).

        This code was automatically generated with the
        code below and then polished.
        It first starts with an implementation of the
        function *fft2d* with *numpy*. It then uses an numpy API
        for ONNX to convert this code into ONNX. Finally,
        the ONNX graph is exported a code following tensor-flow
        API, this code can replicate this ONNX graph.

        ::

            def onnx_dft_real_cst(N, fft_length):
                n = npnx.arange(0, N).astype(np.float32)
                new_shape = npnx.concat(npnx.expand_dims(N, axis=0),
                                        np.array([1], dtype=np.int64))
                k = n.reshape(new_shape).astype(np.float32)
                kn = (k * n /
                      fft_length.astype(np.float32) *
                      npnx.cst(-2 * np.pi, dtype=np.float32)
                mcos = npnx.unsqueeze(npnx.cos(kn), axes=0)
                msin = npnx.unsqueeze(npnx.sin(kn), axes=0)
                return npnx.vstack(mcos, msin)

            def onnx_rfft_3d_1d(x, fft_length, transpose=True):
                if fft_length is None:
                    raise RuntimeError("fft_length must be specified.")

                size = fft_length // 2 + 1
                cst = onnx_dft_real_cst(fft_length, fft_length)
                if transpose:
                    xt = npnx.transpose(x, (0, 2, 1))
                    a = cst[:, :, :fft_length]
                    b = xt[:, :fft_length, :]
                    a = npnx.expand_dims(a, 0)
                    b = npnx.expand_dims(b, 1)
                    res = npnx.matmul(a, b)
                    res2 = res[:, :size, :]
                    return npnx.transpose(res2, (1, 0, 3, 2))
                else:
                    a = cst[:, :, :fft_length]
                    b = x[:, :fft_length, :]
                    a = npnx.expand_dims(a, 0)
                    b = npnx.expand_dims(b, 1)
                    res = npnx.matmul(a, b)
                    return npnx.transpose(res, (1, 0, 2, 3))

            def onnx_rfft_3d_2d(x, fft_length):
                mat = x[:, :fft_length[-2], :fft_length[-1]]

                # first FFT
                res = onnx_rfft_3d_1d(mat, fft_length[-1], transpose=True)

                # second FFT decomposed on FFT on real part and imaginary part
                res2_real = onnx_rfft_3d_1d(res[0], fft_length[0], transpose=False)
                res2_imag = onnx_rfft_3d_1d(res[1], fft_length[0], transpose=False)
                res2_imag2 = npnx.vstack(-res2_imag[1:2], res2_imag[:1])
                res = res2_real + res2_imag2
                size = fft_length[1] // 2 + 1
                return res[:, :, :fft_length[-2], :size]

            @onnxnumpy_np(signature=NDArrayType(("T:all", np.int64), dtypes_out=('T',)))
            def onnx_rfft_2d_any_test(x, fft_length):
                new_shape = npnx.concat(
                    np.array([-1], dtype=np.int64), x.shape[-2:], axis=0)
                mat2 = x.reshape(new_shape)
                f2 = onnx_rfft_3d_2d(mat2, fft_length)
                new_shape = npnx.concat(
                    np.array([2], dtype=np.int64), x.shape[:-2], f2.shape[-2:])
                return f2.reshape(new_shape)

            shape = (3, 1, 4)
            fft_length = np.array([1, 4], dtype=np.int64)
            rnd = np.random.randn(*list(shape)).astype(np.float32)
            fft2d_cus = np.fft.fft2(rnd, fft_length)
            fft2d_onx = onnx_rfft_2d_any_test(rnd, fft_length)
            assert_almost_equal(fft2d_cus[..., :fft2d_onx.shape[-1]], fft2d_onx)

            key = list(onnx_rfft_2d_any_test.signed_compiled)[0]
            onx = onnx_rfft_2d_any_test.signed_compiled[key].compiled.onnx_
            code = export2tf2onnx(onx, name="FFT2D")
            print(code)
        """
        consumers = ctx.find_output_consumers(node.output[0])
        consumer_types = set(op.type for op in consumers)
        utils.make_sure(
            consumer_types == {'ComplexAbs'},
            "Current implementation of RFFT2D only allows ComplexAbs as consumer not %r",
            consumer_types)

        oldnode = node
        fft_length = node.input[1]
        onnx_dtype = ctx.get_dtype(node.input[0])
        np_dtype = utils.map_onnx_to_numpy_type(onnx_dtype)
        utils.make_sure(onnx_dtype in CommonFFT2DOp.supported_dtypes,
                        "Unsupported input type.")

        fft_length_node = ctx.make_node(
            'Cast', inputs=[fft_length], attr={'to': onnx_pb.TensorProto.INT64},
            name=utils.make_name('fft_length_cast'))
        varx = {"x": node.input[0], "fft_length": fft_length_node.output[0]}

        # initializers
        value = np.array(0, dtype=np.int64)
        varx['Ga_Gathercst'] = ctx.make_const(name=utils.make_name('init_Ga_Gathercst'), np_val=value).name

        value = np.array([-1], dtype=np.int64)
        varx['Re_Reshapecst'] = ctx.make_const(name=utils.make_name('init_Re_Reshapecst'), np_val=value).name

        value = np.array([0], dtype=np.int64)
        varx['Cu_CumSumcst'] = ctx.make_const(name=utils.make_name('init_Cu_CumSumcst'), np_val=value).name

        value = np.array([1], dtype=np.int64)
        varx['Co_Concatcst'] = ctx.make_const(name=utils.make_name('init_Co_Concatcst'), np_val=value).name

        value = np.array([-6.2831854820251465], dtype=np_dtype)
        varx['Id_Identitycst'] = ctx.make_const(name=utils.make_name('init_Id_Identitycst'), np_val=value).name

        value = np.array([2], dtype=np.int64)
        varx['Sl_Slicecst1'] = ctx.make_const(name=utils.make_name('init_Sl_Slicecst1'), np_val=value).name

        value = np.array(-1, dtype=np.int64)
        varx['Ga_Gathercst1'] = ctx.make_const(name=utils.make_name('init_Ga_Gathercst1'), np_val=value).name

        value = np.array([-2], dtype=np.int64)
        varx['Sl_Slicecst4'] = ctx.make_const(name=utils.make_name('init_Sl_Slicecst4'), np_val=value).name

        value = np.array([0, 0], dtype=np.int64)
        varx['Sl_Slicecst6'] = ctx.make_const(name=utils.make_name('init_Sl_Slicecst6'), np_val=value).name

        value = np.array(-2, dtype=np.int64)
        varx['Ga_Gathercst3'] = ctx.make_const(name=utils.make_name('init_Ga_Gathercst3'), np_val=value).name

        value = np.array([1, 2], dtype=np.int64)
        varx['Sl_Slicecst7'] = ctx.make_const(name=utils.make_name('init_Sl_Slicecst7'), np_val=value).name

        value = np.array(1, dtype=np.int64)
        varx['Ga_Gathercst7'] = ctx.make_const(name=utils.make_name('init_Ga_Gathercst7'), np_val=value).name

        value = np.array([2, 3], dtype=np.int64)
        varx['Sl_Slicecst25'] = ctx.make_const(name=utils.make_name('init_Sl_Slicecst25'), np_val=value).name

        # nodes
        if getattr(ctx, 'verbose', False):
            print('[nodes] %r' % cls)

        inputs = [varx['fft_length'], varx['Ga_Gathercst']]
        node = ctx.make_node('Gather', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Ga_Gather'))
        varx['Ga_output0'] = node.output[0]

        inputs = [varx['Ga_output0'], varx['Re_Reshapecst']]
        node = ctx.make_node('Reshape', inputs=inputs, name=utils.make_name('Re_Reshape'))
        varx['Re_reshaped01'] = node.output[0]

        inputs = [varx['Re_reshaped01']]
        node = ctx.make_node('ConstantOfShape', inputs=inputs, attr=dict(value=helper.make_tensor(
            "value", onnx_pb.TensorProto.INT64, dims=[1], vals=[1])), name=utils.make_name('Co_ConstantOfShape'))
        varx['Co_output01'] = node.output[0]

        inputs = [varx['Co_output01'], varx['Cu_CumSumcst']]
        node = ctx.make_node('CumSum', inputs=inputs, name=utils.make_name('Cu_CumSum'))
        varx['Cu_y0'] = node.output[0]

        inputs = [varx['Cu_y0'], varx['Re_Reshapecst']]
        node = ctx.make_node('Add', inputs=inputs, name=utils.make_name('Ad_Add'))
        varx['Ad_C01'] = node.output[0]

        inputs = [varx['Ad_C01']]
        node = ctx.make_node('Cast', inputs=inputs, attr=dict(
            to=onnx_pb.TensorProto.FLOAT), name=utils.make_name('Ca_Cast'))
        varx['Ca_output01'] = node.output[0]

        node = GraphBuilder(ctx).make_unsqueeze({'data': varx['Ga_output0'], 'axes': [0]}, return_node=True)
        varx['Un_expanded02'] = node.output[0]

        inputs = [varx['Un_expanded02'], varx['Co_Concatcst']]
        node = ctx.make_node('Concat', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Co_Concat'))
        varx['Co_concat_result01'] = node.output[0]

        inputs = [varx['Ca_output01'], varx['Co_concat_result01']]
        node = ctx.make_node('Reshape', inputs=inputs, name=utils.make_name('Re_Reshape1'))
        varx['Re_reshaped0'] = node.output[0]

        inputs = [varx['Re_reshaped0']]
        node = ctx.make_node('Cast', inputs=inputs, attr=dict(
            to=onnx_pb.TensorProto.FLOAT), name=utils.make_name('Ca_Cast1'))
        varx['Ca_output0'] = node.output[0]

        inputs = [varx['Ca_output0'], varx['Ca_output01']]
        node = ctx.make_node('Mul', inputs=inputs, name=utils.make_name('Mu_Mul'))
        varx['Mu_C01'] = node.output[0]

        inputs = [varx['Ga_output0']]
        node = ctx.make_node('Cast', inputs=inputs, attr=dict(
            to=onnx_pb.TensorProto.FLOAT), name=utils.make_name('Ca_Cast2'))
        varx['Ca_output02'] = node.output[0]

        inputs = [varx['Mu_C01'], varx['Ca_output02']]
        node = ctx.make_node('Div', inputs=inputs, name=utils.make_name('Di_Div'))
        varx['Di_C0'] = node.output[0]

        inputs = [varx['Di_C0'], varx['Id_Identitycst']]
        node = ctx.make_node('Mul', inputs=inputs, name=utils.make_name('Mu_Mul1'))
        varx['Mu_C0'] = node.output[0]

        inputs = [varx['Mu_C0']]
        node = ctx.make_node('Cos', inputs=inputs, name=utils.make_name('Co_Cos'))
        varx['Co_output0'] = node.output[0]

        node = GraphBuilder(ctx).make_unsqueeze({'data': varx['Co_output0'], 'axes': [0]}, return_node=True)
        varx['Un_expanded01'] = node.output[0]

        inputs = [varx['Mu_C0']]
        node = ctx.make_node('Sin', inputs=inputs, name=utils.make_name('Si_Sin'))
        varx['Si_output0'] = node.output[0]

        node = GraphBuilder(ctx).make_unsqueeze({'data': varx['Si_output0'], 'axes': [0]}, return_node=True)
        varx['Un_expanded03'] = node.output[0]

        inputs = [varx['Un_expanded01'], varx['Un_expanded03']]
        node = ctx.make_node('Concat', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Co_Concat1'))
        varx['Co_concat_result0'] = node.output[0]

        inputs = [varx['Co_concat_result0'], varx['Cu_CumSumcst'], varx['Re_reshaped01'], varx['Sl_Slicecst1']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice'))
        varx['Sl_output01'] = node.output[0]

        node = GraphBuilder(ctx).make_unsqueeze({'data': varx['Sl_output01'], 'axes': [0]}, return_node=True)
        varx['Un_expanded0'] = node.output[0]

        inputs = [varx['fft_length'], varx['Ga_Gathercst1']]
        node = ctx.make_node('Gather', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Ga_Gather1'))
        varx['Ga_output03'] = node.output[0]

        inputs = [varx['Ga_output03'], varx['Re_Reshapecst']]
        node = ctx.make_node('Reshape', inputs=inputs, name=utils.make_name('Re_Reshape3'))
        varx['Re_reshaped04'] = node.output[0]

        inputs = [varx['Re_reshaped04']]
        node = ctx.make_node('ConstantOfShape', inputs=inputs, attr=dict(value=helper.make_tensor(
            "value", onnx_pb.TensorProto.INT64, dims=[1], vals=[1])), name=utils.make_name('Co_ConstantOfShape1'))
        varx['Co_output04'] = node.output[0]

        inputs = [varx['Co_output04'], varx['Cu_CumSumcst']]
        node = ctx.make_node('CumSum', inputs=inputs, name=utils.make_name('Cu_CumSum1'))
        varx['Cu_y01'] = node.output[0]

        inputs = [varx['Cu_y01'], varx['Re_Reshapecst']]
        node = ctx.make_node('Add', inputs=inputs, name=utils.make_name('Ad_Add1'))
        varx['Ad_C02'] = node.output[0]

        inputs = [varx['Ad_C02']]
        node = ctx.make_node('Cast', inputs=inputs, attr=dict(
            to=onnx_pb.TensorProto.FLOAT), name=utils.make_name('Ca_Cast3'))
        varx['Ca_output04'] = node.output[0]

        node = GraphBuilder(ctx).make_unsqueeze({'data': varx['Ga_output03'], 'axes': [0]}, return_node=True)
        varx['Un_expanded08'] = node.output[0]

        inputs = [varx['Un_expanded08'], varx['Co_Concatcst']]
        node = ctx.make_node('Concat', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Co_Concat2'))
        varx['Co_concat_result04'] = node.output[0]

        inputs = [varx['Ca_output04'], varx['Co_concat_result04']]
        node = ctx.make_node('Reshape', inputs=inputs, name=utils.make_name('Re_Reshape4'))
        varx['Re_reshaped03'] = node.output[0]

        inputs = [varx['Re_reshaped03']]
        node = ctx.make_node('Cast', inputs=inputs, attr=dict(
            to=onnx_pb.TensorProto.FLOAT), name=utils.make_name('Ca_Cast4'))
        varx['Ca_output03'] = node.output[0]

        inputs = [varx['Ca_output03'], varx['Ca_output04']]
        node = ctx.make_node('Mul', inputs=inputs, name=utils.make_name('Mu_Mul2'))
        varx['Mu_C04'] = node.output[0]

        inputs = [varx['Ga_output03']]
        node = ctx.make_node('Cast', inputs=inputs, attr=dict(
            to=onnx_pb.TensorProto.FLOAT), name=utils.make_name('Ca_Cast5'))
        varx['Ca_output05'] = node.output[0]

        inputs = [varx['Mu_C04'], varx['Ca_output05']]
        node = ctx.make_node('Div', inputs=inputs, name=utils.make_name('Di_Div1'))
        varx['Di_C01'] = node.output[0]

        inputs = [varx['Di_C01'], varx['Id_Identitycst']]
        node = ctx.make_node('Mul', inputs=inputs, name=utils.make_name('Mu_Mul3'))
        varx['Mu_C03'] = node.output[0]

        inputs = [varx['Mu_C03']]
        node = ctx.make_node('Cos', inputs=inputs, name=utils.make_name('Co_Cos1'))
        varx['Co_output03'] = node.output[0]

        node = GraphBuilder(ctx).make_unsqueeze({'data': varx['Co_output03'], 'axes': [0]}, return_node=True)
        varx['Un_expanded07'] = node.output[0]

        inputs = [varx['Mu_C03']]
        node = ctx.make_node('Sin', inputs=inputs, name=utils.make_name('Si_Sin1'))
        varx['Si_output02'] = node.output[0]

        node = GraphBuilder(ctx).make_unsqueeze({'data': varx['Si_output02'], 'axes': [0]}, return_node=True)
        varx['Un_expanded09'] = node.output[0]

        inputs = [varx['Un_expanded07'], varx['Un_expanded09']]
        node = ctx.make_node('Concat', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Co_Concat3'))
        varx['Co_concat_result03'] = node.output[0]

        inputs = [varx['Co_concat_result03'], varx['Cu_CumSumcst'], varx['Re_reshaped04'], varx['Sl_Slicecst1']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice1'))
        varx['Sl_output04'] = node.output[0]

        node = GraphBuilder(ctx).make_unsqueeze({'data': varx['Sl_output04'], 'axes': [0]}, return_node=True)
        varx['Un_expanded06'] = node.output[0]

        inputs = [varx['x']]
        node = ctx.make_node('Shape', inputs=inputs, name=utils.make_name('Sh_Shape'))
        varx['Sh_shape0'] = node.output[0]

        inputs = [varx['Sh_shape0']]
        node = ctx.make_node('Shape', inputs=inputs, name=utils.make_name('Sh_Shape1'))
        varx['Sh_shape01'] = node.output[0]

        inputs = [varx['Sh_shape01'], varx['Cu_CumSumcst']]
        node = ctx.make_node('Gather', inputs=inputs, name=utils.make_name('Ga_Gather2'))
        varx['Ga_output04'] = node.output[0]

        inputs = [varx['Sh_shape0'], varx['Sl_Slicecst4'], varx['Ga_output04'], varx['Cu_CumSumcst']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice2'))
        varx['Sl_output07'] = node.output[0]

        inputs = [varx['Re_Reshapecst'], varx['Sl_output07']]
        node = ctx.make_node('Concat', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Co_Concat4'))
        varx['Co_concat_result05'] = node.output[0]

        inputs = [varx['x'], varx['Co_concat_result05']]
        node = ctx.make_node('Reshape', inputs=inputs, name=utils.make_name('Re_Reshape6'))
        varx['Re_reshaped06'] = node.output[0]

        inputs = [varx['fft_length'], varx['Ga_Gathercst3']]
        node = ctx.make_node('Gather', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Ga_Gather3'))
        varx['Ga_output05'] = node.output[0]

        inputs = [varx['Ga_output05'], varx['Re_Reshapecst']]
        node = ctx.make_node('Reshape', inputs=inputs, name=utils.make_name('Re_Reshape7'))
        varx['Re_reshaped07'] = node.output[0]

        inputs = [varx['Re_reshaped07'], varx['Re_reshaped04']]
        node = ctx.make_node('Concat', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Co_Concat5'))
        varx['Co_concat_result06'] = node.output[0]

        inputs = [varx['Re_reshaped06'], varx['Sl_Slicecst6'], varx['Co_concat_result06'], varx['Sl_Slicecst7']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice3'))
        varx['Sl_output06'] = node.output[0]

        inputs = [varx['Sl_output06']]
        node = ctx.make_node('Transpose', inputs=inputs, attr=dict(
            perm=[0, 2, 1]), name=utils.make_name('Tr_Transpose'))
        varx['Tr_transposed02'] = node.output[0]

        inputs = [varx['Tr_transposed02'], varx['Cu_CumSumcst'], varx['Re_reshaped04'], varx['Co_Concatcst']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice4'))
        varx['Sl_output05'] = node.output[0]

        node = GraphBuilder(ctx).make_unsqueeze({'data': varx['Sl_output05'], 'axes': [1]}, return_node=True)
        varx['Un_expanded010'] = node.output[0]

        inputs = [varx['Un_expanded06'], varx['Un_expanded010']]
        node = ctx.make_node('MatMul', inputs=inputs, name=utils.make_name('Ma_MatMul'))
        varx['Ma_Y01'] = node.output[0]

        inputs = [varx['Ga_output03'], varx['Sl_Slicecst1']]
        node = ctx.make_node('Div', inputs=inputs, name=utils.make_name('Di_Div2'))
        varx['Di_C02'] = node.output[0]

        inputs = [varx['Di_C02'], varx['Co_Concatcst']]
        node = ctx.make_node('Add', inputs=inputs, name=utils.make_name('Ad_Add2'))
        varx['Ad_C03'] = node.output[0]

        inputs = [varx['Ad_C03'], varx['Re_Reshapecst']]
        node = ctx.make_node('Reshape', inputs=inputs, name=utils.make_name('Re_Reshape10'))
        varx['Re_reshaped010'] = node.output[0]

        inputs = [varx['Ma_Y01'], varx['Cu_CumSumcst'], varx['Re_reshaped010'], varx['Co_Concatcst']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice5'))
        varx['Sl_output03'] = node.output[0]

        inputs = [varx['Sl_output03']]
        node = ctx.make_node('Transpose', inputs=inputs, attr=dict(
            perm=[1, 0, 3, 2]), name=utils.make_name('Tr_Transpose1'))
        varx['Tr_transposed01'] = node.output[0]

        inputs = [varx['Tr_transposed01'], varx['Ga_Gathercst']]
        node = ctx.make_node('Gather', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Ga_Gather5'))
        varx['Ga_output02'] = node.output[0]

        inputs = [varx['Ga_output02'], varx['Cu_CumSumcst'], varx['Re_reshaped01'], varx['Co_Concatcst']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice6'))
        varx['Sl_output02'] = node.output[0]

        node = GraphBuilder(ctx).make_unsqueeze({'data': varx['Sl_output02'], 'axes': [1]}, return_node=True)
        varx['Un_expanded05'] = node.output[0]

        inputs = [varx['Un_expanded0'], varx['Un_expanded05']]
        node = ctx.make_node('MatMul', inputs=inputs, name=utils.make_name('Ma_MatMul1'))
        varx['Ma_Y0'] = node.output[0]

        inputs = [varx['Ma_Y0']]
        node = ctx.make_node('Transpose', inputs=inputs, attr=dict(
            perm=[1, 0, 2, 3]), name=utils.make_name('Tr_Transpose2'))
        varx['Tr_transposed0'] = node.output[0]

        inputs = [varx['Tr_transposed01'], varx['Ga_Gathercst7']]
        node = ctx.make_node('Gather', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Ga_Gather7'))
        varx['Ga_output08'] = node.output[0]

        inputs = [varx['Ga_output08'], varx['Cu_CumSumcst'], varx['Re_reshaped01'], varx['Co_Concatcst']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice8'))
        varx['Sl_output010'] = node.output[0]

        node = GraphBuilder(ctx).make_unsqueeze({'data': varx['Sl_output010'], 'axes': [1]}, return_node=True)
        varx['Un_expanded016'] = node.output[0]

        inputs = [varx['Un_expanded0'], varx['Un_expanded016']]
        node = ctx.make_node('MatMul', inputs=inputs, name=utils.make_name('Ma_MatMul2'))
        varx['Ma_Y03'] = node.output[0]

        inputs = [varx['Ma_Y03']]
        node = ctx.make_node('Transpose', inputs=inputs, attr=dict(
            perm=[1, 0, 2, 3]), name=utils.make_name('Tr_Transpose3'))
        varx['Tr_transposed04'] = node.output[0]

        inputs = [varx['Tr_transposed04'], varx['Co_Concatcst'], varx['Sl_Slicecst1'], varx['Cu_CumSumcst']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice9'))
        varx['Sl_output08'] = node.output[0]

        inputs = [varx['Sl_output08']]
        node = ctx.make_node('Neg', inputs=inputs, name=utils.make_name('Ne_Neg'))
        varx['Ne_Y0'] = node.output[0]

        inputs = [varx['Tr_transposed04'], varx['Cu_CumSumcst'], varx['Co_Concatcst'], varx['Cu_CumSumcst']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice10'))
        varx['Sl_output012'] = node.output[0]

        inputs = [varx['Ne_Y0'], varx['Sl_output012']]
        node = ctx.make_node('Concat', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Co_Concat8'))
        varx['Co_concat_result07'] = node.output[0]

        inputs = [varx['Tr_transposed0'], varx['Co_concat_result07']]
        node = ctx.make_node('Add', inputs=inputs, name=utils.make_name('Ad_Add4'))
        varx['Ad_C0'] = node.output[0]

        inputs = [varx['fft_length'], varx['Ga_Gathercst7']]
        node = ctx.make_node('Gather', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Ga_Gather9'))
        varx['Ga_output010'] = node.output[0]

        inputs = [varx['Ga_output010'], varx['Sl_Slicecst1']]
        node = ctx.make_node('Div', inputs=inputs, name=utils.make_name('Di_Div4'))
        varx['Di_C04'] = node.output[0]

        inputs = [varx['Di_C04'], varx['Co_Concatcst']]
        node = ctx.make_node('Add', inputs=inputs, name=utils.make_name('Ad_Add5'))
        varx['Ad_C06'] = node.output[0]

        inputs = [varx['Ad_C06'], varx['Re_Reshapecst']]
        node = ctx.make_node('Reshape', inputs=inputs, name=utils.make_name('Re_Reshape17'))
        varx['Re_reshaped018'] = node.output[0]

        inputs = [varx['Re_reshaped07'], varx['Re_reshaped018']]
        node = ctx.make_node('Concat', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Co_Concat9'))
        varx['Co_concat_result010'] = node.output[0]

        inputs = [varx['Ad_C0'], varx['Sl_Slicecst6'], varx['Co_concat_result010'], varx['Sl_Slicecst25']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice11'))
        varx['Sl_output0'] = node.output[0]

        inputs = [varx['Sh_shape0'], varx['Cu_CumSumcst'], varx['Sl_Slicecst4'], varx['Cu_CumSumcst']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice12'))
        varx['Sl_output014'] = node.output[0]

        inputs = [varx['Sl_output0']]
        node = ctx.make_node('Shape', inputs=inputs, name=utils.make_name('Sh_Shape3'))
        varx['Sh_shape03'] = node.output[0]

        inputs = [varx['Sh_shape03']]
        node = ctx.make_node('Shape', inputs=inputs, name=utils.make_name('Sh_Shape4'))
        varx['Sh_shape04'] = node.output[0]

        inputs = [varx['Sh_shape04'], varx['Cu_CumSumcst']]
        node = ctx.make_node('Gather', inputs=inputs, name=utils.make_name('Ga_Gather10'))
        varx['Ga_output011'] = node.output[0]

        inputs = [varx['Sh_shape03'], varx['Sl_Slicecst4'], varx['Ga_output011'], varx['Cu_CumSumcst']]
        node = ctx.make_node('Slice', inputs=inputs, name=utils.make_name('Sl_Slice13'))
        varx['Sl_output015'] = node.output[0]

        inputs = [varx['Sl_Slicecst1'], varx['Sl_output014'], varx['Sl_output015']]
        node = ctx.make_node('Concat', inputs=inputs, attr=dict(axis=0), name=utils.make_name('Co_Concat10'))
        varx['Co_concat_result012'] = node.output[0]

        inputs = [varx['Sl_output0'], varx['Co_concat_result012']]
        node = ctx.make_node('Reshape', inputs=inputs, name=utils.make_name('Re_Reshape18'))
        varx['y'] = node.output[0]

        # finalize
        if getattr(ctx, 'verbose', False):
            print('[replace_all_inputs] %r' % cls)
        ctx.replace_all_inputs(oldnode.output[0], node.output[0])
        ctx.remove_node(oldnode.name)


@tf_op("RFFT2D")
class RFFT2DOp(CommonFFT2DOp):
    # support more dtype

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        return cls.any_version_2d(True, 1, ctx, node, **kwargs)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # Slice changed in opset 10.
        return cls.any_version_2d(True, 10, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Unsqueeze changed in opset 13.
        return cls.any_version_2d(True, 13, ctx, node, **kwargs)


@tf_op("ComplexAbs")
class ComplexAbsOp:
    # support more dtype
    supported_dtypes = [
        onnx_pb.TensorProto.FLOAT,
        onnx_pb.TensorProto.FLOAT16,
        onnx_pb.TensorProto.DOUBLE,
        onnx_pb.TensorProto.COMPLEX64,
        onnx_pb.TensorProto.COMPLEX128,
    ]

    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        """
        Computes the modules of a complex.
        If the matrix dtype is not complex64 or complex128,
        it assumes the first dimension means real part (0)
        and imaginary part (1, :, :...).
        """
        onnx_dtype = ctx.get_dtype(node.input[0])
        if onnx_dtype is None:
            # This is not the best option. onnx_dtype is unknown when the Slice operator is used.
            # onnx<=1.9.0 fails at inferring the size.
            onnx_dtype = onnx_pb.TensorProto.FLOAT
        utils.make_sure(
            onnx_dtype in ComplexAbsOp.supported_dtypes, "Unsupported input type (node.name=%r, type=%r).",
            node.input[0], onnx_dtype)
        np_dtype = utils.map_onnx_to_numpy_type(onnx_dtype)
        shape = ctx.get_shape(node.input[0])
        if shape is not None:
            utils.make_sure(
                shape[0] == 2, "ComplexAbs expected the first dimension to be 2 but shape is %r", shape)

        ind0 = ctx.make_const(name=utils.make_name('cst0'), np_val=np.array([0], dtype=np.int64))
        ind1 = ctx.make_const(name=utils.make_name('cst1'), np_val=np.array([1], dtype=np.int64))
        p2 = ctx.make_const(name=utils.make_name('p2'), np_val=np.array([2], dtype=np_dtype))

        real_part = ctx.make_node(
            'Gather', inputs=[node.input[0], ind0.name], attr=dict(axis=0),
            name=utils.make_name('Real_' + node.name))
        imag_part = ctx.make_node(
            'Gather', inputs=[node.input[0], ind1.name], attr=dict(axis=0),
            name=utils.make_name('Imag_' + node.name))

        real_part2 = ctx.make_node(
            'Pow', inputs=[real_part.output[0], p2.name],
            name=utils.make_name(real_part.name + 'p2p'))

        imag_part2 = ctx.make_node(
            'Pow', inputs=[imag_part.output[0], p2.name],
            name=utils.make_name(imag_part.name + 'p2p'))

        ctx.remove_node(node.name)
        add = ctx.make_node(
            "Add", inputs=[real_part2.output[0], imag_part2.output[0]],
            name=utils.make_name('ComplexAbs_' + node.name))

        squeezed = GraphBuilder(ctx).make_squeeze(
            {'data': add.output[0], 'axes': [0]}, name=utils.make_name('ComplexAbs' + node.name), return_node=True)

        last_node = ctx.make_node(
            "Sqrt", inputs=squeezed.output[:1],
            name=utils.make_name('ComplexAbs' + node.name))

        ctx.replace_all_inputs(node.output[0], last_node.output[0])  # ops=ctx.get_nodes()

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        cls.any_version(1, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        cls.any_version(13, ctx, node, **kwargs)
