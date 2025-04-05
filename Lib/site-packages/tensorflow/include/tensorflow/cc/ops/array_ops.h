// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_ARRAY_OPS_H_
#define TENSORFLOW_CC_OPS_ARRAY_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup array_ops Array Ops
/// @{

/// BatchToSpace for 4-D tensors of type T.
///
/// This is a legacy version of the more general BatchToSpaceND.
///
/// Rearranges (permutes) data from batch into blocks of spatial data, followed by
/// cropping. This is the reverse transformation of SpaceToBatch. More specifically,
/// this op outputs a copy of the input tensor where values from the `batch`
/// dimension are moved in spatial blocks to the `height` and `width` dimensions,
/// followed by cropping along the `height` and `width` dimensions.
///
/// Args:
/// * scope: A Scope object
/// * input: 4-D tensor with shape
/// `[batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
///   depth]`. Note that the batch size of the input tensor must be divisible by
/// `block_size * block_size`.
/// * crops: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
/// how many elements to crop from the intermediate result across the spatial
/// dimensions as follows:
///
///     crops = [[crop_top, crop_bottom], [crop_left, crop_right]]
///
/// Returns:
/// * `Output`: 4-D with shape `[batch, height, width, depth]`, where:
///
///       height = height_pad - crop_top - crop_bottom
///       width = width_pad - crop_left - crop_right
///
/// The attr `block_size` must be greater than one. It indicates the block size.
///
/// Some examples:
///
/// (1) For the following input of shape `[4, 1, 1, 1]` and block_size of 2:
///
/// ```
/// [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
/// ```
///
/// The output tensor has shape `[1, 2, 2, 1]` and value:
///
/// ```
/// x = [[[[1], [2]], [[3], [4]]]]
/// ```
///
/// (2) For the following input of shape `[4, 1, 1, 3]` and block_size of 2:
///
/// ```
/// [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
/// ```
///
/// The output tensor has shape `[1, 2, 2, 3]` and value:
///
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
///
/// (3) For the following input of shape `[4, 2, 2, 1]` and block_size of 2:
///
/// ```
/// x = [[[[1], [3]], [[9], [11]]],
///      [[[2], [4]], [[10], [12]]],
///      [[[5], [7]], [[13], [15]]],
///      [[[6], [8]], [[14], [16]]]]
/// ```
///
/// The output tensor has shape `[1, 4, 4, 1]` and value:
///
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///      [[5],   [6],  [7],  [8]],
///      [[9],  [10], [11],  [12]],
///      [[13], [14], [15],  [16]]]]
/// ```
///
/// (4) For the following input of shape `[8, 1, 2, 1]` and block_size of 2:
///
/// ```
/// x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
///      [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
/// ```
///
/// The output tensor has shape `[2, 2, 4, 1]` and value:
///
/// ```
/// x = [[[[1], [3]], [[5], [7]]],
///      [[[2], [4]], [[10], [12]]],
///      [[[5], [7]], [[13], [15]]],
///      [[[6], [8]], [[14], [16]]]]
/// ```
class BatchToSpace {
 public:
  BatchToSpace(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
             ::tensorflow::Input crops, int64 block_size);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// BatchToSpace for N-D tensors of type T.
///
/// This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
/// `block_shape + [batch]`, interleaves these blocks back into the grid defined by
/// the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
/// the input.  The spatial dimensions of this intermediate result are then
/// optionally cropped according to `crops` to produce the output.  This is the
/// reverse of SpaceToBatch.  See below for a precise description.
///
/// Args:
/// * scope: A Scope object
/// * input: N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
/// where spatial_shape has M dimensions.
/// * block_shape: 1-D with shape `[M]`, all values must be >= 1.
/// * crops: 2-D with shape `[M, 2]`, all values must be >= 0.
///   `crops[i] = [crop_start, crop_end]` specifies the amount to crop from input
///   dimension `i + 1`, which corresponds to spatial dimension `i`.  It is
///   required that
///   `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1]`.
///
/// This operation is equivalent to the following steps:
///
/// 1. Reshape `input` to `reshaped` of shape:
///      [block_shape[0], ..., block_shape[M-1],
///       batch / prod(block_shape),
///       input_shape[1], ..., input_shape[N-1]]
///
/// 2. Permute dimensions of `reshaped` to produce `permuted` of shape
///      [batch / prod(block_shape),
///
///       input_shape[1], block_shape[0],
///       ...,
///       input_shape[M], block_shape[M-1],
///
///       input_shape[M+1], ..., input_shape[N-1]]
///
/// 3. Reshape `permuted` to produce `reshaped_permuted` of shape
///      [batch / prod(block_shape),
///
///       input_shape[1] * block_shape[0],
///       ...,
///       input_shape[M] * block_shape[M-1],
///
///       input_shape[M+1],
///       ...,
///       input_shape[N-1]]
///
/// 4. Crop the start and end of dimensions `[1, ..., M]` of
///    `reshaped_permuted` according to `crops` to produce the output of shape:
///      [batch / prod(block_shape),
///
///       input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
///       ...,
///       input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],
///
///       input_shape[M+1], ..., input_shape[N-1]]
///
/// Some examples:
///
/// (1) For the following input of shape `[4, 1, 1, 1]`, `block_shape = [2, 2]`, and
///     `crops = [[0, 0], [0, 0]]`:
///
/// ```
/// [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
/// ```
///
/// The output tensor has shape `[1, 2, 2, 1]` and value:
///
/// ```
/// x = [[[[1], [2]], [[3], [4]]]]
/// ```
///
/// (2) For the following input of shape `[4, 1, 1, 3]`, `block_shape = [2, 2]`, and
///     `crops = [[0, 0], [0, 0]]`:
///
/// ```
/// [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
/// ```
///
/// The output tensor has shape `[1, 2, 2, 3]` and value:
///
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
///
/// (3) For the following input of shape `[4, 2, 2, 1]`, `block_shape = [2, 2]`, and
///     `crops = [[0, 0], [0, 0]]`:
///
/// ```
/// x = [[[[1], [3]], [[9], [11]]],
///      [[[2], [4]], [[10], [12]]],
///      [[[5], [7]], [[13], [15]]],
///      [[[6], [8]], [[14], [16]]]]
/// ```
///
/// The output tensor has shape `[1, 4, 4, 1]` and value:
///
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///      [[5],   [6],  [7],  [8]],
///      [[9],  [10], [11],  [12]],
///      [[13], [14], [15],  [16]]]]
/// ```
///
/// (4) For the following input of shape `[8, 1, 3, 1]`, `block_shape = [2, 2]`, and
///     `crops = [[0, 0], [2, 0]]`:
///
/// ```
/// x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
///      [[[0], [2], [4]]], [[[0], [10], [12]]],
///      [[[0], [5], [7]]], [[[0], [13], [15]]],
///      [[[0], [6], [8]]], [[[0], [14], [16]]]]
/// ```
///
/// The output tensor has shape `[2, 2, 4, 1]` and value:
///
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///       [[5],   [6],  [7],  [8]]],
///      [[[9],  [10], [11],  [12]],
///       [[13], [14], [15],  [16]]]]
/// ```
///
/// Returns:
/// * `Output`: The output tensor.
class BatchToSpaceND {
 public:
  BatchToSpaceND(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
               ::tensorflow::Input block_shape, ::tensorflow::Input crops);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Bitcasts a tensor from one type to another without copying data.
///
/// Given a tensor `input`, this operation returns a tensor that has the same buffer
/// data as `input` with datatype `type`.
///
/// If the input datatype `T` is larger than the output datatype `type` then the
/// shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].
///
/// If `T` is smaller than `type`, the operator requires that the rightmost
/// dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
/// [..., sizeof(`type`)/sizeof(`T`)] to [...].
///
/// tf.bitcast() and tf.cast() work differently when real dtype is casted as a complex dtype
/// (e.g. tf.complex64 or tf.complex128) as tf.cast() make imaginary part 0 while tf.bitcast()
/// gives module error.
/// For example,
///
/// Example 1:
///
/// >>> a = [1., 2., 3.]
/// >>> equality_bitcast = tf.bitcast(a, tf.complex128)
/// Traceback (most recent call last):
/// ...
/// InvalidArgumentError: Cannot bitcast from 1 to 18 [Op:Bitcast]
/// >>> equality_cast = tf.cast(a, tf.complex128)
/// >>> print(equality_cast)
/// tf.Tensor([1.+0.j 2.+0.j 3.+0.j], shape=(3,), dtype=complex128)
///
/// Example 2:
///
/// >>> tf.bitcast(tf.constant(0xffffffff, dtype=tf.uint32), tf.uint8)
/// <tf.Tensor: shape=(4,), dtype=uint8, numpy=array([255, 255, 255, 255], dtype=uint8)>
///
/// Example 3:
///
/// >>> x = [1., 2., 3.]
/// >>> y = [0., 2., 3.]
/// >>> equality= tf.equal(x,y)
/// >>> equality_cast = tf.cast(equality,tf.float32)
/// >>> equality_bitcast = tf.bitcast(equality_cast,tf.uint8)
/// >>> print(equality)
/// tf.Tensor([False True True], shape=(3,), dtype=bool)
/// >>> print(equality_cast)
/// tf.Tensor([0. 1. 1.], shape=(3,), dtype=float32)
/// >>> print(equality_bitcast)
/// tf.Tensor(
///     [[  0   0   0   0]
///      [  0   0 128  63]
///      [  0   0 128  63]], shape=(3, 4), dtype=uint8)
///
/// *NOTE*: Bitcast is implemented as a low-level cast, so machines with different
/// endian orderings will give different results. A copy from input buffer to output
/// buffer is made on BE machines when types are of different sizes in order to get
/// the same casting results as on LE machines.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Bitcast {
 public:
  Bitcast(const ::tensorflow::Scope& scope, ::tensorflow::Input input, DataType
        type);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Return the shape of s0 op s1 with broadcast.
///
/// Given `s0` and `s1`, tensors that represent shapes, compute `r0`, the
/// broadcasted shape. `s0`, `s1` and `r0` are all integer vectors.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The r0 tensor.
class BroadcastDynamicShape {
 public:
  BroadcastDynamicShape(const ::tensorflow::Scope& scope, ::tensorflow::Input s0,
                      ::tensorflow::Input s1);
  operator ::tensorflow::Output() const { return r0; }
  operator ::tensorflow::Input() const { return r0; }
  ::tensorflow::Node* node() const { return r0.node(); }

  Operation operation;
  ::tensorflow::Output r0;
};

/// Broadcast an array for a compatible shape.
///
/// Broadcasting is the process of making arrays to have compatible shapes
/// for arithmetic operations. Two shapes are compatible if for each
/// dimension pair they are either equal or one of them is one.
///
/// For example:
///
/// >>> x = tf.constant([[1, 2, 3]])   # Shape (1, 3,)
/// >>> y = tf.broadcast_to(x, [2, 3])
/// >>> print(y)
/// tf.Tensor(
///     [[1 2 3]
///      [1 2 3]], shape=(2, 3), dtype=int32)
///
/// In the above example, the input Tensor with the shape of `[1, 3]`
/// is broadcasted to output Tensor with shape of `[2, 3]`.
///
/// When broadcasting, if a tensor has fewer axes than necessary its shape is
/// padded on the left with ones. So this gives the same result as the previous
/// example:
///
/// >>> x = tf.constant([1, 2, 3])   # Shape (3,)
/// >>> y = tf.broadcast_to(x, [2, 3])
///
///
/// When doing broadcasted operations such as multiplying a tensor
/// by a scalar, broadcasting (usually) confers some time or space
/// benefit, as the broadcasted tensor is never materialized.
///
/// However, `broadcast_to` does not carry with it any such benefits.
/// The newly-created tensor takes the full memory of the broadcasted
/// shape. (In a graph context, `broadcast_to` might be fused to
/// subsequent operation and then be optimized away, however.)
///
/// Args:
/// * scope: A Scope object
/// * input: A Tensor to broadcast.
/// * shape: An 1-D `int` Tensor. The shape of the desired output.
///
/// Returns:
/// * `Output`: A Tensor.
class BroadcastTo {
 public:
  BroadcastTo(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
            ::tensorflow::Input shape);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Checks a tensor for NaN and Inf values.
///
/// When run, reports an `InvalidArgument` error if `tensor` has any values
/// that are not a number (NaN) or infinity (Inf). Otherwise, returns the input
/// tensor.
///
/// Example usage:
///
/// ``` python
/// a = tf.Variable(1.0)
/// tf.debugging.check_numerics(a, message='')
///
/// b = tf.Variable(np.nan)
/// try:
///   tf.debugging.check_numerics(b, message='Checking b')
/// except Exception as e:
///   assert "Checking b : Tensor had NaN values" in e.message
///
/// c = tf.Variable(np.inf)
/// try:
///   tf.debugging.check_numerics(c, message='Checking c')
/// except Exception as e:
///   assert "Checking c : Tensor had Inf values" in e.message
/// ```
///
///
/// Args:
/// * scope: A Scope object
/// * message: Prefix of the error message.
///
/// Returns:
/// * `Output`: The output tensor.
class CheckNumerics {
 public:
  CheckNumerics(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
              StringPiece message);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Concatenates tensors along one dimension.
///
/// Args:
/// * scope: A Scope object
/// * values: List of `N` Tensors to concatenate. Their ranks and types must match,
/// and their sizes must match in all dimensions except `concat_dim`.
/// * axis: 0-D.  The dimension along which to concatenate.  Must be in the
/// range [-rank(values), rank(values)).
///
/// Returns:
/// * `Output`: A `Tensor` with the concatenation of values stacked along the
/// `concat_dim` dimension.  This tensor's shape matches that of `values` except
/// in `concat_dim` where it has the sum of the sizes.
class Concat {
 public:
  Concat(const ::tensorflow::Scope& scope, ::tensorflow::InputList values,
       ::tensorflow::Input axis);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Shuffle dimensions of x according to a permutation and conjugate the result.
///
/// The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
///   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
///   `y[i,j,k,...,s,t,u] == conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]])`
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class ConjugateTranspose {
 public:
  ConjugateTranspose(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
                   ::tensorflow::Input perm);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Identity op for gradient debugging.
///
/// This op is hidden from public in Python. It is used by TensorFlow Debugger to
/// register gradient tensors for gradient debugging.
/// This op operates on non-reference-type tensors.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class DebugGradientIdentity {
 public:
  DebugGradientIdentity(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Identity op for gradient debugging.
///
/// This op is hidden from public in Python. It is used by TensorFlow Debugger to
/// register gradient tensors for gradient debugging.
/// This op operates on reference-type tensors.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class DebugGradientRefIdentity {
 public:
  DebugGradientRefIdentity(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Makes a copy of `x`.
///
/// Args:
/// * scope: A Scope object
/// * x: The source tensor of type `T`.
///
/// Returns:
/// * `Output`:     y: A `Tensor` of type `T`. A copy of `x`. Guaranteed that `y`
///       is not an alias of `x`.
class DeepCopy {
 public:
  DeepCopy(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// DepthToSpace for tensors of type T.
///
/// Rearranges data from depth into blocks of spatial data.
/// This is the reverse transformation of SpaceToDepth. More specifically,
/// this op outputs a copy of the input tensor where values from the `depth`
/// dimension are moved in spatial blocks to the `height` and `width` dimensions.
/// The attr `block_size` indicates the input block size and how the data is moved.
///
///   * Chunks of data of size `block_size * block_size` from depth are rearranged
///     into non-overlapping blocks of size `block_size x block_size`
///   * The width of the output tensor is `input_depth * block_size`, whereas the
///     height is `input_height * block_size`.
///   * The Y, X coordinates within each block of the output image are determined
///     by the high order component of the input channel index.
///   * The depth of the input tensor must be divisible by
///     `block_size * block_size`.
///
/// The `data_format` attr specifies the layout of the input and output tensors
/// with the following options:
///   "NHWC": `[ batch, height, width, channels ]`
///   "NCHW": `[ batch, channels, height, width ]`
///   "NCHW_VECT_C":
///       `qint8 [ batch, channels / 4, height, width, 4 ]`
///
/// It is useful to consider the operation as transforming a 6-D Tensor.
/// e.g. for data_format = NHWC,
///      Each element in the input tensor can be specified via 6 coordinates,
///      ordered by decreasing memory layout significance as:
///      n,iY,iX,bY,bX,oC  (where n=batch index, iX, iY means X or Y coordinates
///                         within the input image, bX, bY means coordinates
///                         within the output block, oC means output channels).
///      The output would be the input transposed to the following layout:
///      n,iY,bY,iX,bX,oC
///
/// This operation is useful for resizing the activations between convolutions
/// (but keeping all data), e.g. instead of pooling. It is also useful for training
/// purely convolutional models.
///
/// For example, given an input of shape `[1, 1, 1, 4]`, data_format = "NHWC" and
/// block_size = 2:
///
/// ```
/// x = [[[[1, 2, 3, 4]]]]
///
/// ```
///
/// This operation will output a tensor of shape `[1, 2, 2, 1]`:
///
/// ```
///    [[[[1], [2]],
///      [[3], [4]]]]
/// ```
///
/// Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
/// the corresponding output will have 2x2 elements and will have a depth of
/// 1 channel (1 = `4 / (block_size * block_size)`).
/// The output element shape is `[2, 2, 1]`.
///
/// For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.
///
/// ```
/// x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
/// ```
///
/// This operation, for block size of 2, will return the following tensor of shape
/// `[1, 2, 2, 3]`
///
/// ```
///    [[[[1, 2, 3], [4, 5, 6]],
///      [[7, 8, 9], [10, 11, 12]]]]
///
/// ```
///
/// Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:
///
/// ```
/// x =  [[[[1, 2, 3, 4],
///        [5, 6, 7, 8]],
///       [[9, 10, 11, 12],
///        [13, 14, 15, 16]]]]
/// ```
///
/// the operator will return the following tensor of shape `[1 4 4 1]`:
///
/// ```
/// x = [[[ [1],   [2],  [5],  [6]],
///       [ [3],   [4],  [7],  [8]],
///       [ [9],  [10], [13],  [14]],
///       [ [11], [12], [15],  [16]]]]
///
/// ```
///
/// Args:
/// * scope: A Scope object
/// * block_size: The size of the spatial block, same as in Space2Depth.
///
/// Returns:
/// * `Output`: The output tensor.
class DepthToSpace {
 public:
  /// Optional attribute setters for DepthToSpace
  struct Attrs {
    /// Defaults to "NHWC"
    TF_MUST_USE_RESULT Attrs DataFormat(StringPiece x) {
      Attrs ret = *this;
      ret.data_format_ = x;
      return ret;
    }

    StringPiece data_format_ = "NHWC";
  };
  DepthToSpace(const ::tensorflow::Scope& scope, ::tensorflow::Input input, int64
             block_size);
  DepthToSpace(const ::tensorflow::Scope& scope, ::tensorflow::Input input, int64
             block_size, const DepthToSpace::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs DataFormat(StringPiece x) {
    return Attrs().DataFormat(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Dequantize the 'input' tensor into a float or bfloat16 Tensor.
///
/// [min_range, max_range] are scalar floats that specify the range for
/// the output. The 'mode' attribute controls exactly which calculations are
/// used to convert the float values to their quantized equivalents.
///
/// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
///
/// ```
/// if T == qint8: in[i] += (range(T) + 1)/ 2.0
/// out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
/// ```
/// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
///
/// *MIN_COMBINED Mode Example*
///
/// If the input comes from a QuantizedRelu6, the output type is
/// quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
/// 0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
/// Dequantize on quint8 will take each value, cast to float, and multiply
/// by 6 / 255.
/// Note that if quantizedtype is qint8, the operation will additionally add
/// each value by 128 prior to casting.
///
/// If the mode is 'MIN_FIRST', then this approach is used:
///
/// ```c++
/// num_discrete_values = 1 << (# of bits in T)
/// range_adjust = num_discrete_values / (num_discrete_values - 1)
/// range = (range_max - range_min) * range_adjust
/// range_scale = range / num_discrete_values
/// const double offset_input = static_cast<double>(input) - lowest_quantized;
/// result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
/// ```
///
/// If the mode is `SCALED`, dequantization is performed by multiplying each
/// input value by a scaling_factor. (Thus an input of 0 always maps to 0.0).
///
/// The scaling_factor is determined from `min_range`, `max_range`, and
/// `narrow_range` in a way that is compatible with `QuantizeAndDequantize{V2|V3}`
/// and `QuantizeV2`, using the following algorithm:
///
/// ```c++
///
///   const int min_expected_T = std::numeric_limits<T>::min() +
///     (narrow_range ? 1 : 0);
///   const int max_expected_T = std::numeric_limits<T>::max();
///   const float max_expected_T = std::numeric_limits<float>::max();
///
///   const float scale_factor =
///     (std::numeric_limits<T>::min() == 0) ? (max_range / max_expected_T)
///                                          : std::max(min_range / min_expected_T,
///                                                     max_range / max_expected_T);
/// ```
///
/// Args:
/// * scope: A Scope object
/// * min_range: The minimum scalar value possibly produced for the input.
/// * max_range: The maximum scalar value possibly produced for the input.
///
/// Optional attributes (see `Attrs`):
/// * dtype: Type of the output tensor. Currently Dequantize supports float and bfloat16.
/// If 'dtype' is 'bfloat16', it only supports 'MIN_COMBINED' mode.
///
/// Returns:
/// * `Output`: The output tensor.
class Dequantize {
 public:
  /// Optional attribute setters for Dequantize
  struct Attrs {
    /// Defaults to "MIN_COMBINED"
    TF_MUST_USE_RESULT Attrs Mode(StringPiece x) {
      Attrs ret = *this;
      ret.mode_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs NarrowRange(bool x) {
      Attrs ret = *this;
      ret.narrow_range_ = x;
      return ret;
    }

    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Axis(int64 x) {
      Attrs ret = *this;
      ret.axis_ = x;
      return ret;
    }

    /// Type of the output tensor. Currently Dequantize supports float and bfloat16.
    /// If 'dtype' is 'bfloat16', it only supports 'MIN_COMBINED' mode.
    ///
    /// Defaults to DT_FLOAT
    TF_MUST_USE_RESULT Attrs Dtype(DataType x) {
      Attrs ret = *this;
      ret.dtype_ = x;
      return ret;
    }

    StringPiece mode_ = "MIN_COMBINED";
    bool narrow_range_ = false;
    int64 axis_ = -1;
    DataType dtype_ = DT_FLOAT;
  };
  Dequantize(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
           ::tensorflow::Input min_range, ::tensorflow::Input max_range);
  Dequantize(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
           ::tensorflow::Input min_range, ::tensorflow::Input max_range, const
           Dequantize::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Mode(StringPiece x) {
    return Attrs().Mode(x);
  }
  static Attrs NarrowRange(bool x) {
    return Attrs().NarrowRange(x);
  }
  static Attrs Axis(int64 x) {
    return Attrs().Axis(x);
  }
  static Attrs Dtype(DataType x) {
    return Attrs().Dtype(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns a diagonal tensor with a given diagonal values.
///
/// Given a `diagonal`, this operation returns a tensor with the `diagonal` and
/// everything else padded with zeros. The diagonal is computed as follows:
///
/// Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
/// rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
///
/// `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.
///
/// For example:
///
/// ```
/// # 'diagonal' is [1, 2, 3, 4]
/// tf.diag(diagonal) ==> [[1, 0, 0, 0]
///                        [0, 2, 0, 0]
///                        [0, 0, 3, 0]
///                        [0, 0, 0, 4]]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * diagonal: Rank k tensor where k is at most 1.
///
/// Returns:
/// * `Output`: The output tensor.
class Diag {
 public:
  Diag(const ::tensorflow::Scope& scope, ::tensorflow::Input diagonal);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns the diagonal part of the tensor.
///
/// This operation returns a tensor with the `diagonal` part
/// of the `input`. The `diagonal` part is computed as follows:
///
/// Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
/// tensor of rank `k` with dimensions `[D1,..., Dk]` where:
///
/// `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.
///
/// For example:
///
/// ```
/// # 'input' is [[1, 0, 0, 0]
///               [0, 2, 0, 0]
///               [0, 0, 3, 0]
///               [0, 0, 0, 4]]
///
/// tf.diag_part(input) ==> [1, 2, 3, 4]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: Rank k tensor where k is even and not zero.
///
/// Returns:
/// * `Output`: The extracted diagonal.
class DiagPart {
 public:
  DiagPart(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return diagonal; }
  operator ::tensorflow::Input() const { return diagonal; }
  ::tensorflow::Node* node() const { return diagonal.node(); }

  Operation operation;
  ::tensorflow::Output diagonal;
};

/// Computes the (possibly normalized) Levenshtein Edit Distance.
///
/// The inputs are variable-length sequences provided by SparseTensors
///   (hypothesis_indices, hypothesis_values, hypothesis_shape)
/// and
///   (truth_indices, truth_values, truth_shape).
///
/// The inputs are:
///
/// Args:
/// * scope: A Scope object
/// * hypothesis_indices: The indices of the hypothesis list SparseTensor.
/// This is an N x R int64 matrix.
/// * hypothesis_values: The values of the hypothesis list SparseTensor.
/// This is an N-length vector.
/// * hypothesis_shape: The shape of the hypothesis list SparseTensor.
/// This is an R-length vector.
/// * truth_indices: The indices of the truth list SparseTensor.
/// This is an M x R int64 matrix.
/// * truth_values: The values of the truth list SparseTensor.
/// This is an M-length vector.
/// * truth_shape: truth indices, vector.
///
/// Optional attributes (see `Attrs`):
/// * normalize: boolean (if true, edit distances are normalized by length of truth).
///
/// The output is:
///
/// Returns:
/// * `Output`: A dense float tensor with rank R - 1.
///
/// For the example input:
///
///     // hypothesis represents a 2x1 matrix with variable-length values:
///     //   (0,0) = ["a"]
///     //   (1,0) = ["b"]
///     hypothesis_indices = [[0, 0, 0],
///                           [1, 0, 0]]
///     hypothesis_values = ["a", "b"]
///     hypothesis_shape = [2, 1, 1]
///
///     // truth represents a 2x2 matrix with variable-length values:
///     //   (0,0) = []
///     //   (0,1) = ["a"]
///     //   (1,0) = ["b", "c"]
///     //   (1,1) = ["a"]
///     truth_indices = [[0, 1, 0],
///                      [1, 0, 0],
///                      [1, 0, 1],
///                      [1, 1, 0]]
///     truth_values = ["a", "b", "c", "a"]
///     truth_shape = [2, 2, 2]
///     normalize = true
///
/// The output will be:
///
///     // output is a 2x2 matrix with edit distances normalized by truth lengths.
///     output = [[inf, 1.0],  // (0,0): no truth, (0,1): no hypothesis
///               [0.5, 1.0]]  // (1,0): addition, (1,1): no hypothesis
class EditDistance {
 public:
  /// Optional attribute setters for EditDistance
  struct Attrs {
    /// boolean (if true, edit distances are normalized by length of truth).
    ///
    /// The output is:
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs Normalize(bool x) {
      Attrs ret = *this;
      ret.normalize_ = x;
      return ret;
    }

    bool normalize_ = true;
  };
  EditDistance(const ::tensorflow::Scope& scope, ::tensorflow::Input
             hypothesis_indices, ::tensorflow::Input hypothesis_values,
             ::tensorflow::Input hypothesis_shape, ::tensorflow::Input
             truth_indices, ::tensorflow::Input truth_values,
             ::tensorflow::Input truth_shape);
  EditDistance(const ::tensorflow::Scope& scope, ::tensorflow::Input
             hypothesis_indices, ::tensorflow::Input hypothesis_values,
             ::tensorflow::Input hypothesis_shape, ::tensorflow::Input
             truth_indices, ::tensorflow::Input truth_values,
             ::tensorflow::Input truth_shape, const EditDistance::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Normalize(bool x) {
    return Attrs().Normalize(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Creates a tensor with the given shape.
///
/// This operation creates a tensor of `shape` and `dtype`.
///
/// Args:
/// * scope: A Scope object
/// * shape: 1-D. Represents the shape of the output tensor.
///
/// Optional attributes (see `Attrs`):
/// * init: If True, initialize the returned tensor with the default value of dtype.  Otherwise, the implementation is free not to initializethe tensor's content.
///
/// Returns:
/// * `Output`: A `Tensor` of type `T`.
class Empty {
 public:
  /// Optional attribute setters for Empty
  struct Attrs {
    /// If True, initialize the returned tensor with the default value of dtype.  Otherwise, the implementation is free not to initializethe tensor's content.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs Init(bool x) {
      Attrs ret = *this;
      ret.init_ = x;
      return ret;
    }

    bool init_ = false;
  };
  Empty(const ::tensorflow::Scope& scope, ::tensorflow::Input shape, DataType
      dtype);
  Empty(const ::tensorflow::Scope& scope, ::tensorflow::Input shape, DataType
      dtype, const Empty::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Init(bool x) {
    return Attrs().Init(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Ensures that the tensor's shape matches the expected shape.
///
/// Raises an error if the input tensor's shape does not match the specified shape.
/// Returns the input tensor otherwise.
///
/// Args:
/// * scope: A Scope object
/// * input: A tensor, whose shape is to be validated.
/// * shape: The expected (possibly partially specified) shape of the input tensor.
///
/// Returns:
/// * `Output`: A tensor with the same shape and contents as the input tensor or value.
class EnsureShape {
 public:
  EnsureShape(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
            PartialTensorShape shape);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Inserts a dimension of 1 into a tensor's shape.
///
/// Given a tensor `input`, this operation inserts a dimension of 1 at the
/// dimension index `axis` of `input`'s shape. The dimension index `axis` starts at
/// zero; if you specify a negative number for `axis` it is counted backward from
/// the end.
///
/// This operation is useful if you want to add a batch dimension to a single
/// element. For example, if you have a single image of shape `[height, width,
/// channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
/// which will make the shape `[1, height, width, channels]`.
///
/// Other examples:
///
/// ```
/// # 't' is a tensor of shape [2]
/// shape(expand_dims(t, 0)) ==> [1, 2]
/// shape(expand_dims(t, 1)) ==> [2, 1]
/// shape(expand_dims(t, -1)) ==> [2, 1]
///
/// # 't2' is a tensor of shape [2, 3, 5]
/// shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
/// shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
/// shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
/// ```
///
/// This operation requires that:
///
/// `-1-input.dims() <= dim <= input.dims()`
///
/// This operation is related to `squeeze()`, which removes dimensions of
/// size 1.
///
/// Args:
/// * scope: A Scope object
/// * axis: 0-D (scalar). Specifies the dimension index at which to
/// expand the shape of `input`. Must be in the range
/// `[-rank(input) - 1, rank(input)]`.
///
/// Returns:
/// * `Output`: Contains the same data as `input`, but its shape has an additional
/// dimension of size 1 added.
class ExpandDims {
 public:
  ExpandDims(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
           ::tensorflow::Input axis);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Extract `patches` from `images` and put them in the "depth" output dimension.
///
/// Args:
/// * scope: A Scope object
/// * images: 4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.
/// * ksizes: The size of the sliding window for each dimension of `images`.
/// * strides: How far the centers of two consecutive patches are in
/// the images. Must be: `[1, stride_rows, stride_cols, 1]`.
/// * rates: Must be: `[1, rate_rows, rate_cols, 1]`. This is the
/// input stride, specifying how far two consecutive patch samples are in the
/// input. Equivalent to extracting patches with
/// `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by
/// subsampling them spatially by a factor of `rates`. This is equivalent to
/// `rate` in dilated (a.k.a. Atrous) convolutions.
/// * padding: The type of padding algorithm to use.
///
/// Returns:
/// * `Output`: 4-D Tensor with shape `[batch, out_rows, out_cols, ksize_rows *
/// ksize_cols * depth]` containing image patches with size
/// `ksize_rows x ksize_cols x depth` vectorized in the "depth" dimension. Note
/// `out_rows` and `out_cols` are the dimensions of the output patches.
class ExtractImagePatches {
 public:
  ExtractImagePatches(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    images, const gtl::ArraySlice<int>& ksizes, const
                    gtl::ArraySlice<int>& strides, const gtl::ArraySlice<int>&
                    rates, StringPiece padding);
  operator ::tensorflow::Output() const { return patches; }
  operator ::tensorflow::Input() const { return patches; }
  ::tensorflow::Node* node() const { return patches.node(); }

  Operation operation;
  ::tensorflow::Output patches;
};

/// Extract `patches` from `input` and put them in the `"depth"` output dimension. 3D extension of `extract_image_patches`.
///
/// Args:
/// * scope: A Scope object
/// * input: 5-D Tensor with shape `[batch, in_planes, in_rows, in_cols, depth]`.
/// * ksizes: The size of the sliding window for each dimension of `input`.
/// * strides: 1-D of length 5. How far the centers of two consecutive patches are in
/// `input`. Must be: `[1, stride_planes, stride_rows, stride_cols, 1]`.
/// * padding: The type of padding algorithm to use.
///
/// The size-related attributes are specified as follows:
///
/// ```python
/// ksizes = [1, ksize_planes, ksize_rows, ksize_cols, 1]
/// strides = [1, stride_planes, strides_rows, strides_cols, 1]
/// ```
///
/// Returns:
/// * `Output`: 5-D Tensor with shape `[batch, out_planes, out_rows, out_cols,
/// ksize_planes * ksize_rows * ksize_cols * depth]` containing patches
/// with size `ksize_planes x ksize_rows x ksize_cols x depth` vectorized
/// in the "depth" dimension. Note `out_planes`, `out_rows` and `out_cols`
/// are the dimensions of the output patches.
class ExtractVolumePatches {
 public:
  ExtractVolumePatches(const ::tensorflow::Scope& scope, ::tensorflow::Input
                     input, const gtl::ArraySlice<int>& ksizes, const
                     gtl::ArraySlice<int>& strides, StringPiece padding);
  operator ::tensorflow::Output() const { return patches; }
  operator ::tensorflow::Input() const { return patches; }
  ::tensorflow::Node* node() const { return patches.node(); }

  Operation operation;
  ::tensorflow::Output patches;
};

/// Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same shape and type.
///
///
///   Quantization is called fake since the output is still in floating point.
///   The API converts inputs into values within the range [min and max] and returns
///   as output.
///
/// Attributes
///
/// *   `[min; max]` define the clamping range for the `inputs` data.
/// *   `inputs` values are quantized into the quantization range (
/// `[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
/// when it is true) and then de-quantized and output as floats in `[min; max]`
/// interval.
/// *   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
///
/// Before quantization, `min` and `max` values are adjusted with the following
/// logic.
/// It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
/// the behavior can be unexpected:
///
/// *   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
/// *   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
/// *   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
/// `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.
///
///
/// Examples
///
/// ```python
///
/// inp = tf.constant ([10.03, -10.23, 3])
/// out = tf.quantization.fake_quant_with_min_max_args(inp, min=-5, max=5,
///                                                    num_bits=16)
/// print(out)
///
/// #  Output:
/// #  tf.Tensor([ 4.9999237 -5.0000763  3.0000763], shape=(3,), dtype=float32)
/// ```
///
/// Raises:
///   * InvalidArgumentError:
///     - If num_bits are outside of range [2, 16].
///     - If min >= max.
///   * ValueError: If `inputs` are of any other type than float32.
///
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The outputs tensor.
class FakeQuantWithMinMaxArgs {
 public:
  /// Optional attribute setters for FakeQuantWithMinMaxArgs
  struct Attrs {
    /// Defaults to -6
    TF_MUST_USE_RESULT Attrs Min(float x) {
      Attrs ret = *this;
      ret.min_ = x;
      return ret;
    }

    /// Defaults to 6
    TF_MUST_USE_RESULT Attrs Max(float x) {
      Attrs ret = *this;
      ret.max_ = x;
      return ret;
    }

    /// Defaults to 8
    TF_MUST_USE_RESULT Attrs NumBits(int64 x) {
      Attrs ret = *this;
      ret.num_bits_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs NarrowRange(bool x) {
      Attrs ret = *this;
      ret.narrow_range_ = x;
      return ret;
    }

    float min_ = -6.0f;
    float max_ = 6.0f;
    int64 num_bits_ = 8;
    bool narrow_range_ = false;
  };
  FakeQuantWithMinMaxArgs(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        inputs);
  FakeQuantWithMinMaxArgs(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        inputs, const FakeQuantWithMinMaxArgs::Attrs& attrs);
  operator ::tensorflow::Output() const { return outputs; }
  operator ::tensorflow::Input() const { return outputs; }
  ::tensorflow::Node* node() const { return outputs.node(); }

  static Attrs Min(float x) {
    return Attrs().Min(x);
  }
  static Attrs Max(float x) {
    return Attrs().Max(x);
  }
  static Attrs NumBits(int64 x) {
    return Attrs().NumBits(x);
  }
  static Attrs NarrowRange(bool x) {
    return Attrs().NarrowRange(x);
  }

  Operation operation;
  ::tensorflow::Output outputs;
};

/// Compute gradients for a FakeQuantWithMinMaxArgs operation.
///
/// Args:
/// * scope: A Scope object
/// * gradients: Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.
/// * inputs: Values passed as inputs to the FakeQuantWithMinMaxArgs operation.
///
/// Returns:
/// * `Output`: Backpropagated gradients below the FakeQuantWithMinMaxArgs operation:
/// `gradients * (inputs >= min && inputs <= max)`.
///
/// ```
/// import tensorflow as tf
///
/// # Define some sample data
/// gradients = tf.random.uniform((2, 3), minval=-5.0, maxval=5.0, dtype=tf.float32)
/// inputs = tf.random.uniform((2, 3), minval=-10.0, maxval=10.0, dtype=tf.float32)
///
/// # Define quantization parameters (adjust as needed)
/// min_val = -2.0
/// max_val = 8.0
/// num_bits = 4  # Number of bits for quantization
///
/// # Calculate gradients for fake quantization with specified parameters
/// output_gradients = tf.quantization.fake_quant_with_min_max_args_gradient(
///     gradients=gradients, inputs=inputs, min=min_val, max=max_val, num_bits=num_bits, narrow_range = False, name=None
/// )
///
/// # Print the original gradients and the gradients after the fake-quant operation
/// print("Original Gradients:")
/// print(gradients)
/// print("\nGradients after Fake-Quantization:")
/// print(output_gradients)
///
/// ```
/// #Original Gradients:
/// #tf.Tensor(
/// #[[ 1.242547    3.217492    3.568469  ]
/// #[-0.55371046  0.23130894  2.608243  ]], shape=(2, 3), dtype=float32)
///
/// #Gradients after Fake-Quantization:
/// #tf.Tensor(
/// #[[ 0.          3.217492    3.568469  ]
/// # [-0.55371046  0.23130894  2.608243  ]], shape=(2, 3), dtype=float32)
///
class FakeQuantWithMinMaxArgsGradient {
 public:
  /// Optional attribute setters for FakeQuantWithMinMaxArgsGradient
  struct Attrs {
    /// Defaults to -6
    TF_MUST_USE_RESULT Attrs Min(float x) {
      Attrs ret = *this;
      ret.min_ = x;
      return ret;
    }

    /// Defaults to 6
    TF_MUST_USE_RESULT Attrs Max(float x) {
      Attrs ret = *this;
      ret.max_ = x;
      return ret;
    }

    /// Defaults to 8
    TF_MUST_USE_RESULT Attrs NumBits(int64 x) {
      Attrs ret = *this;
      ret.num_bits_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs NarrowRange(bool x) {
      Attrs ret = *this;
      ret.narrow_range_ = x;
      return ret;
    }

    float min_ = -6.0f;
    float max_ = 6.0f;
    int64 num_bits_ = 8;
    bool narrow_range_ = false;
  };
  FakeQuantWithMinMaxArgsGradient(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input gradients,
                                ::tensorflow::Input inputs);
  FakeQuantWithMinMaxArgsGradient(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input gradients,
                                ::tensorflow::Input inputs, const
                                FakeQuantWithMinMaxArgsGradient::Attrs& attrs);
  operator ::tensorflow::Output() const { return backprops; }
  operator ::tensorflow::Input() const { return backprops; }
  ::tensorflow::Node* node() const { return backprops.node(); }

  static Attrs Min(float x) {
    return Attrs().Min(x);
  }
  static Attrs Max(float x) {
    return Attrs().Max(x);
  }
  static Attrs NumBits(int64 x) {
    return Attrs().NumBits(x);
  }
  static Attrs NarrowRange(bool x) {
    return Attrs().NarrowRange(x);
  }

  Operation operation;
  ::tensorflow::Output backprops;
};

/// Fake-quantize the 'inputs' tensor of type float via global float scalars
///
/// Fake-quantize the `inputs` tensor of type float via global float scalars
/// `min` and `max` to `outputs` tensor of same shape as `inputs`.
///
/// Attributes
///
/// *   `[min; max]` define the clamping range for the `inputs` data.
/// *   `inputs` values are quantized into the quantization range (
/// `[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
/// when it is true) and then de-quantized and output as floats in `[min; max]`
/// interval.
/// *   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
///
/// Before quantization, `min` and `max` values are adjusted with the following
/// logic.
/// It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
/// the behavior can be unexpected:
///
/// *   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
/// *   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
/// *   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
/// `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.
///
/// This operation has a gradient and thus allows for training `min` and `max`
/// values.
///
/// >>> constant_input = tf.constant([[1.2, -0.3, 0.7], [2.1, 0.5, -1.0]], dtype=tf.float32)
/// >>>
/// >>> min_val = -0.5
/// >>> max_val = 0.8
/// >>> num_bits = 8
/// >>> narrow_range = False #False:for the quantization range [0; 2^num_bits - 1]
/// >>>
/// >>> quantized_data = tf.quantization.fake_quant_with_min_max_vars(
/// ...   inputs=constant_input, min=min_val, max=max_val, num_bits=num_bits, narrow_range=narrow_range
/// ... )
/// >>>
/// >>> print("Input:\n", constant_input.numpy())
/// Input:
/// [[ 1.2 -0.3  0.7]
/// [ 2.1  0.5 -1. ]]
/// >>> print("Output:\n", quantized_data.numpy())
/// Output:
/// [[ 0.8003921 -0.3007843  0.6984313]
/// [ 0.8003921  0.4996078 -0.4996078]]
///
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The outputs tensor.
class FakeQuantWithMinMaxVars {
 public:
  /// Optional attribute setters for FakeQuantWithMinMaxVars
  struct Attrs {
    /// Defaults to 8
    TF_MUST_USE_RESULT Attrs NumBits(int64 x) {
      Attrs ret = *this;
      ret.num_bits_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs NarrowRange(bool x) {
      Attrs ret = *this;
      ret.narrow_range_ = x;
      return ret;
    }

    int64 num_bits_ = 8;
    bool narrow_range_ = false;
  };
  FakeQuantWithMinMaxVars(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        inputs, ::tensorflow::Input min, ::tensorflow::Input
                        max);
  FakeQuantWithMinMaxVars(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        inputs, ::tensorflow::Input min, ::tensorflow::Input
                        max, const FakeQuantWithMinMaxVars::Attrs& attrs);
  operator ::tensorflow::Output() const { return outputs; }
  operator ::tensorflow::Input() const { return outputs; }
  ::tensorflow::Node* node() const { return outputs.node(); }

  static Attrs NumBits(int64 x) {
    return Attrs().NumBits(x);
  }
  static Attrs NarrowRange(bool x) {
    return Attrs().NarrowRange(x);
  }

  Operation operation;
  ::tensorflow::Output outputs;
};

/// Compute gradients for a FakeQuantWithMinMaxVars operation.
///
/// Args:
/// * scope: A Scope object
/// * gradients: Backpropagated gradients above the FakeQuantWithMinMaxVars operation.
/// * inputs: Values passed as inputs to the FakeQuantWithMinMaxVars operation.
/// min, max: Quantization interval, scalar floats.
///
/// Optional attributes (see `Attrs`):
/// * num_bits: The bitwidth of the quantization; between 2 and 8, inclusive.
/// * narrow_range: Whether to quantize into 2^num_bits - 1 distinct values.
///
/// Returns:
/// * `Output` backprops_wrt_input: Backpropagated gradients w.r.t. inputs:
/// `gradients * (inputs >= min && inputs <= max)`.
/// * `Output` backprop_wrt_min: Backpropagated gradients w.r.t. min parameter:
/// `sum(gradients * (inputs < min))`.
/// * `Output` backprop_wrt_max: Backpropagated gradients w.r.t. max parameter:
/// `sum(gradients * (inputs > max))`.
class FakeQuantWithMinMaxVarsGradient {
 public:
  /// Optional attribute setters for FakeQuantWithMinMaxVarsGradient
  struct Attrs {
    /// The bitwidth of the quantization; between 2 and 8, inclusive.
    ///
    /// Defaults to 8
    TF_MUST_USE_RESULT Attrs NumBits(int64 x) {
      Attrs ret = *this;
      ret.num_bits_ = x;
      return ret;
    }

    /// Whether to quantize into 2^num_bits - 1 distinct values.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs NarrowRange(bool x) {
      Attrs ret = *this;
      ret.narrow_range_ = x;
      return ret;
    }

    int64 num_bits_ = 8;
    bool narrow_range_ = false;
  };
  FakeQuantWithMinMaxVarsGradient(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input gradients,
                                ::tensorflow::Input inputs, ::tensorflow::Input
                                min, ::tensorflow::Input max);
  FakeQuantWithMinMaxVarsGradient(const ::tensorflow::Scope& scope,
                                ::tensorflow::Input gradients,
                                ::tensorflow::Input inputs, ::tensorflow::Input
                                min, ::tensorflow::Input max, const
                                FakeQuantWithMinMaxVarsGradient::Attrs& attrs);

  static Attrs NumBits(int64 x) {
    return Attrs().NumBits(x);
  }
  static Attrs NarrowRange(bool x) {
    return Attrs().NarrowRange(x);
  }

  Operation operation;
  ::tensorflow::Output backprops_wrt_input;
  ::tensorflow::Output backprop_wrt_min;
  ::tensorflow::Output backprop_wrt_max;
};

/// Fake-quantize the 'inputs' tensor of type float via per-channel floats
///
/// Fake-quantize the `inputs` tensor of type float per-channel and one of the
/// shapes: `[d]`, `[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max`
/// of shape `[d]` to `outputs` tensor of same shape as `inputs`.
///
/// Attributes
///
/// *   `[min; max]` define the clamping range for the `inputs` data.
/// *   `inputs` values are quantized into the quantization range (
/// `[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
/// when it is true) and then de-quantized and output as floats in `[min; max]`
/// interval.
/// *   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.
///
/// Before quantization, `min` and `max` values are adjusted with the following
/// logic.
/// It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
/// the behavior can be unexpected:
///
/// *   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
/// *   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
/// *   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
/// `min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.
///
/// This operation has a gradient and thus allows for training `min` and `max`
/// values.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The outputs tensor.
class FakeQuantWithMinMaxVarsPerChannel {
 public:
  /// Optional attribute setters for FakeQuantWithMinMaxVarsPerChannel
  struct Attrs {
    /// Defaults to 8
    TF_MUST_USE_RESULT Attrs NumBits(int64 x) {
      Attrs ret = *this;
      ret.num_bits_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs NarrowRange(bool x) {
      Attrs ret = *this;
      ret.narrow_range_ = x;
      return ret;
    }

    int64 num_bits_ = 8;
    bool narrow_range_ = false;
  };
  FakeQuantWithMinMaxVarsPerChannel(const ::tensorflow::Scope& scope,
                                  ::tensorflow::Input inputs,
                                  ::tensorflow::Input min, ::tensorflow::Input
                                  max);
  FakeQuantWithMinMaxVarsPerChannel(const ::tensorflow::Scope& scope,
                                  ::tensorflow::Input inputs,
                                  ::tensorflow::Input min, ::tensorflow::Input
                                  max, const
                                  FakeQuantWithMinMaxVarsPerChannel::Attrs&
                                  attrs);
  operator ::tensorflow::Output() const { return outputs; }
  operator ::tensorflow::Input() const { return outputs; }
  ::tensorflow::Node* node() const { return outputs.node(); }

  static Attrs NumBits(int64 x) {
    return Attrs().NumBits(x);
  }
  static Attrs NarrowRange(bool x) {
    return Attrs().NarrowRange(x);
  }

  Operation operation;
  ::tensorflow::Output outputs;
};

/// Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.
///
/// Args:
/// * scope: A Scope object
/// * gradients: Backpropagated gradients above the FakeQuantWithMinMaxVars operation,
/// shape one of: `[d]`, `[b, d]`,  `[b, h, w, d]`.
/// * inputs: Values passed as inputs to the FakeQuantWithMinMaxVars operation, shape
///   same as `gradients`.
/// min, max: Quantization interval, floats of shape `[d]`.
///
/// Optional attributes (see `Attrs`):
/// * num_bits: The bitwidth of the quantization; between 2 and 16, inclusive.
/// * narrow_range: Whether to quantize into 2^num_bits - 1 distinct values.
///
/// Returns:
/// * `Output` backprops_wrt_input: Backpropagated gradients w.r.t. inputs, shape same as
/// `inputs`:
///   `gradients * (inputs >= min && inputs <= max)`.
/// * `Output` backprop_wrt_min: Backpropagated gradients w.r.t. min parameter, shape `[d]`:
/// `sum_per_d(gradients * (inputs < min))`.
/// * `Output` backprop_wrt_max: Backpropagated gradients w.r.t. max parameter, shape `[d]`:
/// `sum_per_d(gradients * (inputs > max))`.
class FakeQuantWithMinMaxVarsPerChannelGradient {
 public:
  /// Optional attribute setters for FakeQuantWithMinMaxVarsPerChannelGradient
  struct Attrs {
    /// The bitwidth of the quantization; between 2 and 16, inclusive.
    ///
    /// Defaults to 8
    TF_MUST_USE_RESULT Attrs NumBits(int64 x) {
      Attrs ret = *this;
      ret.num_bits_ = x;
      return ret;
    }

    /// Whether to quantize into 2^num_bits - 1 distinct values.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs NarrowRange(bool x) {
      Attrs ret = *this;
      ret.narrow_range_ = x;
      return ret;
    }

    int64 num_bits_ = 8;
    bool narrow_range_ = false;
  };
  FakeQuantWithMinMaxVarsPerChannelGradient(const ::tensorflow::Scope& scope,
                                          ::tensorflow::Input gradients,
                                          ::tensorflow::Input inputs,
                                          ::tensorflow::Input min,
                                          ::tensorflow::Input max);
  FakeQuantWithMinMaxVarsPerChannelGradient(const ::tensorflow::Scope& scope,
                                          ::tensorflow::Input gradients,
                                          ::tensorflow::Input inputs,
                                          ::tensorflow::Input min,
                                          ::tensorflow::Input max, const
                                          FakeQuantWithMinMaxVarsPerChannelGradient::Attrs&
                                          attrs);

  static Attrs NumBits(int64 x) {
    return Attrs().NumBits(x);
  }
  static Attrs NarrowRange(bool x) {
    return Attrs().NarrowRange(x);
  }

  Operation operation;
  ::tensorflow::Output backprops_wrt_input;
  ::tensorflow::Output backprop_wrt_min;
  ::tensorflow::Output backprop_wrt_max;
};

/// Creates a tensor filled with a scalar value.
///
/// This operation creates a tensor of shape `dims` and fills it with `value`.
///
/// For example:
///
/// ```
/// # Output tensor has shape [2, 3].
/// fill([2, 3], 9) ==> [[9, 9, 9]
///                      [9, 9, 9]]
/// ```
///
/// `tf.fill` differs from `tf.constant` in a few ways:
///
/// *   `tf.fill` only supports scalar contents, whereas `tf.constant` supports
///     Tensor values.
/// *   `tf.fill` creates an Op in the computation graph that constructs the actual
///     Tensor value at runtime. This is in contrast to `tf.constant` which embeds
///     the entire Tensor into the graph with a `Const` node.
/// *   Because `tf.fill` evaluates at graph runtime, it supports dynamic shapes
///     based on other runtime Tensors, unlike `tf.constant`.
///
/// Args:
/// * scope: A Scope object
/// * dims: 1-D. Represents the shape of the output tensor.
/// * value: 0-D (scalar). Value to fill the returned tensor.
///
/// @compatibility(numpy)
/// Equivalent to np.full
/// @end_compatibility
///
/// Returns:
/// * `Output`: The output tensor.
class Fill {
 public:
  Fill(const ::tensorflow::Scope& scope, ::tensorflow::Input dims,
     ::tensorflow::Input value);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Generates fingerprint values.
///
/// Generates fingerprint values of `data`.
///
/// Fingerprint op considers the first dimension of `data` as the batch dimension,
/// and `output[i]` contains the fingerprint value generated from contents in
/// `data[i, ...]` for all `i`.
///
/// Fingerprint op writes fingerprint values as byte arrays. For example, the
/// default method `farmhash64` generates a 64-bit fingerprint value at a time.
/// This 8-byte value is written out as an `uint8` array of size 8, in little-endian
/// order.
///
/// For example, suppose that `data` has data type `DT_INT32` and shape (2, 3, 4),
/// and that the fingerprint method is `farmhash64`. In this case, the output shape
/// is (2, 8), where 2 is the batch dimension size of `data`, and 8 is the size of
/// each fingerprint value in bytes. `output[0, :]` is generated from 12 integers in
/// `data[0, :, :]` and similarly `output[1, :]` is generated from other 12 integers
/// in `data[1, :, :]`.
///
/// Note that this op fingerprints the raw underlying buffer, and it does not
/// fingerprint Tensor's metadata such as data type and/or shape. For example, the
/// fingerprint values are invariant under reshapes and bitcasts as long as the
/// batch dimension remain the same:
///
/// ```
/// Fingerprint(data) == Fingerprint(Reshape(data, ...))
/// Fingerprint(data) == Fingerprint(Bitcast(data, ...))
/// ```
///
/// For string data, one should expect `Fingerprint(data) !=
/// Fingerprint(ReduceJoin(data))` in general.
///
/// Args:
/// * scope: A Scope object
/// * data: Must have rank 1 or higher.
/// * method: Fingerprint method used by this op. Currently available method is
/// `farmhash::fingerprint64`.
///
/// Returns:
/// * `Output`: A two-dimensional `Tensor` of type `tf.uint8`. The first dimension equals to
/// `data`'s first dimension, and the second dimension size depends on the
/// fingerprint algorithm.
class Fingerprint {
 public:
  Fingerprint(const ::tensorflow::Scope& scope, ::tensorflow::Input data,
            ::tensorflow::Input method);
  operator ::tensorflow::Output() const { return fingerprint; }
  operator ::tensorflow::Input() const { return fingerprint; }
  ::tensorflow::Node* node() const { return fingerprint.node(); }

  Operation operation;
  ::tensorflow::Output fingerprint;
};

/// Gather slices from `params` according to `indices`.
///
/// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
/// Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
///
/// ```python
///     # Scalar indices
///     output[:, ..., :] = params[indices, :, ... :]
///
///     # Vector indices
///     output[i, :, ..., :] = params[indices[i], :, ... :]
///
///     # Higher rank indices
///     output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
/// ```
///
/// If `indices` is a permutation and `len(indices) == params.shape[0]` then
/// this operation will permute `params` accordingly.
///
/// `validate_indices`: DEPRECATED. If this operation is assigned to CPU, values in
/// `indices` are always validated to be within range. If assigned to GPU,
/// out-of-bound indices result in safe but unspecified behavior, which may include
/// raising an error.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
/// </div>
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Gather {
 public:
  /// Optional attribute setters for Gather
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs ValidateIndices(bool x) {
      Attrs ret = *this;
      ret.validate_indices_ = x;
      return ret;
    }

    bool validate_indices_ = true;
  };
  Gather(const ::tensorflow::Scope& scope, ::tensorflow::Input params,
       ::tensorflow::Input indices);
  Gather(const ::tensorflow::Scope& scope, ::tensorflow::Input params,
       ::tensorflow::Input indices, const Gather::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs ValidateIndices(bool x) {
    return Attrs().ValidateIndices(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Gather slices from `params` into a Tensor with shape specified by `indices`.
///
/// `indices` is a K-dimensional integer tensor, best thought of as a
/// (K-1)-dimensional tensor of indices into `params`, where each element defines a
/// slice of `params`:
///
///     output[\\(i_0, ..., i_{K-2}\\)] = params[indices[\\(i_0, ..., i_{K-2}\\)]]
///
/// Whereas in `tf.gather` `indices` defines slices into the `axis`
/// dimension of `params`, in `tf.gather_nd`, `indices` defines slices into the
/// first `N` dimensions of `params`, where `N = indices.shape[-1]`.
///
/// The last dimension of `indices` can be at most the rank of
/// `params`:
///
///     indices.shape[-1] <= params.rank
///
/// The last dimension of `indices` corresponds to elements
/// (if `indices.shape[-1] == params.rank`) or slices
/// (if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
/// of `params`.  The output tensor has shape
///
///     indices.shape[:-1] + params.shape[indices.shape[-1]:]
///
/// If `indices` contains any out-of-bound indices, depending on
/// `bad_indices_policy`, the op will either return an error or ignore the
/// out-of-bound indices. `bad_indices_policy` can be one of the following values:
/// 1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
///    historically on CPU and GPU we handle errors in different ways, and for
///    backward compatibility we keep the default behavior.
/// 2. "ERROR": raises error; GPU does not support this value.
/// 3. "IGNORE": ignore error and set the corresponding output to 0;
///    supported on both CPU and GPU.
///
/// Some examples below.
///
/// Simple indexing into a matrix:
///
/// ```python
///     indices = [[0, 0], [1, 1]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = ['a', 'd']
/// ```
///
/// Slice indexing into a matrix:
///
/// ```python
///     indices = [[1], [0]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = [['c', 'd'], ['a', 'b']]
/// ```
///
/// Indexing into a 3-tensor:
///
/// ```python
///     indices = [[1]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [[['a1', 'b1'], ['c1', 'd1']]]
///
///
///     indices = [[0, 1], [1, 0]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [['c0', 'd0'], ['a1', 'b1']]
///
///
///     indices = [[0, 0, 1], [1, 0, 1]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = ['b0', 'b1']
/// ```
///
/// Batched indexing into a matrix:
///
/// ```python
///     indices = [[[0, 0]], [[0, 1]]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = [['a'], ['b']]
/// ```
///
/// Batched slice indexing into a matrix:
///
/// ```python
///     indices = [[[1]], [[0]]]
///     params = [['a', 'b'], ['c', 'd']]
///     output = [[['c', 'd']], [['a', 'b']]]
/// ```
///
/// Batched indexing into a 3-tensor:
///
/// ```python
///     indices = [[[1]], [[0]]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [[[['a1', 'b1'], ['c1', 'd1']]],
///               [[['a0', 'b0'], ['c0', 'd0']]]]
///
///     indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [[['c0', 'd0'], ['a1', 'b1']],
///               [['a0', 'b0'], ['c1', 'd1']]]
///
///
///     indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
///     params = [[['a0', 'b0'], ['c0', 'd0']],
///               [['a1', 'b1'], ['c1', 'd1']]]
///     output = [['b0', 'b1'], ['d0', 'c1']]
/// ```
///
/// See also `tf.gather` and `tf.batch_gather`.
///
/// Args:
/// * scope: A Scope object
/// * params: The tensor from which to gather values.
/// * indices: Index tensor.
///
/// Returns:
/// * `Output`: Values from `params` gathered from indices given by `indices`, with
/// shape `indices.shape[:-1] + params.shape[indices.shape[-1]:]`.
class GatherNd {
 public:
  /// Optional attribute setters for GatherNd
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    StringPiece bad_indices_policy_ = "";
  };
  GatherNd(const ::tensorflow::Scope& scope, ::tensorflow::Input params,
         ::tensorflow::Input indices);
  GatherNd(const ::tensorflow::Scope& scope, ::tensorflow::Input params,
         ::tensorflow::Input indices, const GatherNd::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Gather slices from `params` axis `axis` according to `indices`.
///
/// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
/// Produces an output tensor with shape `params.shape[:axis] +
/// indices.shape[batch_dims:] + params.shape[axis + 1:]` where:
///
/// ```python
///     # Scalar indices (output is rank(params) - 1).
///     output[a_0, ..., a_n, b_0, ..., b_n] =
///       params[a_0, ..., a_n, indices, b_0, ..., b_n]
///
///     # Vector indices (output is rank(params)).
///     output[a_0, ..., a_n, i, b_0, ..., b_n] =
///       params[a_0, ..., a_n, indices[i], b_0, ..., b_n]
///
///     # Higher rank indices (output is rank(params) + rank(indices) - 1).
///     output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
///       params[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
/// ```
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png" alt>
/// </div>
///
/// Note that on CPU, if an out of bound index is found, an error is returned.
/// On GPU, if an out of bound index is found, a 0 is stored in the
/// corresponding output value.
///
/// Note that on TPU, if any dimension of `params` is of size 0 then the output will
/// be the expected shape filled with zeros. On CPU and GPU an error will be
/// returned.
///
/// See also `tf.batch_gather` and `tf.gather_nd`.
///
/// Args:
/// * scope: A Scope object
/// * params: The tensor from which to gather values. Must be at least rank
/// `axis + 1`.
/// * indices: Index tensor. Must be in range `[0, params.shape[axis])`.
/// * axis: The axis in `params` to gather `indices` from. Defaults to the first
/// dimension. Supports negative indexes.
///
/// Returns:
/// * `Output`: Values from `params` gathered from indices given by `indices`, with
/// shape `params.shape[:axis] + indices.shape + params.shape[axis + 1:]`.
class GatherV2 {
 public:
  /// Optional attribute setters for GatherV2
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs BatchDims(int64 x) {
      Attrs ret = *this;
      ret.batch_dims_ = x;
      return ret;
    }

    int64 batch_dims_ = 0;
  };
  GatherV2(const ::tensorflow::Scope& scope, ::tensorflow::Input params,
         ::tensorflow::Input indices, ::tensorflow::Input axis);
  GatherV2(const ::tensorflow::Scope& scope, ::tensorflow::Input params,
         ::tensorflow::Input indices, ::tensorflow::Input axis, const
         GatherV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BatchDims(int64 x) {
    return Attrs().BatchDims(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Gives a guarantee to the TF runtime that the input tensor is a constant.
///
/// The runtime is then free to make optimizations based on this.
///
/// Only accepts value typed tensors as inputs and rejects resource variable handles
/// as input.
///
/// Returns the input tensor without modification.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class GuaranteeConst {
 public:
  GuaranteeConst(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Return a tensor with the same shape and contents as the input tensor or value.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Identity {
 public:
  Identity(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns a list of tensors with the same shapes and contents as the input
///
/// tensors.
///
/// This op can be used to override the gradient for complicated functions. For
/// example, suppose y = f(x) and we wish to apply a custom function g for backprop
/// such that dx = g(dy). In Python,
///
/// ```python
/// with tf.get_default_graph().gradient_override_map(
///     {'IdentityN': 'OverrideGradientWithG'}):
///   y, _ = identity_n([f(x), x])
///
/// @tf.RegisterGradient('OverrideGradientWithG')
/// def ApplyG(op, dy, _):
///   return [None, g(dy)]  # Do not backprop to f(x).
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class IdentityN {
 public:
  IdentityN(const ::tensorflow::Scope& scope, ::tensorflow::InputList input);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// Returns immutable tensor from memory region.
///
/// The current implementation memmaps the tensor from a file.
///
/// Args:
/// * scope: A Scope object
/// * dtype: Type of the returned tensor.
/// * shape: Shape of the returned tensor.
/// * memory_region_name: Name of readonly memory region used by the tensor, see
/// NewReadOnlyMemoryRegionFromFile in tensorflow::Env.
///
/// Returns:
/// * `Output`: The tensor tensor.
class ImmutableConst {
 public:
  ImmutableConst(const ::tensorflow::Scope& scope, DataType dtype,
               PartialTensorShape shape, StringPiece memory_region_name);
  operator ::tensorflow::Output() const { return tensor; }
  operator ::tensorflow::Input() const { return tensor; }
  ::tensorflow::Node* node() const { return tensor.node(); }

  Operation operation;
  ::tensorflow::Output tensor;
};

/// Adds v into specified rows of x.
///
///     Computes y = x; y[i, :] += v; return y.
///
/// Args:
/// * scope: A Scope object
/// * x: A `Tensor` of type T.
/// * i: A vector. Indices into the left-most dimension of `x`.
/// * v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
///
/// Returns:
/// * `Output`: A `Tensor` of type T. An alias of `x`. The content of `y` is undefined if there are duplicates in `i`.
class InplaceAdd {
 public:
  InplaceAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
           ::tensorflow::Input i, ::tensorflow::Input v);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

///     Subtracts `v` into specified rows of `x`.
///
///     Computes y = x; y[i, :] -= v; return y.
///
/// Args:
/// * scope: A Scope object
/// * x: A `Tensor` of type T.
/// * i: A vector. Indices into the left-most dimension of `x`.
/// * v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
///
/// Returns:
/// * `Output`: A `Tensor` of type T. An alias of `x`. The content of `y` is undefined if there are duplicates in `i`.
class InplaceSub {
 public:
  InplaceSub(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
           ::tensorflow::Input i, ::tensorflow::Input v);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Updates specified rows 'i' with values 'v'.
///
/// Computes `x[i, :] = v; return x`.
///
/// Originally this function is mutative however for compilation we make this
/// operation create / operate on a copy of `x`.
///
/// Args:
/// * scope: A Scope object
/// * x: A tensor of type `T`.
/// * i: A vector. Indices into the left-most dimension of `x`.
/// * v: A `Tensor` of type T. Same dimension sizes as x except the first dimension, which must be the same as i's size.
///
/// Returns:
/// * `Output`: A `Tensor` of type T. An alias of `x`. The content of `y` is undefined if there are duplicates in `i`.
class InplaceUpdate {
 public:
  InplaceUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
              ::tensorflow::Input i, ::tensorflow::Input v);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes the inverse permutation of a tensor.
///
/// This operation computes the inverse of an index permutation. It takes a 1-D
/// integer tensor `x`, which represents the indices of a zero-based array, and
/// swaps each value with its index position. In other words, for an output tensor
/// `y` and an input tensor `x`, this operation computes the following:
///
/// `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`
///
/// The values must include 0. There can be no duplicate values or negative values.
///
/// For example:
///
/// ```
/// # tensor `x` is [3, 4, 0, 2, 1]
/// invert_permutation(x) ==> [2, 4, 3, 0, 1]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * x: 1-D.
///
/// Returns:
/// * `Output`: 1-D.
class InvertPermutation {
 public:
  InvertPermutation(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Computes the difference between two lists of numbers or strings.
///
/// Given a list `x` and a list `y`, this operation returns a list `out` that
/// represents all values that are in `x` but not in `y`. The returned list `out`
/// is sorted in the same order that the numbers appear in `x` (duplicates are
/// preserved). This operation also returns a list `idx` that represents the
/// position of each `out` element in `x`. In other words:
///
/// `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`
///
/// For example, given this input:
///
/// ```
/// x = [1, 2, 3, 4, 5, 6]
/// y = [1, 3, 5]
/// ```
///
/// This operation would return:
///
/// ```
/// out ==> [2, 4, 6]
/// idx ==> [1, 3, 5]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * x: 1-D. Values to keep.
/// * y: 1-D. Values to remove.
///
/// Returns:
/// * `Output` out: 1-D. Values present in `x` but not in `y`.
/// * `Output` idx: 1-D. Positions of `x` values preserved in `out`.
class SetDiff1D {
 public:
  /// Optional attribute setters for SetDiff1D
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutIdx(DataType x) {
      Attrs ret = *this;
      ret.out_idx_ = x;
      return ret;
    }

    DataType out_idx_ = DT_INT32;
  };
  SetDiff1D(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
          ::tensorflow::Input y);
  SetDiff1D(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
          ::tensorflow::Input y, const SetDiff1D::Attrs& attrs);

  static Attrs OutIdx(DataType x) {
    return Attrs().OutIdx(x);
  }

  Operation operation;
  ::tensorflow::Output out;
  ::tensorflow::Output idx;
};

/// Copy a tensor setting everything outside a central band in each innermost matrix to zero.
///
/// The `band` part is computed as follows:
/// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
/// tensor with the same shape where
///
/// `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.
///
/// The indicator function
///
/// `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) &&
///                  (num_upper < 0 || (n-m) <= num_upper)`.
///
/// For example:
///
/// ```
/// # if 'input' is [[ 0,  1,  2, 3]
/// #                [-1,  0,  1, 2]
/// #                [-2, -1,  0, 1]
/// #                [-3, -2, -1, 0]],
///
/// tf.linalg.band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
///                                        [-1,  0,  1, 2]
///                                        [ 0, -1,  0, 1]
///                                        [ 0,  0, -1, 0]],
///
/// tf.linalg.band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
///                                       [-1,  0,  1, 0]
///                                       [-2, -1,  0, 1]
///                                       [ 0, -2, -1, 0]]
/// ```
///
/// Useful special cases:
///
/// ```
///  tf.linalg.band_part(input, 0, -1) ==> Upper triangular part.
///  tf.linalg.band_part(input, -1, 0) ==> Lower triangular part.
///  tf.linalg.band_part(input, 0, 0) ==> Diagonal.
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: Rank `k` tensor.
/// * num_lower: 0-D tensor. Number of subdiagonals to keep. If negative, keep entire
/// lower triangle.
/// * num_upper: 0-D tensor. Number of superdiagonals to keep. If negative, keep
/// entire upper triangle.
///
/// Returns:
/// * `Output`: Rank `k` tensor of the same shape as input. The extracted banded tensor.
class MatrixBandPart {
 public:
  MatrixBandPart(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
               ::tensorflow::Input num_lower, ::tensorflow::Input num_upper);
  operator ::tensorflow::Output() const { return band; }
  operator ::tensorflow::Input() const { return band; }
  ::tensorflow::Node* node() const { return band.node(); }

  Operation operation;
  ::tensorflow::Output band;
};

/// Returns a batched diagonal tensor with a given batched diagonal values.
///
/// Given a `diagonal`, this operation returns a tensor with the `diagonal` and
/// everything else padded with zeros. The diagonal is computed as follows:
///
/// Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
/// tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:
///
/// `output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.
///
/// For example:
///
/// ```
/// # 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]
///
/// and diagonal.shape = (2, 4)
///
/// tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
///                                      [0, 2, 0, 0]
///                                      [0, 0, 3, 0]
///                                      [0, 0, 0, 4]],
///                                     [[5, 0, 0, 0]
///                                      [0, 6, 0, 0]
///                                      [0, 0, 7, 0]
///                                      [0, 0, 0, 8]]]
///
/// which has shape (2, 4, 4)
/// ```
///
/// Args:
/// * scope: A Scope object
/// * diagonal: Rank `k`, where `k >= 1`.
///
/// Returns:
/// * `Output`: Rank `k+1`, with `output.shape = diagonal.shape + [diagonal.shape[-1]]`.
class MatrixDiag {
 public:
  MatrixDiag(const ::tensorflow::Scope& scope, ::tensorflow::Input diagonal);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns the batched diagonal part of a batched tensor.
///
/// This operation returns a tensor with the `diagonal` part
/// of the batched `input`. The `diagonal` part is computed as follows:
///
/// Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
/// tensor of rank `k - 1` with dimensions `[I, J, K, ..., min(M, N)]` where:
///
/// `diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.
///
/// The input must be at least a matrix.
///
/// For example:
///
/// ```
/// # 'input' is [[[1, 0, 0, 0]
///                [0, 2, 0, 0]
///                [0, 0, 3, 0]
///                [0, 0, 0, 4]],
///               [[5, 0, 0, 0]
///                [0, 6, 0, 0]
///                [0, 0, 7, 0]
///                [0, 0, 0, 8]]]
///
/// and input.shape = (2, 4, 4)
///
/// tf.matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]
///
/// which has shape (2, 4)
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: Rank `k` tensor where `k >= 2`.
///
/// Returns:
/// * `Output`: The extracted diagonal(s) having shape
/// `diagonal.shape = input.shape[:-2] + [min(input.shape[-2:])]`.
class MatrixDiagPart {
 public:
  MatrixDiagPart(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return diagonal; }
  operator ::tensorflow::Input() const { return diagonal; }
  ::tensorflow::Node* node() const { return diagonal.node(); }

  Operation operation;
  ::tensorflow::Output diagonal;
};

/// Returns the batched diagonal part of a batched tensor.
///
/// Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched
/// `input`.
///
/// Assume `input` has `r` dimensions `[I, J, ..., L, M, N]`.
/// Let `max_diag_len` be the maximum length among all diagonals to be extracted,
/// `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
/// Let `num_diags` be the number of diagonals to extract,
/// `num_diags = k[1] - k[0] + 1`.
///
/// If `num_diags == 1`, the output tensor is of rank `r - 1` with shape
/// `[I, J, ..., L, max_diag_len]` and values:
///
/// ```
/// diagonal[i, j, ..., l, n]
///   = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
///     padding_value                 ; otherwise.
/// ```
/// where `y = max(-k[1], 0)`, `x = max(k[1], 0)`.
///
/// Otherwise, the output tensor has rank `r` with dimensions
/// `[I, J, ..., L, num_diags, max_diag_len]` with values:
///
/// ```
/// diagonal[i, j, ..., l, m, n]
///   = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
///     padding_value                 ; otherwise.
/// ```
/// where `d = k[1] - m`, `y = max(-d, 0)`, and `x = max(d, 0)`.
///
/// The input must be at least a matrix.
///
/// For example:
///
/// ```
/// input = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
///                    [5, 6, 7, 8],
///                    [9, 8, 7, 6]],
///                   [[5, 4, 3, 2],
///                    [1, 2, 3, 4],
///                    [5, 6, 7, 8]]])
///
/// # A main diagonal from each batch.
/// tf.matrix_diag_part(input) ==> [[1, 6, 7],  # Output shape: (2, 3)
///                                 [5, 2, 7]]
///
/// # A superdiagonal from each batch.
/// tf.matrix_diag_part(input, k = 1)
///   ==> [[2, 7, 6],  # Output shape: (2, 3)
///        [4, 3, 8]]
///
/// # A tridiagonal band from each batch.
/// tf.matrix_diag_part(input, k = (-1, 1))
///   ==> [[[2, 7, 6],  # Output shape: (2, 3, 3)
///         [1, 6, 7],
///         [5, 8, 0]],
///        [[4, 3, 8],
///         [5, 2, 7],
///         [1, 6, 0]]]
///
/// # Padding value = 9
/// tf.matrix_diag_part(input, k = (1, 3), padding_value = 9)
///   ==> [[[4, 9, 9],  # Output shape: (2, 3, 3)
///         [3, 8, 9],
///         [2, 7, 6]],
///        [[2, 9, 9],
///         [3, 4, 9],
///         [4, 3, 8]]]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: Rank `r` tensor where `r >= 2`.
/// * k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
/// diagonal, and negative value means subdiagonals. `k` can be a single integer
/// (for a single diagonal) or a pair of integers specifying the low and high ends
/// of a matrix band. `k[0]` must not be larger than `k[1]`.
/// * padding_value: The value to fill the area outside the specified diagonal band with.
/// Default is 0.
///
/// Returns:
/// * `Output`: The extracted diagonal(s).
class MatrixDiagPartV2 {
 public:
  MatrixDiagPartV2(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                 ::tensorflow::Input k, ::tensorflow::Input padding_value);
  operator ::tensorflow::Output() const { return diagonal; }
  operator ::tensorflow::Input() const { return diagonal; }
  ::tensorflow::Node* node() const { return diagonal.node(); }

  Operation operation;
  ::tensorflow::Output diagonal;
};

/// Returns the batched diagonal part of a batched tensor.
///
/// Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched
/// `input`.
///
/// Assume `input` has `r` dimensions `[I, J, ..., L, M, N]`.
/// Let `max_diag_len` be the maximum length among all diagonals to be extracted,
/// `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
/// Let `num_diags` be the number of diagonals to extract,
/// `num_diags = k[1] - k[0] + 1`.
///
/// If `num_diags == 1`, the output tensor is of rank `r - 1` with shape
/// `[I, J, ..., L, max_diag_len]` and values:
///
/// ```
/// diagonal[i, j, ..., l, n]
///   = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
///     padding_value                 ; otherwise.
/// ```
/// where `y = max(-k[1], 0)`, `x = max(k[1], 0)`.
///
/// Otherwise, the output tensor has rank `r` with dimensions
/// `[I, J, ..., L, num_diags, max_diag_len]` with values:
///
/// ```
/// diagonal[i, j, ..., l, m, n]
///   = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
///     padding_value                 ; otherwise.
/// ```
/// where `d = k[1] - m`, `y = max(-d, 0) - offset`, and `x = max(d, 0) - offset`.
///
/// `offset` is zero except when the alignment of the diagonal is to the right.
/// ```
/// offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
///                                            and `d >= 0`) or
///                                          (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
///                                            and `d <= 0`)
///          0                          ; otherwise
/// ```
/// where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.
///
/// The input must be at least a matrix.
///
/// For example:
///
/// ```
/// input = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
///                    [5, 6, 7, 8],
///                    [9, 8, 7, 6]],
///                   [[5, 4, 3, 2],
///                    [1, 2, 3, 4],
///                    [5, 6, 7, 8]]])
///
/// # A main diagonal from each batch.
/// tf.matrix_diag_part(input) ==> [[1, 6, 7],  # Output shape: (2, 3)
///                                 [5, 2, 7]]
///
/// # A superdiagonal from each batch.
/// tf.matrix_diag_part(input, k = 1)
///   ==> [[2, 7, 6],  # Output shape: (2, 3)
///        [4, 3, 8]]
///
/// # A band from each batch.
/// tf.matrix_diag_part(input, k = (-1, 2))
///   ==> [[[0, 3, 8],  # Output shape: (2, 4, 3)
///         [2, 7, 6],
///         [1, 6, 7],
///         [5, 8, 0]],
///        [[0, 3, 4],
///         [4, 3, 8],
///         [5, 2, 7],
///         [1, 6, 0]]]
///
/// # LEFT_RIGHT alignment.
/// tf.matrix_diag_part(input, k = (-1, 2), align="LEFT_RIGHT")
///   ==> [[[3, 8, 0],  # Output shape: (2, 4, 3)
///         [2, 7, 6],
///         [1, 6, 7],
///         [0, 5, 8]],
///        [[3, 4, 0],
///         [4, 3, 8],
///         [5, 2, 7],
///         [0, 1, 6]]]
///
/// # max_diag_len can be shorter than the main diagonal.
/// tf.matrix_diag_part(input, k = (-2, -1))
///   ==> [[[5, 8],
///         [9, 0]],
///        [[1, 6],
///         [5, 0]]]
///
/// # padding_value = 9
/// tf.matrix_diag_part(input, k = (1, 3), padding_value = 9)
///   ==> [[[9, 9, 4],  # Output shape: (2, 3, 3)
///         [9, 3, 8],
///         [2, 7, 6]],
///        [[9, 9, 2],
///         [9, 3, 4],
///         [4, 3, 8]]]
///
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: Rank `r` tensor where `r >= 2`.
/// * k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
/// diagonal, and negative value means subdiagonals. `k` can be a single integer
/// (for a single diagonal) or a pair of integers specifying the low and high ends
/// of a matrix band. `k[0]` must not be larger than `k[1]`.
/// * padding_value: The value to fill the area outside the specified diagonal band with.
/// Default is 0.
///
/// Optional attributes (see `Attrs`):
/// * align: Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
/// a string specifying how superdiagonals and subdiagonals should be aligned,
/// respectively. There are four possible alignments: "RIGHT_LEFT" (default),
/// "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
/// to the right (left-pads the row) and subdiagonals to the left (right-pads the
/// row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
/// the opposite alignment.
///
/// Returns:
/// * `Output`: The extracted diagonal(s).
class MatrixDiagPartV3 {
 public:
  /// Optional attribute setters for MatrixDiagPartV3
  struct Attrs {
    /// Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
    /// a string specifying how superdiagonals and subdiagonals should be aligned,
    /// respectively. There are four possible alignments: "RIGHT_LEFT" (default),
    /// "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
    /// to the right (left-pads the row) and subdiagonals to the left (right-pads the
    /// row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
    /// the opposite alignment.
    ///
    /// Defaults to "RIGHT_LEFT"
    TF_MUST_USE_RESULT Attrs Align(StringPiece x) {
      Attrs ret = *this;
      ret.align_ = x;
      return ret;
    }

    StringPiece align_ = "RIGHT_LEFT";
  };
  MatrixDiagPartV3(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                 ::tensorflow::Input k, ::tensorflow::Input padding_value);
  MatrixDiagPartV3(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                 ::tensorflow::Input k, ::tensorflow::Input padding_value,
                 const MatrixDiagPartV3::Attrs& attrs);
  operator ::tensorflow::Output() const { return diagonal; }
  operator ::tensorflow::Input() const { return diagonal; }
  ::tensorflow::Node* node() const { return diagonal.node(); }

  static Attrs Align(StringPiece x) {
    return Attrs().Align(x);
  }

  Operation operation;
  ::tensorflow::Output diagonal;
};

/// Returns a batched diagonal tensor with given batched diagonal values.
///
/// Returns a tensor with the contents in `diagonal` as `k[0]`-th to `k[1]`-th
/// diagonals of a matrix, with everything else padded with `padding`. `num_rows`
/// and `num_cols` specify the dimension of the innermost matrix of the output. If
/// both are not specified, the op assumes the innermost matrix is square and infers
/// its size from `k` and the innermost dimension of `diagonal`. If only one of them
/// is specified, the op assumes the unspecified value is the smallest possible
/// based on other criteria.
///
/// Let `diagonal` have `r` dimensions `[I, J, ..., L, M, N]`. The output tensor has
/// rank `r+1` with shape `[I, J, ..., L, M, num_rows, num_cols]` when only one
/// diagonal is given (`k` is an integer or `k[0] == k[1]`). Otherwise, it has rank
/// `r` with shape `[I, J, ..., L, num_rows, num_cols]`.
///
/// The second innermost dimension of `diagonal` has double meaning.
/// When `k` is scalar or `k[0] == k[1]`, `M` is part of the batch size
/// [I, J, ..., M], and the output tensor is:
///
/// ```
/// output[i, j, ..., l, m, n]
///   = diagonal[i, j, ..., l, n-max(d_upper, 0)] ; if n - m == d_upper
///     padding_value                             ; otherwise
/// ```
///
/// Otherwise, `M` is treated as the number of diagonals for the matrix in the
/// same batch (`M = k[1]-k[0]+1`), and the output tensor is:
///
/// ```
/// output[i, j, ..., l, m, n]
///   = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
///     padding_value                                     ; otherwise
/// ```
/// where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0)`.
///
/// For example:
///
/// ```
/// # The main diagonal.
/// diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
///                      [5, 6, 7, 8]])
/// tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
///                                [0, 2, 0, 0],
///                                [0, 0, 3, 0],
///                                [0, 0, 0, 4]],
///                               [[5, 0, 0, 0],
///                                [0, 6, 0, 0],
///                                [0, 0, 7, 0],
///                                [0, 0, 0, 8]]]
///
/// # A superdiagonal (per batch).
/// diagonal = np.array([[1, 2, 3],  # Input shape: (2, 3)
///                      [4, 5, 6]])
/// tf.matrix_diag(diagonal, k = 1)
///   ==> [[[0, 1, 0, 0],  # Output shape: (2, 4, 4)
///         [0, 0, 2, 0],
///         [0, 0, 0, 3],
///         [0, 0, 0, 0]],
///        [[0, 4, 0, 0],
///         [0, 0, 5, 0],
///         [0, 0, 0, 6],
///         [0, 0, 0, 0]]]
///
/// # A band of diagonals.
/// diagonals = np.array([[[1, 2, 3],  # Input shape: (2, 2, 3)
///                        [4, 5, 0]],
///                       [[6, 7, 9],
///                        [9, 1, 0]]])
/// tf.matrix_diag(diagonals, k = (-1, 0))
///   ==> [[[1, 0, 0],  # Output shape: (2, 3, 3)
///         [4, 2, 0],
///         [0, 5, 3]],
///        [[6, 0, 0],
///         [9, 7, 0],
///         [0, 1, 9]]]
///
/// # Rectangular matrix.
/// diagonal = np.array([1, 2])  # Input shape: (2)
/// tf.matrix_diag(diagonal, k = -1, num_rows = 3, num_cols = 4)
///   ==> [[0, 0, 0, 0],  # Output shape: (3, 4)
///        [1, 0, 0, 0],
///        [0, 2, 0, 0]]
///
/// # Rectangular matrix with inferred num_cols and padding_value = 9.
/// tf.matrix_diag(diagonal, k = -1, num_rows = 3, padding_value = 9)
///   ==> [[9, 9],  # Output shape: (3, 2)
///        [1, 9],
///        [9, 2]]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * diagonal: Rank `r`, where `r >= 1`
/// * k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
/// diagonal, and negative value means subdiagonals. `k` can be a single integer
/// (for a single diagonal) or a pair of integers specifying the low and high ends
/// of a matrix band. `k[0]` must not be larger than `k[1]`.
/// * num_rows: The number of rows of the output matrix. If it is not provided, the op assumes
/// the output matrix is a square matrix and infers the matrix size from k and the
/// innermost dimension of `diagonal`.
/// * num_cols: The number of columns of the output matrix. If it is not provided, the op
/// assumes the output matrix is a square matrix and infers the matrix size from
/// k and the innermost dimension of `diagonal`.
/// * padding_value: The number to fill the area outside the specified diagonal band with.
/// Default is 0.
///
/// Returns:
/// * `Output`: Has rank `r+1` when `k` is an integer or `k[0] == k[1]`, rank `r` otherwise.
class MatrixDiagV2 {
 public:
  MatrixDiagV2(const ::tensorflow::Scope& scope, ::tensorflow::Input diagonal,
             ::tensorflow::Input k, ::tensorflow::Input num_rows,
             ::tensorflow::Input num_cols, ::tensorflow::Input padding_value);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns a batched diagonal tensor with given batched diagonal values.
///
/// Returns a tensor with the contents in `diagonal` as `k[0]`-th to `k[1]`-th
/// diagonals of a matrix, with everything else padded with `padding`. `num_rows`
/// and `num_cols` specify the dimension of the innermost matrix of the output. If
/// both are not specified, the op assumes the innermost matrix is square and infers
/// its size from `k` and the innermost dimension of `diagonal`. If only one of them
/// is specified, the op assumes the unspecified value is the smallest possible
/// based on other criteria.
///
/// Let `diagonal` have `r` dimensions `[I, J, ..., L, M, N]`. The output tensor has
/// rank `r+1` with shape `[I, J, ..., L, M, num_rows, num_cols]` when only one
/// diagonal is given (`k` is an integer or `k[0] == k[1]`). Otherwise, it has rank
/// `r` with shape `[I, J, ..., L, num_rows, num_cols]`.
///
/// The second innermost dimension of `diagonal` has double meaning.
/// When `k` is scalar or `k[0] == k[1]`, `M` is part of the batch size
/// [I, J, ..., M], and the output tensor is:
///
/// ```
/// output[i, j, ..., l, m, n]
///   = diagonal[i, j, ..., l, n-max(d_upper, 0)] ; if n - m == d_upper
///     padding_value                             ; otherwise
/// ```
///
/// Otherwise, `M` is treated as the number of diagonals for the matrix in the
/// same batch (`M = k[1]-k[0]+1`), and the output tensor is:
///
/// ```
/// output[i, j, ..., l, m, n]
///   = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
///     padding_value                                     ; otherwise
/// ```
/// where `d = n - m`, `diag_index = [k] - d`, and
/// `index_in_diag = n - max(d, 0) + offset`.
///
/// `offset` is zero except when the alignment of the diagonal is to the right.
/// ```
/// offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
///                                            and `d >= 0`) or
///                                          (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
///                                            and `d <= 0`)
///          0                          ; otherwise
/// ```
/// where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.
///
/// For example:
///
/// ```
/// # The main diagonal.
/// diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
///                      [5, 6, 7, 8]])
/// tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
///                                [0, 2, 0, 0],
///                                [0, 0, 3, 0],
///                                [0, 0, 0, 4]],
///                               [[5, 0, 0, 0],
///                                [0, 6, 0, 0],
///                                [0, 0, 7, 0],
///                                [0, 0, 0, 8]]]
///
/// # A superdiagonal (per batch).
/// diagonal = np.array([[1, 2, 3],  # Input shape: (2, 3)
///                      [4, 5, 6]])
/// tf.matrix_diag(diagonal, k = 1)
///   ==> [[[0, 1, 0, 0],  # Output shape: (2, 4, 4)
///         [0, 0, 2, 0],
///         [0, 0, 0, 3],
///         [0, 0, 0, 0]],
///        [[0, 4, 0, 0],
///         [0, 0, 5, 0],
///         [0, 0, 0, 6],
///         [0, 0, 0, 0]]]
///
/// # A tridiagonal band (per batch).
/// diagonals = np.array([[[0, 8, 9],  # Input shape: (2, 2, 3)
///                        [1, 2, 3],
///                        [4, 5, 0]],
///                       [[0, 2, 3],
///                        [6, 7, 9],
///                        [9, 1, 0]]])
/// tf.matrix_diag(diagonals, k = (-1, 1))
///   ==> [[[1, 8, 0],  # Output shape: (2, 3, 3)
///         [4, 2, 9],
///         [0, 5, 3]],
///        [[6, 2, 0],
///         [9, 7, 3],
///         [0, 1, 9]]]
///
/// # LEFT_RIGHT alignment.
/// diagonals = np.array([[[8, 9, 0],  # Input shape: (2, 2, 3)
///                        [1, 2, 3],
///                        [0, 4, 5]],
///                       [[2, 3, 0],
///                        [6, 7, 9],
///                        [0, 9, 1]]])
/// tf.matrix_diag(diagonals, k = (-1, 1), align="LEFT_RIGHT")
///   ==> [[[1, 8, 0],  # Output shape: (2, 3, 3)
///         [4, 2, 9],
///         [0, 5, 3]],
///        [[6, 2, 0],
///         [9, 7, 3],
///         [0, 1, 9]]]
///
/// # Rectangular matrix.
/// diagonal = np.array([1, 2])  # Input shape: (2)
/// tf.matrix_diag(diagonal, k = -1, num_rows = 3, num_cols = 4)
///   ==> [[0, 0, 0, 0],  # Output shape: (3, 4)
///        [1, 0, 0, 0],
///        [0, 2, 0, 0]]
///
/// # Rectangular matrix with inferred num_cols and padding_value = 9.
/// tf.matrix_diag(diagonal, k = -1, num_rows = 3, padding_value = 9)
///   ==> [[9, 9],  # Output shape: (3, 2)
///        [1, 9],
///        [9, 2]]
///
/// ```
///
/// Args:
/// * scope: A Scope object
/// * diagonal: Rank `r`, where `r >= 1`
/// * k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
/// diagonal, and negative value means subdiagonals. `k` can be a single integer
/// (for a single diagonal) or a pair of integers specifying the low and high ends
/// of a matrix band. `k[0]` must not be larger than `k[1]`.
/// * num_rows: The number of rows of the output matrix. If it is not provided, the op assumes
/// the output matrix is a square matrix and infers the matrix size from k and the
/// innermost dimension of `diagonal`.
/// * num_cols: The number of columns of the output matrix. If it is not provided, the op
/// assumes the output matrix is a square matrix and infers the matrix size from
/// k and the innermost dimension of `diagonal`.
/// * padding_value: The number to fill the area outside the specified diagonal band with.
/// Default is 0.
///
/// Optional attributes (see `Attrs`):
/// * align: Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
/// a string specifying how superdiagonals and subdiagonals should be aligned,
/// respectively. There are four possible alignments: "RIGHT_LEFT" (default),
/// "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
/// to the right (left-pads the row) and subdiagonals to the left (right-pads the
/// row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
/// the opposite alignment.
///
/// Returns:
/// * `Output`: Has rank `r+1` when `k` is an integer or `k[0] == k[1]`, rank `r` otherwise.
class MatrixDiagV3 {
 public:
  /// Optional attribute setters for MatrixDiagV3
  struct Attrs {
    /// Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
    /// a string specifying how superdiagonals and subdiagonals should be aligned,
    /// respectively. There are four possible alignments: "RIGHT_LEFT" (default),
    /// "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
    /// to the right (left-pads the row) and subdiagonals to the left (right-pads the
    /// row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
    /// the opposite alignment.
    ///
    /// Defaults to "RIGHT_LEFT"
    TF_MUST_USE_RESULT Attrs Align(StringPiece x) {
      Attrs ret = *this;
      ret.align_ = x;
      return ret;
    }

    StringPiece align_ = "RIGHT_LEFT";
  };
  MatrixDiagV3(const ::tensorflow::Scope& scope, ::tensorflow::Input diagonal,
             ::tensorflow::Input k, ::tensorflow::Input num_rows,
             ::tensorflow::Input num_cols, ::tensorflow::Input padding_value);
  MatrixDiagV3(const ::tensorflow::Scope& scope, ::tensorflow::Input diagonal,
             ::tensorflow::Input k, ::tensorflow::Input num_rows,
             ::tensorflow::Input num_cols, ::tensorflow::Input padding_value,
             const MatrixDiagV3::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Align(StringPiece x) {
    return Attrs().Align(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns a batched matrix tensor with new batched diagonal values.
///
/// Given `input` and `diagonal`, this operation returns a tensor with the
/// same shape and values as `input`, except for the main diagonal of the
/// innermost matrices.  These will be overwritten by the values in `diagonal`.
///
/// The output is computed as follows:
///
/// Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
/// `k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
/// tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:
///
///   * `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
///   * `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`.
///
/// Args:
/// * scope: A Scope object
/// * input: Rank `k+1`, where `k >= 1`.
/// * diagonal: Rank `k`, where `k >= 1`.
///
/// Returns:
/// * `Output`: Rank `k+1`, with `output.shape = input.shape`.
class MatrixSetDiag {
 public:
  MatrixSetDiag(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
              ::tensorflow::Input diagonal);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns a batched matrix tensor with new batched diagonal values.
///
/// Given `input` and `diagonal`, this operation returns a tensor with the
/// same shape and values as `input`, except for the specified diagonals of the
/// innermost matrices. These will be overwritten by the values in `diagonal`.
///
/// `input` has `r+1` dimensions `[I, J, ..., L, M, N]`. When `k` is scalar or
/// `k[0] == k[1]`, `diagonal` has `r` dimensions `[I, J, ..., L, max_diag_len]`.
/// Otherwise, it has `r+1` dimensions `[I, J, ..., L, num_diags, max_diag_len]`.
/// `num_diags` is the number of diagonals, `num_diags = k[1] - k[0] + 1`.
/// `max_diag_len` is the longest diagonal in the range `[k[0], k[1]]`,
/// `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
///
/// The output is a tensor of rank `k+1` with dimensions `[I, J, ..., L, M, N]`.
/// If `k` is scalar or `k[0] == k[1]`:
///
/// ```
/// output[i, j, ..., l, m, n]
///   = diagonal[i, j, ..., l, n-max(k[1], 0)] ; if n - m == k[1]
///     input[i, j, ..., l, m, n]              ; otherwise
/// ```
///
/// Otherwise,
///
/// ```
/// output[i, j, ..., l, m, n]
///   = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
///     input[i, j, ..., l, m, n]                         ; otherwise
/// ```
/// where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0)`.
///
/// For example:
///
/// ```
/// # The main diagonal.
/// input = np.array([[[7, 7, 7, 7],              # Input shape: (2, 3, 4)
///                    [7, 7, 7, 7],
///                    [7, 7, 7, 7]],
///                   [[7, 7, 7, 7],
///                    [7, 7, 7, 7],
///                    [7, 7, 7, 7]]])
/// diagonal = np.array([[1, 2, 3],               # Diagonal shape: (2, 3)
///                      [4, 5, 6]])
/// tf.matrix_set_diag(diagonal) ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
///                                    [7, 2, 7, 7],
///                                    [7, 7, 3, 7]],
///                                   [[4, 7, 7, 7],
///                                    [7, 5, 7, 7],
///                                    [7, 7, 6, 7]]]
///
/// # A superdiagonal (per batch).
/// tf.matrix_set_diag(diagonal, k = 1)
///   ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
///         [7, 7, 2, 7],
///         [7, 7, 7, 3]],
///        [[7, 4, 7, 7],
///         [7, 7, 5, 7],
///         [7, 7, 7, 6]]]
///
/// # A band of diagonals.
/// diagonals = np.array([[[1, 2, 3],  # Diagonal shape: (2, 2, 3)
///                        [4, 5, 0]],
///                       [[6, 1, 2],
///                        [3, 4, 0]]])
/// tf.matrix_set_diag(diagonals, k = (-1, 0))
///   ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
///         [4, 2, 7, 7],
///         [0, 5, 3, 7]],
///        [[6, 7, 7, 7],
///         [3, 1, 7, 7],
///         [7, 4, 2, 7]]]
///
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: Rank `r+1`, where `r >= 1`.
/// * diagonal: Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`.
/// `k >= 1`.
/// * k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
/// diagonal, and negative value means subdiagonals. `k` can be a single integer
/// (for a single diagonal) or a pair of integers specifying the low and high ends
/// of a matrix band. `k[0]` must not be larger than `k[1]`.
///
/// Returns:
/// * `Output`: Rank `r+1`, with `output.shape = input.shape`.
class MatrixSetDiagV2 {
 public:
  MatrixSetDiagV2(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                ::tensorflow::Input diagonal, ::tensorflow::Input k);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns a batched matrix tensor with new batched diagonal values.
///
/// Given `input` and `diagonal`, this operation returns a tensor with the
/// same shape and values as `input`, except for the specified diagonals of the
/// innermost matrices. These will be overwritten by the values in `diagonal`.
///
/// `input` has `r+1` dimensions `[I, J, ..., L, M, N]`. When `k` is scalar or
/// `k[0] == k[1]`, `diagonal` has `r` dimensions `[I, J, ..., L, max_diag_len]`.
/// Otherwise, it has `r+1` dimensions `[I, J, ..., L, num_diags, max_diag_len]`.
/// `num_diags` is the number of diagonals, `num_diags = k[1] - k[0] + 1`.
/// `max_diag_len` is the longest diagonal in the range `[k[0], k[1]]`,
/// `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
///
/// The output is a tensor of rank `k+1` with dimensions `[I, J, ..., L, M, N]`.
/// If `k` is scalar or `k[0] == k[1]`:
///
/// ```
/// output[i, j, ..., l, m, n]
///   = diagonal[i, j, ..., l, n-max(k[1], 0)] ; if n - m == k[1]
///     input[i, j, ..., l, m, n]              ; otherwise
/// ```
///
/// Otherwise,
///
/// ```
/// output[i, j, ..., l, m, n]
///   = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
///     input[i, j, ..., l, m, n]                         ; otherwise
/// ```
/// where `d = n - m`, `diag_index = k[1] - d`, and
/// `index_in_diag = n - max(d, 0) + offset`.
///
/// `offset` is zero except when the alignment of the diagonal is to the right.
/// ```
/// offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
///                                            and `d >= 0`) or
///                                          (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
///                                            and `d <= 0`)
///          0                          ; otherwise
/// ```
/// where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.
///
/// For example:
///
/// ```
/// # The main diagonal.
/// input = np.array([[[7, 7, 7, 7],              # Input shape: (2, 3, 4)
///                    [7, 7, 7, 7],
///                    [7, 7, 7, 7]],
///                   [[7, 7, 7, 7],
///                    [7, 7, 7, 7],
///                    [7, 7, 7, 7]]])
/// diagonal = np.array([[1, 2, 3],               # Diagonal shape: (2, 3)
///                      [4, 5, 6]])
/// tf.matrix_set_diag(input, diagonal)
///   ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
///         [7, 2, 7, 7],
///         [7, 7, 3, 7]],
///        [[4, 7, 7, 7],
///         [7, 5, 7, 7],
///         [7, 7, 6, 7]]]
///
/// # A superdiagonal (per batch).
/// tf.matrix_set_diag(input, diagonal, k = 1)
///   ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
///         [7, 7, 2, 7],
///         [7, 7, 7, 3]],
///        [[7, 4, 7, 7],
///         [7, 7, 5, 7],
///         [7, 7, 7, 6]]]
///
/// # A band of diagonals.
/// diagonals = np.array([[[0, 9, 1],  # Diagonal shape: (2, 4, 3)
///                        [6, 5, 8],
///                        [1, 2, 3],
///                        [4, 5, 0]],
///                       [[0, 1, 2],
///                        [5, 6, 4],
///                        [6, 1, 2],
///                        [3, 4, 0]]])
/// tf.matrix_set_diag(input, diagonals, k = (-1, 2))
///   ==> [[[1, 6, 9, 7],  # Output shape: (2, 3, 4)
///         [4, 2, 5, 1],
///         [7, 5, 3, 8]],
///        [[6, 5, 1, 7],
///         [3, 1, 6, 2],
///         [7, 4, 2, 4]]]
///
/// # LEFT_RIGHT alignment.
/// diagonals = np.array([[[9, 1, 0],  # Diagonal shape: (2, 4, 3)
///                        [6, 5, 8],
///                        [1, 2, 3],
///                        [0, 4, 5]],
///                       [[1, 2, 0],
///                        [5, 6, 4],
///                        [6, 1, 2],
///                        [0, 3, 4]]])
/// tf.matrix_set_diag(input, diagonals, k = (-1, 2), align="LEFT_RIGHT")
///   ==> [[[1, 6, 9, 7],  # Output shape: (2, 3, 4)
///         [4, 2, 5, 1],
///         [7, 5, 3, 8]],
///        [[6, 5, 1, 7],
///         [3, 1, 6, 2],
///         [7, 4, 2, 4]]]
///
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: Rank `r+1`, where `r >= 1`.
/// * diagonal: Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`.
/// `k >= 1`.
/// * k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
/// diagonal, and negative value means subdiagonals. `k` can be a single integer
/// (for a single diagonal) or a pair of integers specifying the low and high ends
/// of a matrix band. `k[0]` must not be larger than `k[1]`.
///
/// Optional attributes (see `Attrs`):
/// * align: Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
/// a string specifying how superdiagonals and subdiagonals should be aligned,
/// respectively. There are four possible alignments: "RIGHT_LEFT" (default),
/// "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
/// to the right (left-pads the row) and subdiagonals to the left (right-pads the
/// row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
/// the opposite alignment.
///
/// Returns:
/// * `Output`: Rank `r+1`, with `output.shape = input.shape`.
class MatrixSetDiagV3 {
 public:
  /// Optional attribute setters for MatrixSetDiagV3
  struct Attrs {
    /// Some diagonals are shorter than `max_diag_len` and need to be padded. `align` is
    /// a string specifying how superdiagonals and subdiagonals should be aligned,
    /// respectively. There are four possible alignments: "RIGHT_LEFT" (default),
    /// "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals
    /// to the right (left-pads the row) and subdiagonals to the left (right-pads the
    /// row). It is the packing format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is
    /// the opposite alignment.
    ///
    /// Defaults to "RIGHT_LEFT"
    TF_MUST_USE_RESULT Attrs Align(StringPiece x) {
      Attrs ret = *this;
      ret.align_ = x;
      return ret;
    }

    StringPiece align_ = "RIGHT_LEFT";
  };
  MatrixSetDiagV3(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                ::tensorflow::Input diagonal, ::tensorflow::Input k);
  MatrixSetDiagV3(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                ::tensorflow::Input diagonal, ::tensorflow::Input k, const
                MatrixSetDiagV3::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Align(StringPiece x) {
    return Attrs().Align(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Pads a tensor with mirrored values.
///
/// This operation pads a `input` with mirrored values according to the `paddings`
/// you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
/// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
/// how many values to add before the contents of `input` in that dimension, and
/// `paddings[D, 1]` indicates how many values to add after the contents of `input`
/// in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
/// than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
/// (if false, respectively).
///
/// The padded size of each dimension D of the output is:
///
/// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
///
/// For example:
///
/// ```
/// # 't' is [[1, 2, 3], [4, 5, 6]].
/// # 'paddings' is [[1, 1]], [2, 2]].
/// # 'mode' is SYMMETRIC.
/// # rank of 't' is 2.
/// pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
///                       [2, 1, 1, 2, 3, 3, 2]
///                       [5, 4, 4, 5, 6, 6, 5]
///                       [5, 4, 4, 5, 6, 6, 5]]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: The input tensor to be padded.
/// * paddings: A two-column matrix specifying the padding sizes. The number of
/// rows must be the same as the rank of `input`.
/// * mode: Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions
/// do not include the borders, while in symmetric mode the padded regions
/// do include the borders. For example, if `input` is `[1, 2, 3]` and `paddings`
/// is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in reflect mode, and
/// it is `[1, 2, 3, 3, 2]` in symmetric mode.
///
/// Returns:
/// * `Output`: The padded tensor.
class MirrorPad {
 public:
  MirrorPad(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
          ::tensorflow::Input paddings, StringPiece mode);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns a one-hot tensor.
///
/// The locations represented by indices in `indices` take value `on_value`,
/// while all other locations take value `off_value`.
///
/// If the input `indices` is rank `N`, the output will have rank `N+1`,
/// The new axis is created at dimension `axis` (default: the new axis is
/// appended at the end).
///
/// If `indices` is a scalar the output shape will be a vector of length `depth`.
///
/// If `indices` is a vector of length `features`, the output shape will be:
/// ```
///   features x depth if axis == -1
///   depth x features if axis == 0
/// ```
///
/// If `indices` is a matrix (batch) with shape `[batch, features]`,
/// the output shape will be:
/// ```
///   batch x features x depth if axis == -1
///   batch x depth x features if axis == 1
///   depth x batch x features if axis == 0
/// ```
///
///
/// Examples
/// =========
///
/// Suppose that
/// ```
///   indices = [0, 2, -1, 1]
///   depth = 3
///   on_value = 5.0
///   off_value = 0.0
///   axis = -1
/// ```
///
/// Then output is `[4 x 3]`:
/// ```
/// output =
///   [5.0 0.0 0.0]  // one_hot(0)
///   [0.0 0.0 5.0]  // one_hot(2)
///   [0.0 0.0 0.0]  // one_hot(-1)
///   [0.0 5.0 0.0]  // one_hot(1)
/// ```
///
/// Suppose that
/// ```
///   indices = [0, 2, -1, 1]
///   depth = 3
///   on_value = 0.0
///   off_value = 3.0
///   axis = 0
/// ```
///
/// Then output is `[3 x 4]`:
/// ```
/// output =
///   [0.0 3.0 3.0 3.0]
///   [3.0 3.0 3.0 0.0]
///   [3.0 3.0 3.0 3.0]
///   [3.0 0.0 3.0 3.0]
/// //  ^                one_hot(0)
/// //      ^            one_hot(2)
/// //          ^        one_hot(-1)
/// //              ^    one_hot(1)
/// ```
///
/// Suppose that
/// ```
///   indices = [[0, 2], [1, -1]]
///   depth = 3
///   on_value = 1.0
///   off_value = 0.0
///   axis = -1
/// ```
///
/// Then output is `[2 x 2 x 3]`:
/// ```
/// output =
///   [
///     [1.0, 0.0, 0.0]  // one_hot(0)
///     [0.0, 0.0, 1.0]  // one_hot(2)
///   ][
///     [0.0, 1.0, 0.0]  // one_hot(1)
///     [0.0, 0.0, 0.0]  // one_hot(-1)
///   ]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * indices: A tensor of indices.
/// * depth: A scalar defining the depth of the one hot dimension.
/// * on_value: A scalar defining the value to fill in output when `indices[j] = i`.
/// * off_value: A scalar defining the value to fill in output when `indices[j] != i`.
///
/// Optional attributes (see `Attrs`):
/// * axis: The axis to fill (default: -1, a new inner-most axis).
///
/// Returns:
/// * `Output`: The one-hot tensor.
class OneHot {
 public:
  /// Optional attribute setters for OneHot
  struct Attrs {
    /// The axis to fill (default: -1, a new inner-most axis).
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Axis(int64 x) {
      Attrs ret = *this;
      ret.axis_ = x;
      return ret;
    }

    int64 axis_ = -1;
  };
  OneHot(const ::tensorflow::Scope& scope, ::tensorflow::Input indices,
       ::tensorflow::Input depth, ::tensorflow::Input on_value,
       ::tensorflow::Input off_value);
  OneHot(const ::tensorflow::Scope& scope, ::tensorflow::Input indices,
       ::tensorflow::Input depth, ::tensorflow::Input on_value,
       ::tensorflow::Input off_value, const OneHot::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Axis(int64 x) {
    return Attrs().Axis(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns a tensor of ones with the same shape and type as x.
///
/// Args:
/// * scope: A Scope object
/// * x: a tensor of type T.
///
/// Returns:
/// * `Output`: a tensor of the same shape and type as x but filled with ones.
class OnesLike {
 public:
  OnesLike(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.
///
/// Packs the `N` tensors in `values` into a tensor with rank one higher than each
/// tensor in `values`, by packing them along the `axis` dimension.
/// Given a list of tensors of shape `(A, B, C)`;
///
/// if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
/// if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
/// Etc.
///
/// For example:
///
/// ```
/// # 'x' is [1, 4]
/// # 'y' is [2, 5]
/// # 'z' is [3, 6]
/// pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
/// pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
/// ```
///
/// This is the opposite of `unpack`.
///
/// Args:
/// * scope: A Scope object
/// * values: Must be of same shape and type.
///
/// Optional attributes (see `Attrs`):
/// * axis: Dimension along which to pack.  Negative values wrap around, so the
/// valid range is `[-(R+1), R+1)`.
///
/// Returns:
/// * `Output`: The packed tensor.
class Stack {
 public:
  /// Optional attribute setters for Stack
  struct Attrs {
    /// Dimension along which to pack.  Negative values wrap around, so the
    /// valid range is `[-(R+1), R+1)`.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Axis(int64 x) {
      Attrs ret = *this;
      ret.axis_ = x;
      return ret;
    }

    int64 axis_ = 0;
  };
  Stack(const ::tensorflow::Scope& scope, ::tensorflow::InputList values);
  Stack(const ::tensorflow::Scope& scope, ::tensorflow::InputList values, const
      Stack::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Axis(int64 x) {
    return Attrs().Axis(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Pads a tensor with zeros.
///
/// This operation pads a `input` with zeros according to the `paddings` you
/// specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
/// rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
/// how many zeros to add before the contents of `input` in that dimension, and
/// `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
/// in that dimension.
///
/// The padded size of each dimension D of the output is:
///
/// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
///
/// For example:
///
/// ```
/// # 't' is [[1, 1], [2, 2]]
/// # 'paddings' is [[1, 1], [2, 2]]
/// # rank of 't' is 2
/// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
///                       [0, 0, 1, 1, 0, 0]
///                       [0, 0, 2, 2, 0, 0]
///                       [0, 0, 0, 0, 0, 0]]
/// ```
///
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Pad {
 public:
  Pad(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
    ::tensorflow::Input paddings);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Pads a tensor.
///
/// This operation pads `input` according to the `paddings` and `constant_values`
/// you specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is
/// the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
/// how many padding values to add before the contents of `input` in that dimension,
/// and `paddings[D, 1]` indicates how many padding values to add after the contents
/// of `input` in that dimension. `constant_values` is a scalar tensor of the same
/// type as `input` that indicates the value to use for padding `input`.
///
/// The padded size of each dimension D of the output is:
///
/// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
///
/// For example:
///
/// ```
/// # 't' is [[1, 1], [2, 2]]
/// # 'paddings' is [[1, 1], [2, 2]]
/// # 'constant_values' is 0
/// # rank of 't' is 2
/// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
///                       [0, 0, 1, 1, 0, 0]
///                       [0, 0, 2, 2, 0, 0]
///                       [0, 0, 0, 0, 0, 0]]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class PadV2 {
 public:
  PadV2(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
      ::tensorflow::Input paddings, ::tensorflow::Input constant_values);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Concatenates a list of `N` tensors along the first dimension.
///
/// The input tensors are all required to have size 1 in the first dimension.
///
/// For example:
///
/// ```
/// # 'x' is [[1, 4]]
/// # 'y' is [[2, 5]]
/// # 'z' is [[3, 6]]
/// parallel_concat([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
/// ```
///
/// The difference between concat and parallel_concat is that concat requires all
/// of the inputs be computed before the operation will begin but doesn't require
/// that the input shapes be known during graph construction.  Parallel concat
/// will copy pieces of the input into the output as they become available, in
/// some situations this can provide a performance benefit.
///
/// Args:
/// * scope: A Scope object
/// * values: Tensors to be concatenated. All must have size 1 in the first dimension
/// and same shape.
/// * shape: the final shape of the result; should be equal to the shapes of any input
/// but with the number of input values in the first dimension.
///
/// Returns:
/// * `Output`: The concatenated tensor.
class ParallelConcat {
 public:
  ParallelConcat(const ::tensorflow::Scope& scope, ::tensorflow::InputList
               values, PartialTensorShape shape);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// A placeholder op for a value that will be fed into the computation.
///
/// N.B. This operation will fail with an error if it is executed. It is
/// intended as a way to represent a value that will always be fed, and to
/// provide attrs that enable the fed value to be checked at runtime.
///
/// Args:
/// * scope: A Scope object
/// * dtype: The type of elements in the tensor.
///
/// Optional attributes (see `Attrs`):
/// * shape: (Optional) The shape of the tensor. If the shape has 0 dimensions, the
/// shape is unconstrained.
///
/// Returns:
/// * `Output`: A placeholder tensor that must be replaced using the feed mechanism.
class Placeholder {
 public:
  /// Optional attribute setters for Placeholder
  struct Attrs {
    /// (Optional) The shape of the tensor. If the shape has 0 dimensions, the
    /// shape is unconstrained.
    ///
    /// Defaults to <unknown>
    TF_MUST_USE_RESULT Attrs Shape(PartialTensorShape x) {
      Attrs ret = *this;
      ret.shape_ = x;
      return ret;
    }

    PartialTensorShape shape_ = ::tensorflow::PartialTensorShape() /* unknown */;
  };
  Placeholder(const ::tensorflow::Scope& scope, DataType dtype);
  Placeholder(const ::tensorflow::Scope& scope, DataType dtype, const
            Placeholder::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Shape(PartialTensorShape x) {
    return Attrs().Shape(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// A placeholder op that passes through `input` when its output is not fed.
///
/// Args:
/// * scope: A Scope object
/// * input: The default value to produce when `output` is not fed.
/// * shape: The (possibly partial) shape of the tensor.
///
/// Returns:
/// * `Output`: A placeholder tensor that defaults to `input` if it is not fed.
class PlaceholderWithDefault {
 public:
  PlaceholderWithDefault(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       input, PartialTensorShape shape);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// An identity op that triggers an error if a gradient is requested.
///
/// When executed in a graph, this op outputs its input tensor as-is.
///
/// When building ops to compute gradients, the TensorFlow gradient system
/// will return an error when trying to lookup the gradient of this op,
/// because no gradient must ever be registered for this function.  This
/// op exists to prevent subtle bugs from silently returning unimplemented
/// gradients in some corner cases.
///
/// Args:
/// * scope: A Scope object
/// * input: any tensor.
///
/// Optional attributes (see `Attrs`):
/// * message: Will be printed in the error when anyone tries to differentiate
/// this operation.
///
/// Returns:
/// * `Output`: the same input tensor.
class PreventGradient {
 public:
  /// Optional attribute setters for PreventGradient
  struct Attrs {
    /// Will be printed in the error when anyone tries to differentiate
    /// this operation.
    ///
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs Message(StringPiece x) {
      Attrs ret = *this;
      ret.message_ = x;
      return ret;
    }

    StringPiece message_ = "";
  };
  PreventGradient(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  PreventGradient(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                const PreventGradient::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Message(StringPiece x) {
    return Attrs().Message(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Quantizes then dequantizes a tensor.
///
/// This op simulates the precision loss from the quantized forward pass by:
///
/// 1. Quantizing the tensor to fixed point numbers, which should match the target
///    quantization method when it is used in inference.
/// 2. Dequantizing it back to floating point numbers for the following ops, most
///    likely matmul.
///
/// There are different ways to quantize. This version uses only scaling, so 0.0
/// maps to 0.
///
/// From the specified 'num_bits' in the quantized output type, it determines
/// minimum and maximum representable quantized values.
///
/// e.g.
///
/// *   [-128, 127] for signed, num_bits = 8, or
/// *   [0, 255] for unsigned, num_bits = 8.
///
/// If range_given == False, the initial input_min, input_max will be determined
/// automatically as the minimum and maximum values in the input tensor, otherwise
/// the specified values of input_min, input_max are used.
///
/// Note: If the input_min, input_max are specified, they do not need to equal the
/// actual minimum and maximum values in the tensor. e.g. in some cases it may be
/// beneficial to specify these values such that the low probability extremes of the
/// input distribution are clipped.
///
/// This op determines the maximum scale_factor that would map the initial
/// [input_min, input_max] range to a range that lies within the representable
/// quantized range.
///
/// It determines the scale from one of input_min and input_max, then updates the
/// other one to maximize the representable range.
///
/// e.g.
///
/// *   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
///     5.0]: it would use a scale_factor of -128 / -10.0 = 12.8 In this case, it
///     would update input_max to be 127 / 12.8 = 9.921875
/// *   if the output is signed, num_bits = 8, [input_min, input_max] = [-10.0,
///     10.0]: it would use a scale_factor of 127 / 10.0 = 12.7 In this case, it
///     would update input_min to be 128.0 / 12.7 = -10.07874
/// *   if the output is unsigned, input_min is forced to be 0, and only the
///     specified input_max is used.
///
/// After determining the scale_factor and updating the input range, it applies the
/// following to each value in the 'input' tensor.
///
/// output = round(clamp(value, input_min, input_max) * scale_factor) / scale_factor.
///
/// The above round function rounds the value based on the given round_mode.
///
///
/// Args:
/// * scope: A Scope object
/// * input: Tensor to quantize and then dequantize.
/// * input_min: If `range_given == True`, this specifies the minimum input value that needs to
/// be represented, otherwise it is determined from the min value of the `input`
/// tensor.
/// * input_max: If `range_given == True`, this specifies the maximum input value that needs to
/// be represented, otherwise it is determined from the max value of the `input`
/// tensor.
///
/// Optional attributes (see `Attrs`):
/// * signed_input: Whether the quantization is signed or unsigned. (actually this parameter should
/// have been called <b>`signed_output`</b>)
/// * num_bits: The bitwidth of the quantization.
/// * range_given: Whether the range is given or should be determined from the `input` tensor.
/// * round_mode: The 'round_mode' attribute controls which rounding tie-breaking algorithm is
/// used when rounding float values to their quantized equivalents. The following
/// rounding modes are currently supported:
///
/// *   HALF_TO_EVEN: this is the default round_mode.
/// *   HALF_UP: round towards positive. In this mode 7.5 rounds up to 8 and -7.5
///     rounds up to -7.
///
/// * narrow_range: If True, then the absolute value of the quantized minimum value is the same as
/// the quantized maximum value, instead of 1 greater.
/// i.e. for 8 bit quantization, the minimum value is -127 instead of -128.
/// * axis: If specified, this axis is treated as a channel or slice axis, and a separate
/// quantization range is used for each channel or slice along this axis.
///
/// Returns:
/// * `Output`: The output tensor.
class QuantizeAndDequantizeV2 {
 public:
  /// Optional attribute setters for QuantizeAndDequantizeV2
  struct Attrs {
    /// Whether the quantization is signed or unsigned. (actually this parameter should
    /// have been called <b>`signed_output`</b>)
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs SignedInput(bool x) {
      Attrs ret = *this;
      ret.signed_input_ = x;
      return ret;
    }

    /// The bitwidth of the quantization.
    ///
    /// Defaults to 8
    TF_MUST_USE_RESULT Attrs NumBits(int64 x) {
      Attrs ret = *this;
      ret.num_bits_ = x;
      return ret;
    }

    /// Whether the range is given or should be determined from the `input` tensor.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs RangeGiven(bool x) {
      Attrs ret = *this;
      ret.range_given_ = x;
      return ret;
    }

    /// The 'round_mode' attribute controls which rounding tie-breaking algorithm is
    /// used when rounding float values to their quantized equivalents. The following
    /// rounding modes are currently supported:
    ///
    /// *   HALF_TO_EVEN: this is the default round_mode.
    /// *   HALF_UP: round towards positive. In this mode 7.5 rounds up to 8 and -7.5
    ///     rounds up to -7.
    ///
    ///
    /// Defaults to "HALF_TO_EVEN"
    TF_MUST_USE_RESULT Attrs RoundMode(StringPiece x) {
      Attrs ret = *this;
      ret.round_mode_ = x;
      return ret;
    }

    /// If True, then the absolute value of the quantized minimum value is the same as
    /// the quantized maximum value, instead of 1 greater.
    /// i.e. for 8 bit quantization, the minimum value is -127 instead of -128.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs NarrowRange(bool x) {
      Attrs ret = *this;
      ret.narrow_range_ = x;
      return ret;
    }

    /// If specified, this axis is treated as a channel or slice axis, and a separate
    /// quantization range is used for each channel or slice along this axis.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Axis(int64 x) {
      Attrs ret = *this;
      ret.axis_ = x;
      return ret;
    }

    bool signed_input_ = true;
    int64 num_bits_ = 8;
    bool range_given_ = false;
    StringPiece round_mode_ = "HALF_TO_EVEN";
    bool narrow_range_ = false;
    int64 axis_ = -1;
  };
  QuantizeAndDequantizeV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, ::tensorflow::Input input_min,
                        ::tensorflow::Input input_max);
  QuantizeAndDequantizeV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, ::tensorflow::Input input_min,
                        ::tensorflow::Input input_max, const
                        QuantizeAndDequantizeV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs SignedInput(bool x) {
    return Attrs().SignedInput(x);
  }
  static Attrs NumBits(int64 x) {
    return Attrs().NumBits(x);
  }
  static Attrs RangeGiven(bool x) {
    return Attrs().RangeGiven(x);
  }
  static Attrs RoundMode(StringPiece x) {
    return Attrs().RoundMode(x);
  }
  static Attrs NarrowRange(bool x) {
    return Attrs().NarrowRange(x);
  }
  static Attrs Axis(int64 x) {
    return Attrs().Axis(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Quantizes then dequantizes a tensor.
///
/// This is almost identical to QuantizeAndDequantizeV2, except that num_bits is a
/// tensor, so its value can change during training.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class QuantizeAndDequantizeV3 {
 public:
  /// Optional attribute setters for QuantizeAndDequantizeV3
  struct Attrs {
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs SignedInput(bool x) {
      Attrs ret = *this;
      ret.signed_input_ = x;
      return ret;
    }

    /// Defaults to true
    TF_MUST_USE_RESULT Attrs RangeGiven(bool x) {
      Attrs ret = *this;
      ret.range_given_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs NarrowRange(bool x) {
      Attrs ret = *this;
      ret.narrow_range_ = x;
      return ret;
    }

    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Axis(int64 x) {
      Attrs ret = *this;
      ret.axis_ = x;
      return ret;
    }

    bool signed_input_ = true;
    bool range_given_ = true;
    bool narrow_range_ = false;
    int64 axis_ = -1;
  };
  QuantizeAndDequantizeV3(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, ::tensorflow::Input input_min,
                        ::tensorflow::Input input_max, ::tensorflow::Input
                        num_bits);
  QuantizeAndDequantizeV3(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, ::tensorflow::Input input_min,
                        ::tensorflow::Input input_max, ::tensorflow::Input
                        num_bits, const QuantizeAndDequantizeV3::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs SignedInput(bool x) {
    return Attrs().SignedInput(x);
  }
  static Attrs RangeGiven(bool x) {
    return Attrs().RangeGiven(x);
  }
  static Attrs NarrowRange(bool x) {
    return Attrs().NarrowRange(x);
  }
  static Attrs Axis(int64 x) {
    return Attrs().Axis(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Quantizes then dequantizes a tensor.
///
/// This is almost identical to QuantizeAndDequantizeV2, except that it returns a
/// gradient of 1 for inputs that are within the quantization range, or 0 otherwise.
///
/// Args:
/// * scope: A Scope object
/// * input: Tensor to quantize and then dequantize.
/// * input_min: If `range_given == True`, this specifies the minimum input value that needs to
/// be represented, otherwise it is determined from the min value of the `input`
/// tensor.
/// * input_max: If `range_given == True`, this specifies the maximum input value that needs to
/// be represented, otherwise it is determined from the max value of the `input`
/// tensor.
///
/// Optional attributes (see `Attrs`):
/// * signed_input: Whether the quantization is signed or unsigned. (actually this parameter should
/// have been called <b>`signed_output`</b>)
/// * num_bits: The bitwidth of the quantization.
/// * range_given: Whether the range is given or should be determined from the `input` tensor.
/// * round_mode: The 'round_mode' attribute controls which rounding tie-breaking algorithm is
/// used when rounding float values to their quantized equivalents. The following
/// rounding modes are currently supported:
///
/// *   HALF_TO_EVEN: this is the default round_mode.
/// *   HALF_UP: round towards positive. In this mode 7.5 rounds up to 8 and -7.5
///     rounds up to -7.
///
/// * narrow_range: If True, then the absolute value of the quantized minimum value is the same as
/// the quantized maximum value, instead of 1 greater.
/// i.e. for 8 bit quantization, the minimum value is -127 instead of -128.
/// * axis: If specified, this axis is treated as a channel or slice axis, and a separate
/// quantization range is used for each channel or slice along this axis.
///
/// Returns:
/// * `Output`: The output tensor.
class QuantizeAndDequantizeV4 {
 public:
  /// Optional attribute setters for QuantizeAndDequantizeV4
  struct Attrs {
    /// Whether the quantization is signed or unsigned. (actually this parameter should
    /// have been called <b>`signed_output`</b>)
    ///
    /// Defaults to true
    TF_MUST_USE_RESULT Attrs SignedInput(bool x) {
      Attrs ret = *this;
      ret.signed_input_ = x;
      return ret;
    }

    /// The bitwidth of the quantization.
    ///
    /// Defaults to 8
    TF_MUST_USE_RESULT Attrs NumBits(int64 x) {
      Attrs ret = *this;
      ret.num_bits_ = x;
      return ret;
    }

    /// Whether the range is given or should be determined from the `input` tensor.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs RangeGiven(bool x) {
      Attrs ret = *this;
      ret.range_given_ = x;
      return ret;
    }

    /// The 'round_mode' attribute controls which rounding tie-breaking algorithm is
    /// used when rounding float values to their quantized equivalents. The following
    /// rounding modes are currently supported:
    ///
    /// *   HALF_TO_EVEN: this is the default round_mode.
    /// *   HALF_UP: round towards positive. In this mode 7.5 rounds up to 8 and -7.5
    ///     rounds up to -7.
    ///
    ///
    /// Defaults to "HALF_TO_EVEN"
    TF_MUST_USE_RESULT Attrs RoundMode(StringPiece x) {
      Attrs ret = *this;
      ret.round_mode_ = x;
      return ret;
    }

    /// If True, then the absolute value of the quantized minimum value is the same as
    /// the quantized maximum value, instead of 1 greater.
    /// i.e. for 8 bit quantization, the minimum value is -127 instead of -128.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs NarrowRange(bool x) {
      Attrs ret = *this;
      ret.narrow_range_ = x;
      return ret;
    }

    /// If specified, this axis is treated as a channel or slice axis, and a separate
    /// quantization range is used for each channel or slice along this axis.
    ///
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Axis(int64 x) {
      Attrs ret = *this;
      ret.axis_ = x;
      return ret;
    }

    bool signed_input_ = true;
    int64 num_bits_ = 8;
    bool range_given_ = false;
    StringPiece round_mode_ = "HALF_TO_EVEN";
    bool narrow_range_ = false;
    int64 axis_ = -1;
  };
  QuantizeAndDequantizeV4(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, ::tensorflow::Input input_min,
                        ::tensorflow::Input input_max);
  QuantizeAndDequantizeV4(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, ::tensorflow::Input input_min,
                        ::tensorflow::Input input_max, const
                        QuantizeAndDequantizeV4::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs SignedInput(bool x) {
    return Attrs().SignedInput(x);
  }
  static Attrs NumBits(int64 x) {
    return Attrs().NumBits(x);
  }
  static Attrs RangeGiven(bool x) {
    return Attrs().RangeGiven(x);
  }
  static Attrs RoundMode(StringPiece x) {
    return Attrs().RoundMode(x);
  }
  static Attrs NarrowRange(bool x) {
    return Attrs().NarrowRange(x);
  }
  static Attrs Axis(int64 x) {
    return Attrs().Axis(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns the gradient of `QuantizeAndDequantizeV4`.
///
/// Returns a gradient of 1 for inputs that are within the quantization range,
/// or 0 otherwise.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output` input_backprop
/// * `Output` input_min_backprop
/// * `Output` input_max_backprop
class QuantizeAndDequantizeV4Grad {
 public:
  /// Optional attribute setters for QuantizeAndDequantizeV4Grad
  struct Attrs {
    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Axis(int64 x) {
      Attrs ret = *this;
      ret.axis_ = x;
      return ret;
    }

    int64 axis_ = -1;
  };
  QuantizeAndDequantizeV4Grad(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input gradients, ::tensorflow::Input
                            input, ::tensorflow::Input input_min,
                            ::tensorflow::Input input_max);
  QuantizeAndDequantizeV4Grad(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input gradients, ::tensorflow::Input
                            input, ::tensorflow::Input input_min,
                            ::tensorflow::Input input_max, const
                            QuantizeAndDequantizeV4Grad::Attrs& attrs);

  static Attrs Axis(int64 x) {
    return Attrs().Axis(x);
  }

  Operation operation;
  ::tensorflow::Output input_backprop;
  ::tensorflow::Output input_min_backprop;
  ::tensorflow::Output input_max_backprop;
};

/// Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.
///
/// [min_range, max_range] are scalar floats that specify the range for
/// the 'input' data. The 'mode' attribute controls exactly which calculations are
/// used to convert the float values to their quantized equivalents.  The
/// 'round_mode' attribute controls which rounding tie-breaking algorithm is used
/// when rounding float values to their quantized equivalents.
///
/// In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:
///
/// ```
/// out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
/// if T == qint8: out[i] -= (range(T) + 1) / 2.0
/// ```
///
/// here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`
///
/// *MIN_COMBINED Mode Example*
///
/// Assume the input is type float and has a possible range of [0.0, 6.0] and the
/// output type is quint8 ([0, 255]). The min_range and max_range values should be
/// specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
/// value of the input by 255/6 and cast to quint8.
///
/// If the output type was qint8 ([-128, 127]), the operation will additionally
/// subtract each value by 128 prior to casting, so that the range of values aligns
/// with the range of qint8.
///
/// If the mode is 'MIN_FIRST', then this approach is used:
///
/// ```
/// num_discrete_values = 1 << (# of bits in T)
/// range_adjust = num_discrete_values / (num_discrete_values - 1)
/// range = (range_max - range_min) * range_adjust
/// range_scale = num_discrete_values / range
/// quantized = round(input * range_scale) - round(range_min * range_scale) +
///   numeric_limits<T>::min()
/// quantized = max(quantized, numeric_limits<T>::min())
/// quantized = min(quantized, numeric_limits<T>::max())
/// ```
///
/// The biggest difference between this and MIN_COMBINED is that the minimum range
/// is rounded first, before it's subtracted from the rounded value. With
/// MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
/// and dequantizing will introduce a larger and larger error.
///
/// *SCALED mode Example*
///
/// `SCALED` mode matches the quantization approach used in
/// `QuantizeAndDequantize{V2|V3}`.
///
/// If the mode is `SCALED`, the quantization is performed by multiplying each
/// input value by a scaling_factor.
/// The scaling_factor is determined from `min_range` and `max_range` to be as large
/// as possible such that the range from `min_range` to `max_range` is representable
/// within values of type T.
///
/// ```c++
///
///   const int min_T = std::numeric_limits<T>::min();
///   const int max_T = std::numeric_limits<T>::max();
///   const float max_float = std::numeric_limits<float>::max();
///
///   const float scale_factor_from_min_side =
///       (min_T * min_range > 0) ? min_T / min_range : max_float;
///   const float scale_factor_from_max_side =
///       (max_T * max_range > 0) ? max_T / max_range : max_float;
///
///   const float scale_factor = std::min(scale_factor_from_min_side,
///                                       scale_factor_from_max_side);
/// ```
///
/// We next use the scale_factor to adjust min_range and max_range as follows:
///
/// ```c++
///       min_range = min_T / scale_factor;
///       max_range = max_T / scale_factor;
/// ```
///
///
/// e.g. if T = qint8, and initially min_range = -10, and max_range = 9, we would
/// compare -128/-10.0 = 12.8 to 127/9.0 = 14.11, and set scaling_factor = 12.8
/// In this case, min_range would remain -10, but max_range would be adjusted to
/// 127 / 12.8 = 9.921875
///
/// So we will quantize input values in the range (-10, 9.921875) to (-128, 127).
///
/// The input tensor can now be quantized by clipping values to the range
/// `min_range` to `max_range`, then multiplying by scale_factor as follows:
///
/// ```c++
/// result = round(min(max_range, max(min_range, input)) * scale_factor)
/// ```
///
/// The adjusted `min_range` and `max_range` are returned as outputs 2 and 3 of
/// this operation. These outputs should be used as the range for any further
/// calculations.
///
///
/// *narrow_range (bool) attribute*
///
/// If true, we do not use the minimum quantized value.
/// i.e. for int8 the quantized output, it would be restricted to the range
/// -127..127 instead of the full -128..127 range.
/// This is provided for compatibility with certain inference backends.
/// (Only applies to SCALED mode)
///
///
/// *axis (int) attribute*
///
/// An optional `axis` attribute can specify a dimension index of the input tensor,
/// such that quantization ranges will be calculated and applied separately for each
/// slice of the tensor along that dimension. This is useful for per-channel
/// quantization.
///
/// If axis is specified, min_range and max_range
///
/// if `axis`=None, per-tensor quantization is performed as normal.
///
///
/// *ensure_minimum_range (float) attribute*
///
/// Ensures the minimum quantization range is at least this value.
/// The legacy default value for this is 0.01, but it is strongly suggested to
/// set it to 0 for new uses.
///
///
/// Args:
/// * scope: A Scope object
/// * min_range: The minimum value of the quantization range. This value may be adjusted by the
/// op depending on other parameters. The adjusted value is written to `output_min`.
/// If the `axis` attribute is specified, this must be a 1-D tensor whose size
/// matches the `axis` dimension of the input and output tensors.
/// * max_range: The maximum value of the quantization range. This value may be adjusted by the
/// op depending on other parameters. The adjusted value is written to `output_max`.
/// If the `axis` attribute is specified, this must be a 1-D tensor whose size
/// matches the `axis` dimension of the input and output tensors.
///
/// Returns:
/// * `Output` output: The quantized data produced from the float input.
/// * `Output` output_min: The final quantization range minimum, used to clip input values before scaling
/// and rounding them to quantized values.
/// If the `axis` attribute is specified, this will be a 1-D tensor whose size
/// matches the `axis` dimension of the input and output tensors.
/// * `Output` output_max: The final quantization range maximum, used to clip input values before scaling
/// and rounding them to quantized values.
/// If the `axis` attribute is specified, this will be a 1-D tensor whose size
/// matches the `axis` dimension of the input and output tensors.
class QuantizeV2 {
 public:
  /// Optional attribute setters for QuantizeV2
  struct Attrs {
    /// Defaults to "MIN_COMBINED"
    TF_MUST_USE_RESULT Attrs Mode(StringPiece x) {
      Attrs ret = *this;
      ret.mode_ = x;
      return ret;
    }

    /// Defaults to "HALF_AWAY_FROM_ZERO"
    TF_MUST_USE_RESULT Attrs RoundMode(StringPiece x) {
      Attrs ret = *this;
      ret.round_mode_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs NarrowRange(bool x) {
      Attrs ret = *this;
      ret.narrow_range_ = x;
      return ret;
    }

    /// Defaults to -1
    TF_MUST_USE_RESULT Attrs Axis(int64 x) {
      Attrs ret = *this;
      ret.axis_ = x;
      return ret;
    }

    /// Defaults to 0.01
    TF_MUST_USE_RESULT Attrs EnsureMinimumRange(float x) {
      Attrs ret = *this;
      ret.ensure_minimum_range_ = x;
      return ret;
    }

    StringPiece mode_ = "MIN_COMBINED";
    StringPiece round_mode_ = "HALF_AWAY_FROM_ZERO";
    bool narrow_range_ = false;
    int64 axis_ = -1;
    float ensure_minimum_range_ = 0.01f;
  };
  QuantizeV2(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
           ::tensorflow::Input min_range, ::tensorflow::Input max_range,
           DataType T);
  QuantizeV2(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
           ::tensorflow::Input min_range, ::tensorflow::Input max_range,
           DataType T, const QuantizeV2::Attrs& attrs);

  static Attrs Mode(StringPiece x) {
    return Attrs().Mode(x);
  }
  static Attrs RoundMode(StringPiece x) {
    return Attrs().RoundMode(x);
  }
  static Attrs NarrowRange(bool x) {
    return Attrs().NarrowRange(x);
  }
  static Attrs Axis(int64 x) {
    return Attrs().Axis(x);
  }
  static Attrs EnsureMinimumRange(float x) {
    return Attrs().EnsureMinimumRange(x);
  }

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output output_min;
  ::tensorflow::Output output_max;
};

/// Concatenates quantized tensors along one dimension.
///
/// Args:
/// * scope: A Scope object
/// * concat_dim: 0-D.  The dimension along which to concatenate.  Must be in the
/// range [0, rank(values)).
/// * values: The `N` Tensors to concatenate. Their ranks and types must match,
/// and their sizes must match in all dimensions except `concat_dim`.
/// * input_mins: The minimum scalar values for each of the input tensors.
/// * input_maxes: The maximum scalar values for each of the input tensors.
///
/// Returns:
/// * `Output` output: A `Tensor` with the concatenation of values stacked along the
/// `concat_dim` dimension.  This tensor's shape matches that of `values` except
/// in `concat_dim` where it has the sum of the sizes.
/// * `Output` output_min: The float value that the minimum quantized output value represents.
/// * `Output` output_max: The float value that the maximum quantized output value represents.
class QuantizedConcat {
 public:
  QuantizedConcat(const ::tensorflow::Scope& scope, ::tensorflow::Input
                concat_dim, ::tensorflow::InputList values,
                ::tensorflow::InputList input_mins, ::tensorflow::InputList
                input_maxes);

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output output_min;
  ::tensorflow::Output output_max;
};

/// Quantized Instance normalization.
///
/// Args:
/// * scope: A Scope object
/// * x: A 4D input Tensor.
/// * x_min: The value represented by the lowest quantized input.
/// * x_max: The value represented by the highest quantized input.
///
/// Optional attributes (see `Attrs`):
/// * output_range_given: If True, `given_y_min` and `given_y_min`
/// and `given_y_max` are used as the output range. Otherwise,
/// the implementation computes the output range.
/// * given_y_min: Output in `y_min` if `output_range_given` is True.
/// * given_y_max: Output in `y_max` if `output_range_given` is True.
/// * variance_epsilon: A small float number to avoid dividing by 0.
/// * min_separation: Minimum value of `y_max - y_min`
///
/// Returns:
/// * `Output` y: A 4D Tensor.
/// * `Output` y_min: The value represented by the lowest quantized output.
/// * `Output` y_max: The value represented by the highest quantized output.
class QuantizedInstanceNorm {
 public:
  /// Optional attribute setters for QuantizedInstanceNorm
  struct Attrs {
    /// If True, `given_y_min` and `given_y_min`
    /// and `given_y_max` are used as the output range. Otherwise,
    /// the implementation computes the output range.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs OutputRangeGiven(bool x) {
      Attrs ret = *this;
      ret.output_range_given_ = x;
      return ret;
    }

    /// Output in `y_min` if `output_range_given` is True.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs GivenYMin(float x) {
      Attrs ret = *this;
      ret.given_y_min_ = x;
      return ret;
    }

    /// Output in `y_max` if `output_range_given` is True.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs GivenYMax(float x) {
      Attrs ret = *this;
      ret.given_y_max_ = x;
      return ret;
    }

    /// A small float number to avoid dividing by 0.
    ///
    /// Defaults to 1e-05
    TF_MUST_USE_RESULT Attrs VarianceEpsilon(float x) {
      Attrs ret = *this;
      ret.variance_epsilon_ = x;
      return ret;
    }

    /// Minimum value of `y_max - y_min`
    ///
    /// Defaults to 0.001
    TF_MUST_USE_RESULT Attrs MinSeparation(float x) {
      Attrs ret = *this;
      ret.min_separation_ = x;
      return ret;
    }

    bool output_range_given_ = false;
    float given_y_min_ = 0.0f;
    float given_y_max_ = 0.0f;
    float variance_epsilon_ = 1e-05f;
    float min_separation_ = 0.001f;
  };
  QuantizedInstanceNorm(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
                      ::tensorflow::Input x_min, ::tensorflow::Input x_max);
  QuantizedInstanceNorm(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
                      ::tensorflow::Input x_min, ::tensorflow::Input x_max,
                      const QuantizedInstanceNorm::Attrs& attrs);

  static Attrs OutputRangeGiven(bool x) {
    return Attrs().OutputRangeGiven(x);
  }
  static Attrs GivenYMin(float x) {
    return Attrs().GivenYMin(x);
  }
  static Attrs GivenYMax(float x) {
    return Attrs().GivenYMax(x);
  }
  static Attrs VarianceEpsilon(float x) {
    return Attrs().VarianceEpsilon(x);
  }
  static Attrs MinSeparation(float x) {
    return Attrs().MinSeparation(x);
  }

  Operation operation;
  ::tensorflow::Output y;
  ::tensorflow::Output y_min;
  ::tensorflow::Output y_max;
};

/// Reshapes a quantized tensor as per the Reshape op.
///
/// ```
///
/// Args:
/// * scope: A Scope object
/// * shape: Defines the shape of the output tensor.
/// * input_min: The minimum value of the input.
/// * input_max: The maximum value of the input.
///
/// Returns:
/// * `Output` output
/// * `Output` output_min: This value is copied from input_min.
/// * `Output` output_max: This value is copied from input_max.
class QuantizedReshape {
 public:
  QuantizedReshape(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
                 ::tensorflow::Input shape, ::tensorflow::Input input_min,
                 ::tensorflow::Input input_max);

  Operation operation;
  ::tensorflow::Output output;
  ::tensorflow::Output output_min;
  ::tensorflow::Output output_max;
};

/// Returns the rank of a tensor.
///
/// This operation returns an integer representing the rank of `input`.
///
/// For example:
///
/// ```
/// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
/// # shape of tensor 't' is [2, 2, 3]
/// rank(t) ==> 3
/// ```
///
/// **Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
/// of a tensor is the number of indices required to uniquely select each element
/// of the tensor. Rank is also known as "order", "degree", or "ndims."
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Rank {
 public:
  Rank(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Reshapes a tensor.
///
/// Given `tensor`, this operation returns a tensor that has the same values
/// as `tensor` with shape `shape`.
///
/// If one component of 1-D tensor `shape` is the special value -1, the size of that
/// dimension is computed so that the total size remains constant.  In particular, a
/// `shape` of `[-1]` flattens into 1-D.  At most one component of `shape` may be
/// unknown.
///
/// The `shape` must be 1-D and the operation returns a tensor with shape
/// `shape` filled with the values of `tensor`. In this case, the number of elements
/// implied by `shape` must be the same as the number of elements in `tensor`.
///
/// It is an error if `shape` is not 1-D.
///
/// For example:
///
/// ```
/// # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
/// # tensor 't' has shape [9]
/// reshape(t, [3, 3]) ==> [[1, 2, 3],
///                         [4, 5, 6],
///                         [7, 8, 9]]
///
/// # tensor 't' is [[[1, 1], [2, 2]],
/// #                [[3, 3], [4, 4]]]
/// # tensor 't' has shape [2, 2, 2]
/// reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
///                         [3, 3, 4, 4]]
///
/// # tensor 't' is [[[1, 1, 1],
/// #                 [2, 2, 2]],
/// #                [[3, 3, 3],
/// #                 [4, 4, 4]],
/// #                [[5, 5, 5],
/// #                 [6, 6, 6]]]
/// # tensor 't' has shape [3, 2, 3]
/// # pass '[-1]' to flatten 't'
/// reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
///
/// # -1 can also be used to infer the shape
///
/// # -1 is inferred to be 9:
/// reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
///                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
/// # -1 is inferred to be 2:
/// reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
///                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
/// # -1 is inferred to be 3:
/// reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
///                               [2, 2, 2],
///                               [3, 3, 3]],
///                              [[4, 4, 4],
///                               [5, 5, 5],
///                               [6, 6, 6]]]
///
/// # tensor 't' is [7]
/// # shape `[]` reshapes to a scalar
/// reshape(t, []) ==> 7
/// ```
///
/// Args:
/// * scope: A Scope object
/// * shape: Defines the shape of the output tensor.
///
/// Returns:
/// * `Output`: The output tensor.
class Reshape {
 public:
  Reshape(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
        ::tensorflow::Input shape);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Assign `value` to the sliced l-value reference of `ref`.
///
/// The values of `value` are assigned to the positions in the variable
/// `ref` that are selected by the slice parameters. The slice parameters
/// `begin, `end`, `strides`, etc. work exactly as in `StridedSlice`.
///
/// NOTE this op currently does not support broadcasting and so `value`'s
/// shape must be exactly the shape produced by the slice of `ref`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * the created `Operation`
class ResourceStridedSliceAssign {
 public:
  /// Optional attribute setters for ResourceStridedSliceAssign
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs BeginMask(int64 x) {
      Attrs ret = *this;
      ret.begin_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs EndMask(int64 x) {
      Attrs ret = *this;
      ret.end_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs EllipsisMask(int64 x) {
      Attrs ret = *this;
      ret.ellipsis_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NewAxisMask(int64 x) {
      Attrs ret = *this;
      ret.new_axis_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs ShrinkAxisMask(int64 x) {
      Attrs ret = *this;
      ret.shrink_axis_mask_ = x;
      return ret;
    }

    int64 begin_mask_ = 0;
    int64 end_mask_ = 0;
    int64 ellipsis_mask_ = 0;
    int64 new_axis_mask_ = 0;
    int64 shrink_axis_mask_ = 0;
  };
  ResourceStridedSliceAssign(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input ref, ::tensorflow::Input begin,
                           ::tensorflow::Input end, ::tensorflow::Input
                           strides, ::tensorflow::Input value);
  ResourceStridedSliceAssign(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input ref, ::tensorflow::Input begin,
                           ::tensorflow::Input end, ::tensorflow::Input
                           strides, ::tensorflow::Input value, const
                           ResourceStridedSliceAssign::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs BeginMask(int64 x) {
    return Attrs().BeginMask(x);
  }
  static Attrs EndMask(int64 x) {
    return Attrs().EndMask(x);
  }
  static Attrs EllipsisMask(int64 x) {
    return Attrs().EllipsisMask(x);
  }
  static Attrs NewAxisMask(int64 x) {
    return Attrs().NewAxisMask(x);
  }
  static Attrs ShrinkAxisMask(int64 x) {
    return Attrs().ShrinkAxisMask(x);
  }

  Operation operation;
};

/// Reverses variable length slices.
///
/// This op first slices `input` along the dimension `batch_dim`, and for each
/// slice `i`, reverses the first `seq_lengths[i]` elements along
/// the dimension `seq_dim`.
///
/// The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
/// and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.
///
/// The output slice `i` along dimension `batch_dim` is then given by input
/// slice `i`, with the first `seq_lengths[i]` slices along dimension
/// `seq_dim` reversed.
///
/// For example:
///
/// ```
/// # Given this:
/// batch_dim = 0
/// seq_dim = 1
/// input.dims = (4, 8, ...)
/// seq_lengths = [7, 2, 3, 5]
///
/// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
/// output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
/// output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
/// output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
/// output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]
///
/// # while entries past seq_lens are copied through:
/// output[0, 7:, :, ...] = input[0, 7:, :, ...]
/// output[1, 2:, :, ...] = input[1, 2:, :, ...]
/// output[2, 3:, :, ...] = input[2, 3:, :, ...]
/// output[3, 2:, :, ...] = input[3, 2:, :, ...]
/// ```
///
/// In contrast, if:
///
/// ```
/// # Given this:
/// batch_dim = 2
/// seq_dim = 0
/// input.dims = (8, ?, 4, ...)
/// seq_lengths = [7, 2, 3, 5]
///
/// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
/// output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
/// output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
/// output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
/// output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]
///
/// # while entries past seq_lens are copied through:
/// output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
/// output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
/// output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
/// output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: The input to reverse.
/// * seq_lengths: 1-D with length `input.dims(batch_dim)` and
/// `max(seq_lengths) <= input.dims(seq_dim)`
/// * seq_dim: The dimension which is partially reversed.
///
/// Optional attributes (see `Attrs`):
/// * batch_dim: The dimension along which reversal is performed.
///
/// Returns:
/// * `Output`: The partially reversed input. It has the same shape as `input`.
class ReverseSequence {
 public:
  /// Optional attribute setters for ReverseSequence
  struct Attrs {
    /// The dimension along which reversal is performed.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs BatchDim(int64 x) {
      Attrs ret = *this;
      ret.batch_dim_ = x;
      return ret;
    }

    int64 batch_dim_ = 0;
  };
  ReverseSequence(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                ::tensorflow::Input seq_lengths, int64 seq_dim);
  ReverseSequence(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
                ::tensorflow::Input seq_lengths, int64 seq_dim, const
                ReverseSequence::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BatchDim(int64 x) {
    return Attrs().BatchDim(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Reverses specific dimensions of a tensor.
///
/// Given a `tensor`, and a `int32` tensor `axis` representing the set of
/// dimensions of `tensor` to reverse. This operation reverses each dimension
/// `i` for which there exists `j` s.t. `axis[j] == i`.
///
/// `tensor` can have up to 8 dimensions. The number of dimensions specified
/// in `axis` may be 0 or more entries. If an index is specified more than
/// once, a InvalidArgument error is raised.
///
/// For example:
///
/// ```
/// # tensor 't' is [[[[ 0,  1,  2,  3],
/// #                  [ 4,  5,  6,  7],
/// #                  [ 8,  9, 10, 11]],
/// #                 [[12, 13, 14, 15],
/// #                  [16, 17, 18, 19],
/// #                  [20, 21, 22, 23]]]]
/// # tensor 't' shape is [1, 2, 3, 4]
///
/// # 'dims' is [3] or 'dims' is [-1]
/// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
///                         [ 7,  6,  5,  4],
///                         [ 11, 10, 9, 8]],
///                        [[15, 14, 13, 12],
///                         [19, 18, 17, 16],
///                         [23, 22, 21, 20]]]]
///
/// # 'dims' is '[1]' (or 'dims' is '[-3]')
/// reverse(t, dims) ==> [[[[12, 13, 14, 15],
///                         [16, 17, 18, 19],
///                         [20, 21, 22, 23]
///                        [[ 0,  1,  2,  3],
///                         [ 4,  5,  6,  7],
///                         [ 8,  9, 10, 11]]]]
///
/// # 'dims' is '[2]' (or 'dims' is '[-2]')
/// reverse(t, dims) ==> [[[[8, 9, 10, 11],
///                         [4, 5, 6, 7],
///                         [0, 1, 2, 3]]
///                        [[20, 21, 22, 23],
///                         [16, 17, 18, 19],
///                         [12, 13, 14, 15]]]]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * tensor: Up to 8-D.
/// * axis: 1-D. The indices of the dimensions to reverse. Must be in the range
/// `[-rank(tensor), rank(tensor))`.
///
/// Returns:
/// * `Output`: The same shape as `tensor`.
class Reverse {
 public:
  Reverse(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
        ::tensorflow::Input axis);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Scatters `updates` into a tensor of shape `shape` according to `indices`.
///
/// Scatter sparse `updates` according to individual values at the specified
/// `indices`. This op returns an output tensor with the `shape` you specify. This
/// op is the inverse of the `tf.gather_nd` operator which extracts values or slices
/// from a given tensor.
///
/// This operation is similar to `tf.tensor_scatter_nd_add`, except that the tensor
/// is zero-initialized. Calling `tf.scatter_nd(indices, updates, shape)`
/// is identical to calling
/// `tf.tensor_scatter_nd_add(tf.zeros(shape, updates.dtype), indices, updates)`
///
/// If `indices` contains duplicates, the associated `updates` are accumulated
/// (summed) into the output tensor.
///
/// **WARNING**: For floating-point data types, the output may be nondeterministic.
/// This is because the order in which the updates are applied is nondeterministic
/// and when floating-point numbers are added in different orders the resulting
/// numerical approximation error can be slightly different. However, the output
/// will be deterministic if op determinism is enabled via
/// `tf.config.experimental.enable_op_determinism`.
///
/// `indices` is an integer tensor containing indices into the output tensor. The
/// last dimension of `indices` can be at most the rank of `shape`:
///
///     indices.shape[-1] <= shape.rank
///
/// The last dimension of `indices` corresponds to indices of elements
/// (if `indices.shape[-1] = shape.rank`) or slices
/// (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
/// `shape`.
///
/// `updates` is a tensor with shape:
///
///     indices.shape[:-1] + shape[indices.shape[-1]:]
///
/// The simplest form of the scatter op is to insert individual elements in
/// a tensor by index. Consider an example where you want to insert 4 scattered
/// elements in a rank-1 tensor with 8 elements.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd1.png" alt>
/// </div>
///
/// In Python, this scatter operation would look like this:
///
/// ```python
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     shape = tf.constant([8])
///     scatter = tf.scatter_nd(indices, updates, shape)
///     print(scatter)
/// ```
///
/// The resulting tensor would look like this:
///
///     [0, 11, 0, 10, 9, 0, 0, 12]
///
/// You can also insert entire slices of a higher rank tensor all at once. For
/// example, you can insert two slices in the first dimension of a rank-3 tensor
/// with two matrices of new values.
///
/// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
/// <img style="width:100%" src="https://www.tensorflow.org/images/ScatterNd2.png" alt>
/// </div>
///
/// In Python, this scatter operation would look like this:
///
/// ```python
///     indices = tf.constant([[1], [3]])
///     updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]],
///                            [[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]]])
///     shape = tf.constant([4, 4, 4])
///     scatter = tf.scatter_nd(indices, updates, shape)
///     print(scatter)
/// ```
///
/// The resulting tensor would look like this:
///
///     [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
///      [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
///      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
///      [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]
///
/// If `indices` contains any out-of-bound indices, depending on
/// `bad_indices_policy`, the op will either return an error or ignore the
/// out-of-bound indices. `bad_indices_policy` can be one of the following values:
/// 1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
///    historically on CPU and GPU we handle errors in different ways, and for
///    backward compatibility we keep the default behavior.
/// 2. "ERROR": raises error; GPU does not support this value.
/// 3. "IGNORE": ignore the bad indices; supported on both CPU and GPU.
///
/// Args:
/// * scope: A Scope object
/// * indices: Tensor of indices.
/// * updates: Values to scatter into the output tensor.
/// * shape: 1-D. The shape of the output tensor.
///
/// Returns:
/// * `Output`: A new tensor with the given shape and updates applied according
/// to the indices.
class ScatterNd {
 public:
  /// Optional attribute setters for ScatterNd
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    StringPiece bad_indices_policy_ = "";
  };
  ScatterNd(const ::tensorflow::Scope& scope, ::tensorflow::Input indices,
          ::tensorflow::Input updates, ::tensorflow::Input shape);
  ScatterNd(const ::tensorflow::Scope& scope, ::tensorflow::Input indices,
          ::tensorflow::Input updates, ::tensorflow::Input shape, const
          ScatterNd::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Applies sparse addition to `input` using individual values or slices
///
/// from `updates` according to indices `indices`.  The updates are non-aliasing:
/// `input` is only modified in-place if no other operations will use it.
/// Otherwise, a copy of `input` is made.  This operation has a gradient with
/// respect to both `input` and `updates`.
///
/// `input` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.
///
/// `indices` must be integer tensor, containing indices into `input`.
/// It must be shape \\([d_0, ..., d_{Q-2}, K]\\) where `0 < K <= P`.
///
/// The innermost dimension of `indices` (with length `K`) corresponds to
/// indices into elements (if `K = P`) or `(P-K)`-dimensional slices
/// (if `K < P`) along the `K`th dimension of `input`.
///
/// `updates` is `Tensor` of rank `Q-1+P-K` with shape:
///
/// $$[d_0, ..., d_{Q-2}, input.shape[K], ..., input.shape[P-1]].$$
///
/// For example, say we want to add 4 scattered elements to a rank-1 tensor to 8
/// elements. In Python, that addition would look like this:
///
///     input = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     output = tf.scatter_nd_non_aliasing_add(input, indices, updates)
///     with tf.Session() as sess:
///       print(sess.run(output))
///
/// The resulting value `output` would look like this:
///
///     [1, 13, 3, 14, 14, 6, 7, 20]
///
/// See `tf.scatter_nd` for more details about how to make updates to slices.
///
/// Args:
/// * scope: A Scope object
/// * input: A Tensor.
/// * indices: A Tensor. Must be one of the following types: `int32`, `int64`.
/// A tensor of indices into `input`.
/// * updates: A Tensor. Must have the same type as ref. A tensor of updated values
/// to add to `input`.
///
/// Returns:
/// * `Output`: A `Tensor` with the same shape as `input`, containing values of `input`
/// updated with `updates`.
class ScatterNdNonAliasingAdd {
 public:
  /// Optional attribute setters for ScatterNdNonAliasingAdd
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    StringPiece bad_indices_policy_ = "";
  };
  ScatterNdNonAliasingAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, ::tensorflow::Input indices, ::tensorflow::Input
                        updates);
  ScatterNdNonAliasingAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        input, ::tensorflow::Input indices, ::tensorflow::Input
                        updates, const ScatterNdNonAliasingAdd::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns the shape of a tensor.
///
/// This operation returns a 1-D integer tensor representing the shape of `input`.
///
/// For example:
///
/// ```
/// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
/// shape(t) ==> [2, 2, 3]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Shape {
 public:
  /// Optional attribute setters for Shape
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    DataType out_type_ = DT_INT32;
  };
  Shape(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  Shape(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
      Shape::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns shape of tensors.
///
/// This operation returns N 1-D integer tensors representing shape of `input[i]s`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class ShapeN {
 public:
  /// Optional attribute setters for ShapeN
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    DataType out_type_ = DT_INT32;
  };
  ShapeN(const ::tensorflow::Scope& scope, ::tensorflow::InputList input);
  ShapeN(const ::tensorflow::Scope& scope, ::tensorflow::InputList input, const
       ShapeN::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// Returns the size of a tensor.
///
/// This operation returns an integer representing the number of elements in
/// `input`.
///
/// For example:
///
/// ```
/// # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
/// size(t) ==> 12
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Size {
 public:
  /// Optional attribute setters for Size
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutType(DataType x) {
      Attrs ret = *this;
      ret.out_type_ = x;
      return ret;
    }

    DataType out_type_ = DT_INT32;
  };
  Size(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  Size(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
     Size::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs OutType(DataType x) {
    return Attrs().OutType(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Return a slice from 'input'.
///
/// The output tensor is a tensor with dimensions described by 'size'
/// whose values are extracted from 'input' starting at the offsets in
/// 'begin'.
///
/// *Requirements*:
///   0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)
///
/// Args:
/// * scope: A Scope object
/// * begin: begin[i] specifies the offset into the 'i'th dimension of
/// 'input' to slice from.
/// * size: size[i] specifies the number of elements of the 'i'th dimension
/// of 'input' to slice. If size[i] is -1, all remaining elements in dimension
/// i are included in the slice (i.e. this is equivalent to setting
/// size[i] = input.dim_size(i) - begin[i]).
///
/// Returns:
/// * `Output`: The output tensor.
class Slice {
 public:
  Slice(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
      ::tensorflow::Input begin, ::tensorflow::Input size);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns a copy of the input tensor.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class Snapshot {
 public:
  Snapshot(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// SpaceToBatch for 4-D tensors of type T.
///
/// This is a legacy version of the more general SpaceToBatchND.
///
/// Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
/// More specifically, this op outputs a copy of the input tensor where values from
/// the `height` and `width` dimensions are moved to the `batch` dimension. After
/// the zero-padding, both `height` and `width` of the input must be divisible by the
/// block size.
///
/// The attr `block_size` must be greater than one. It indicates the block size.
///
///   * Non-overlapping blocks of size `block_size x block size` in the height and
///     width dimensions are rearranged into the batch dimension at each location.
///   * The batch of the output tensor is `batch * block_size * block_size`.
///   * Both height_pad and width_pad must be divisible by block_size.
///
/// The shape of the output will be:
///
///     [batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
///      depth]
///
/// Some examples:
///
/// (1) For the following input of shape `[1, 2, 2, 1]` and block_size of 2:
///
/// ```
/// x = [[[[1], [2]], [[3], [4]]]]
/// ```
///
/// The output tensor has shape `[4, 1, 1, 1]` and value:
///
/// ```
/// [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
/// ```
///
/// (2) For the following input of shape `[1, 2, 2, 3]` and block_size of 2:
///
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
///
/// The output tensor has shape `[4, 1, 1, 3]` and value:
///
/// ```
/// [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
/// ```
///
/// (3) For the following input of shape `[1, 4, 4, 1]` and block_size of 2:
///
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///       [[5],   [6],  [7],  [8]],
///       [[9],  [10], [11],  [12]],
///       [[13], [14], [15],  [16]]]]
/// ```
///
/// The output tensor has shape `[4, 2, 2, 1]` and value:
///
/// ```
/// x = [[[[1], [3]], [[9], [11]]],
///      [[[2], [4]], [[10], [12]]],
///      [[[5], [7]], [[13], [15]]],
///      [[[6], [8]], [[14], [16]]]]
/// ```
///
/// (4) For the following input of shape `[2, 2, 4, 1]` and block_size of 2:
///
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///       [[5],   [6],  [7],  [8]]],
///      [[[9],  [10], [11],  [12]],
///       [[13], [14], [15],  [16]]]]
/// ```
///
/// The output tensor has shape `[8, 1, 2, 1]` and value:
///
/// ```
/// x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
///      [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
/// ```
///
/// Among others, this operation is useful for reducing atrous convolution into
/// regular convolution.
///
/// Args:
/// * scope: A Scope object
/// * input: 4-D with shape `[batch, height, width, depth]`.
/// * paddings: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
///   the padding of the input with zeros across the spatial dimensions as follows:
///
///       paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
///
///   The effective spatial dimensions of the zero-padded input tensor will be:
///
///       height_pad = pad_top + height + pad_bottom
///       width_pad = pad_left + width + pad_right
///
/// Returns:
/// * `Output`: The output tensor.
class SpaceToBatch {
 public:
  SpaceToBatch(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
             ::tensorflow::Input paddings, int64 block_size);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// SpaceToBatch for N-D tensors of type T.
///
/// This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
/// grid of blocks of shape `block_shape`, and interleaves these blocks with the
/// "batch" dimension (0) such that in the output, the spatial dimensions
/// `[1, ..., M]` correspond to the position within the grid, and the batch
/// dimension combines both the position within a spatial block and the original
/// batch position.  Prior to division into blocks, the spatial dimensions of the
/// input are optionally zero padded according to `paddings`. See below for a
/// precise description.
///
/// This operation is equivalent to the following steps:
///
/// 1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
///    input according to `paddings` to produce `padded` of shape `padded_shape`.
///
/// 2. Reshape `padded` to `reshaped_padded` of shape:
///
///      [batch] +
///      [padded_shape[1] / block_shape[0],
///        block_shape[0],
///       ...,
///       padded_shape[M] / block_shape[M-1],
///       block_shape[M-1]] +
///      remaining_shape
///
/// 3. Permute dimensions of `reshaped_padded` to produce
///    `permuted_reshaped_padded` of shape:
///
///      block_shape +
///      [batch] +
///      [padded_shape[1] / block_shape[0],
///       ...,
///       padded_shape[M] / block_shape[M-1]] +
///      remaining_shape
///
/// 4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the batch
///    dimension, producing an output tensor of shape:
///
///      [batch * prod(block_shape)] +
///      [padded_shape[1] / block_shape[0],
///       ...,
///       padded_shape[M] / block_shape[M-1]] +
///      remaining_shape
///
/// Some examples:
///
/// (1) For the following input of shape `[1, 2, 2, 1]`, `block_shape = [2, 2]`, and
///     `paddings = [[0, 0], [0, 0]]`:
///
/// ```
/// x = [[[[1], [2]], [[3], [4]]]]
/// ```
///
/// The output tensor has shape `[4, 1, 1, 1]` and value:
///
/// ```
/// [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
/// ```
///
/// (2) For the following input of shape `[1, 2, 2, 3]`, `block_shape = [2, 2]`, and
///     `paddings = [[0, 0], [0, 0]]`:
///
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
///
/// The output tensor has shape `[4, 1, 1, 3]` and value:
///
/// ```
/// [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
/// ```
///
/// (3) For the following input of shape `[1, 4, 4, 1]`, `block_shape = [2, 2]`, and
///     `paddings = [[0, 0], [0, 0]]`:
///
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///       [[5],   [6],  [7],  [8]],
///       [[9],  [10], [11],  [12]],
///       [[13], [14], [15],  [16]]]]
/// ```
///
/// The output tensor has shape `[4, 2, 2, 1]` and value:
///
/// ```
/// x = [[[[1], [3]], [[9], [11]]],
///      [[[2], [4]], [[10], [12]]],
///      [[[5], [7]], [[13], [15]]],
///      [[[6], [8]], [[14], [16]]]]
/// ```
///
/// (4) For the following input of shape `[2, 2, 4, 1]`, block_shape = `[2, 2]`, and
///     paddings = `[[0, 0], [2, 0]]`:
///
/// ```
/// x = [[[[1],   [2],  [3],  [4]],
///       [[5],   [6],  [7],  [8]]],
///      [[[9],  [10], [11],  [12]],
///       [[13], [14], [15],  [16]]]]
/// ```
///
/// The output tensor has shape `[8, 1, 3, 1]` and value:
///
/// ```
/// x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
///      [[[0], [2], [4]]], [[[0], [10], [12]]],
///      [[[0], [5], [7]]], [[[0], [13], [15]]],
///      [[[0], [6], [8]]], [[[0], [14], [16]]]]
/// ```
///
/// Among others, this operation is useful for reducing atrous convolution into
/// regular convolution.
///
/// Args:
/// * scope: A Scope object
/// * input: N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
/// where spatial_shape has `M` dimensions.
/// * block_shape: 1-D with shape `[M]`, all values must be >= 1.
/// * paddings: 2-D with shape `[M, 2]`, all values must be >= 0.
///   `paddings[i] = [pad_start, pad_end]` specifies the padding for input dimension
///   `i + 1`, which corresponds to spatial dimension `i`.  It is required that
///   `block_shape[i]` divides `input_shape[i + 1] + pad_start + pad_end`.
///
/// Returns:
/// * `Output`: The output tensor.
class SpaceToBatchND {
 public:
  SpaceToBatchND(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
               ::tensorflow::Input block_shape, ::tensorflow::Input paddings);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// SpaceToDepth for tensors of type T.
///
/// Rearranges blocks of spatial data, into depth. More specifically,
/// this op outputs a copy of the input tensor where values from the `height`
/// and `width` dimensions are moved to the `depth` dimension.
/// The attr `block_size` indicates the input block size.
///
///   * Non-overlapping blocks of size `block_size x block size` are rearranged
///     into depth at each location.
///   * The depth of the output tensor is `block_size * block_size * input_depth`.
///   * The Y, X coordinates within each block of the input become the high order
///     component of the output channel index.
///   * The input tensor's height and width must be divisible by block_size.
///
/// The `data_format` attr specifies the layout of the input and output tensors
/// with the following options:
///   "NHWC": `[ batch, height, width, channels ]`
///   "NCHW": `[ batch, channels, height, width ]`
///   "NCHW_VECT_C":
///       `qint8 [ batch, channels / 4, height, width, 4 ]`
///
/// It is useful to consider the operation as transforming a 6-D Tensor.
/// e.g. for data_format = NHWC,
///      Each element in the input tensor can be specified via 6 coordinates,
///      ordered by decreasing memory layout significance as:
///      n,oY,bY,oX,bX,iC  (where n=batch index, oX, oY means X or Y coordinates
///                         within the output image, bX, bY means coordinates
///                         within the input block, iC means input channels).
///      The output would be a transpose to the following layout:
///      n,oY,oX,bY,bX,iC
///
/// This operation is useful for resizing the activations between convolutions
/// (but keeping all data), e.g. instead of pooling. It is also useful for training
/// purely convolutional models.
///
/// For example, given an input of shape `[1, 2, 2, 1]`, data_format = "NHWC" and
/// block_size = 2:
///
/// ```
/// x = [[[[1], [2]],
///       [[3], [4]]]]
/// ```
///
/// This operation will output a tensor of shape `[1, 1, 1, 4]`:
///
/// ```
/// [[[[1, 2, 3, 4]]]]
/// ```
///
/// Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
/// the corresponding output will have a single element (i.e. width and height are
/// both 1) and will have a depth of 4 channels (1 * block_size * block_size).
/// The output element shape is `[1, 1, 4]`.
///
/// For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.
///
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
///
/// This operation, for block_size of 2, will return the following tensor of shape
/// `[1, 1, 1, 12]`
///
/// ```
/// [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
/// ```
///
/// Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:
///
/// ```
/// x = [[[[1],   [2],  [5],  [6]],
///       [[3],   [4],  [7],  [8]],
///       [[9],  [10], [13],  [14]],
///       [[11], [12], [15],  [16]]]]
/// ```
///
/// the operator will return the following tensor of shape `[1 2 2 4]`:
///
/// ```
/// x = [[[[1, 2, 3, 4],
///        [5, 6, 7, 8]],
///       [[9, 10, 11, 12],
///        [13, 14, 15, 16]]]]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * block_size: The size of the spatial block.
///
/// Returns:
/// * `Output`: The output tensor.
class SpaceToDepth {
 public:
  /// Optional attribute setters for SpaceToDepth
  struct Attrs {
    /// Defaults to "NHWC"
    TF_MUST_USE_RESULT Attrs DataFormat(StringPiece x) {
      Attrs ret = *this;
      ret.data_format_ = x;
      return ret;
    }

    StringPiece data_format_ = "NHWC";
  };
  SpaceToDepth(const ::tensorflow::Scope& scope, ::tensorflow::Input input, int64
             block_size);
  SpaceToDepth(const ::tensorflow::Scope& scope, ::tensorflow::Input input, int64
             block_size, const SpaceToDepth::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs DataFormat(StringPiece x) {
    return Attrs().DataFormat(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Splits a tensor into `num_split` tensors along one dimension.
///
/// Args:
/// * scope: A Scope object
/// * axis: 0-D.  The dimension along which to split.  Must be in the range
/// `[-rank(value), rank(value))`.
/// * value: The tensor to split.
/// * num_split: The number of ways to split.  Must evenly divide
/// `value.shape[split_dim]`.
///
/// Returns:
/// * `OutputList`: They are identically shaped tensors, whose shape matches that of `value`
/// except along `axis`, where their sizes are
/// `values.shape[split_dim] / num_split`.
class Split {
 public:
  Split(const ::tensorflow::Scope& scope, ::tensorflow::Input axis,
      ::tensorflow::Input value, int64 num_split);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// Splits a tensor into `num_split` tensors along one dimension.
///
/// Args:
/// * scope: A Scope object
/// * value: The tensor to split.
/// * size_splits: list containing the sizes of each output tensor along the split
/// dimension. Must sum to the dimension of value along split_dim.
/// Can contain one -1 indicating that dimension is to be inferred.
/// * axis: 0-D.  The dimension along which to split.  Must be in the range
/// `[-rank(value), rank(value))`.
///
/// Returns:
/// * `OutputList`: Tensors whose shape matches that of `value`
/// except along `axis`, where their sizes are
/// `size_splits[i]`.
class SplitV {
 public:
  SplitV(const ::tensorflow::Scope& scope, ::tensorflow::Input value,
       ::tensorflow::Input size_splits, ::tensorflow::Input axis, int64
       num_split);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  Operation operation;
  ::tensorflow::OutputList output;
};

/// Removes dimensions of size 1 from the shape of a tensor.
///
/// Given a tensor `input`, this operation returns a tensor of the same type with
/// all dimensions of size 1 removed. If you don't want to remove all size 1
/// dimensions, you can remove specific size 1 dimensions by specifying
/// `axis`.
///
/// For example:
///
/// ```
/// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
/// shape(squeeze(t)) ==> [2, 3]
/// ```
///
/// Or, to remove specific size 1 dimensions:
///
/// ```
/// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
/// shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * input: The `input` to squeeze.
///
/// Optional attributes (see `Attrs`):
/// * axis: If specified, only squeezes the dimensions listed. The dimension
/// index starts at 0. It is an error to squeeze a dimension that is not 1. Must
/// be in the range `[-rank(input), rank(input))`.
///
/// Returns:
/// * `Output`: Contains the same data as `input`, but has one or more dimensions of
/// size 1 removed.
class Squeeze {
 public:
  /// Optional attribute setters for Squeeze
  struct Attrs {
    /// If specified, only squeezes the dimensions listed. The dimension
    /// index starts at 0. It is an error to squeeze a dimension that is not 1. Must
    /// be in the range `[-rank(input), rank(input))`.
    ///
    /// Defaults to []
    TF_MUST_USE_RESULT Attrs Axis(const gtl::ArraySlice<int>& x) {
      Attrs ret = *this;
      ret.axis_ = x;
      return ret;
    }

    gtl::ArraySlice<int> axis_ = {};
  };
  Squeeze(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  Squeeze(const ::tensorflow::Scope& scope, ::tensorflow::Input input, const
        Squeeze::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Axis(const gtl::ArraySlice<int>& x) {
    return Attrs().Axis(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Stops gradient computation.
///
/// When executed in a graph, this op outputs its input tensor as-is.
///
/// When building ops to compute gradients, this op prevents the contribution of
/// its inputs to be taken into account.  Normally, the gradient generator adds ops
/// to a graph to compute the derivatives of a specified 'loss' by recursively
/// finding out inputs that contributed to its computation.  If you insert this op
/// in the graph it inputs are masked from the gradient generator.  They are not
/// taken into account for computing gradients.
///
/// This is useful any time you want to compute a value with TensorFlow but need
/// to pretend that the value was a constant. For example, the softmax function
/// for a vector x can be written as
///
/// ```python
///
///   def softmax(x):
///     numerator = tf.exp(x)
///     denominator = tf.reduce_sum(numerator)
///     return numerator / denominator
/// ```
///
/// This however is susceptible to overflow if the values in x are large. An
/// alternative more stable way is to subtract the maximum of x from each of the
/// values.
///
/// ```python
///
///   def stable_softmax(x):
///     z = x - tf.reduce_max(x)
///     numerator = tf.exp(z)
///     denominator = tf.reduce_sum(numerator)
///     return numerator / denominator
/// ```
///
/// However, when we backprop through the softmax to x, we dont want to backprop
/// through the `tf.reduce_max(x)` (if the max values are not unique then the
/// gradient could flow to the wrong input) calculation and treat that as a
/// constant. Therefore, we should write this out as
///
/// ```python
///
///   def stable_softmax(x):
///     z = x - tf.stop_gradient(tf.reduce_max(x))
///     numerator = tf.exp(z)
///     denominator = tf.reduce_sum(numerator)
///     return numerator / denominator
/// ```
///
/// Some other examples include:
///
/// *  The *EM* algorithm where the *M-step* should not involve backpropagation
///    through the output of the *E-step*.
/// *  Contrastive divergence training of Boltzmann machines where, when
///    differentiating the energy function, the training must not backpropagate
///    through the graph that generated the samples from the model.
/// *  Adversarial training, where no backprop should happen through the adversarial
///    example generation process.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class StopGradient {
 public:
  StopGradient(const ::tensorflow::Scope& scope, ::tensorflow::Input input);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Return a strided slice from `input`.
///
/// Note, most python users will want to use the Python `Tensor.__getitem__`
/// or `Variable.__getitem__` rather than this op directly.
///
/// The goal of this op is to produce a new tensor with a subset of
/// the elements from the `n` dimensional `input` tensor. The subset is chosen using
/// a sequence of `m` sparse range specifications encoded into the arguments
/// of this function. Note, in some cases
/// `m` could be equal to `n`, but this need not be the case. Each
/// range specification entry can be one of the following:
///
/// - An ellipsis (...). Ellipses are used to imply zero or more
///   dimensions of full-dimension selection and are produced using
///   `ellipsis_mask`. For example, `foo[...]` is the identity slice.
///
/// - A new axis. This is used to insert a new shape=1 dimension and is
///   produced using `new_axis_mask`. For example, `foo[:, ...]` where
///   `foo` is shape `(3, 4)` produces a `(1, 3, 4)` tensor.
///
///
/// - A range `begin:end:stride`. This is used to specify how much to choose from
///   a given dimension. `stride` can be any integer but 0.  `begin` is an integer
///   which represents the index of the first value to select while `end` represents
///   the index of the last value to select. The number of values selected in each
///   dimension is `end - begin` if `stride > 0` and `begin - end` if `stride < 0`.
///   `begin` and `end` can be negative where `-1` is the last element, `-2` is
///   the second to last. `begin_mask` controls whether to replace the explicitly
///   given `begin` with an implicit effective value of `0` if `stride > 0` and
///   `-1` if `stride < 0`. `end_mask` is analogous but produces the number
///   required to create the largest open interval. For example, given a shape
///   `(3,)` tensor `foo[:]`, the effective `begin` and `end` are `0` and `3`. Do
///   not assume this is equivalent to `foo[0:-1]` which has an effective `begin`
///   and `end` of `0` and `2`. Another example is `foo[-2::-1]` which reverses the
///   first dimension of a tensor while dropping the last two (in the original
///   order elements). For example `foo = [1,2,3,4]; foo[-2::-1]` is `[4,3]`.
///
/// - A single index. This is used to keep only elements that have a given
///   index. For example (`foo[2, :]` on a shape `(5,6)` tensor produces a
///   shape `(6,)` tensor. This is encoded in `begin` and `end` and
///   `shrink_axis_mask`.
///
/// Each conceptual range specification is encoded in the op's argument. This
/// encoding is best understand by considering a non-trivial example. In
/// particular,
/// `foo[1, 2:4, None, ..., :-3:-1, :]` will be encoded as
///
/// ```
/// begin = [1, 2, x, x, 0, x] # x denotes don't care (usually 0)
/// end = [2, 4, x, x, -3, x]
/// strides = [1, 1, x, x, -1, 1]
/// begin_mask = 1<<4 | 1<<5 = 48
/// end_mask = 1<<5 = 32
/// ellipsis_mask = 1<<3 = 8
/// new_axis_mask = 1<<2 = 4
/// shrink_axis_mask = 1<<0 = 1
/// ```
///
/// In this case if `foo.shape` is (5, 5, 5, 5, 5, 5) the final shape of
/// the slice becomes (2, 1, 5, 5, 2, 5).
/// Let us walk step by step through each argument specification.
///
/// 1.  The first argument in the example slice is turned into `begin = 1` and
/// `end = begin + 1 = 2`. To disambiguate from the original spec `2:4` we
/// also set the appropriate bit in `shrink_axis_mask`.
///
/// 2. `2:4` is contributes 2, 4, 1 to begin, end, and stride. All masks have
/// zero bits contributed.
///
/// 3. None is a synonym for `tf.newaxis`. This means insert a dimension of size 1
/// dimension in the final shape. Dummy values are contributed to begin,
/// end and stride, while the new_axis_mask bit is set.
///
/// 4. `...` grab the full ranges from as many dimensions as needed to
/// fully specify a slice for every dimension of the input shape.
///
/// 5. `:-3:-1` shows the use of negative indices. A negative index `i` associated
/// with a dimension that has shape `s` is converted to a positive index
/// `s + i`. So `-1` becomes `s-1` (i.e. the last element). This conversion
/// is done internally so begin, end and strides receive x, -3, and -1.
/// The appropriate begin_mask bit is set to indicate the start range is the
/// full range (ignoring the x).
///
/// 6. `:` indicates that the entire contents of the corresponding dimension
/// is selected. This is equivalent to `::` or `0::1`. begin, end, and strides
/// receive 0, 0, and 1, respectively. The appropriate bits in `begin_mask` and
/// `end_mask` are also set.
///
/// *Requirements*:
///   `0 != strides[i] for i in [0, m)`
///   `ellipsis_mask must be a power of two (only one ellipsis)`
///
/// Args:
/// * scope: A Scope object
/// * begin: `begin[k]` specifies the offset into the `k`th range specification.
/// The exact dimension this corresponds to will be determined by context.
/// Out-of-bounds values will be silently clamped. If the `k`th bit of
/// `begin_mask` then `begin[k]` is ignored and the full range of the
/// appropriate dimension is used instead. Negative values causes indexing
/// to start from the highest element e.g. If `foo==[1,2,3]` then `foo[-1]==3`.
/// * end: `end[i]` is like `begin` with the exception that `end_mask` is
/// used to determine full ranges.
/// * strides: `strides[i]` specifies the increment in the `i`th specification
/// after extracting a given element. Negative indices will reverse
/// the original order. Out or range values are
/// clamped to `[0,dim[i]) if slice[i]>0` or `[-1,dim[i]-1] if slice[i] < 0`
///
/// Optional attributes (see `Attrs`):
/// * begin_mask: a bitmask where a bit i being 1 means to ignore the begin
/// value and instead use the largest interval possible. At runtime
/// begin[i] will be replaced with `[0, n-1)` if `stride[i] > 0` or
/// `[-1, n-1]` if `stride[i] < 0`
/// * end_mask: analogous to `begin_mask`
/// * ellipsis_mask: a bitmask where bit `i` being 1 means the `i`th
/// position is actually an ellipsis. One bit at most can be 1.
/// If `ellipsis_mask == 0`, then an implicit ellipsis mask of `1 << (m+1)`
/// is provided. This means that `foo[3:5] == foo[3:5, ...]`. An ellipsis
/// implicitly creates as many range specifications as necessary to fully
/// specify the sliced range for every dimension. For example for a 4-dimensional
/// tensor `foo` the slice `foo[2, ..., 5:8]` implies `foo[2, :, :, 5:8]`.
/// * new_axis_mask: a bitmask where bit `i` being 1 means the `i`th
/// specification creates a new shape 1 dimension. For example
/// `foo[:4, tf.newaxis, :2]` would produce a shape `(4, 1, 2)` tensor.
/// * shrink_axis_mask: a bitmask where bit `i` implies that the `i`th
/// specification should shrink the dimensionality. begin and end
/// must imply a slice of size 1 in the dimension. For example in
/// python one might do `foo[:, 3, :]` which would result in
/// `shrink_axis_mask` being 2.
///
/// Returns:
/// * `Output`: The output tensor.
class StridedSlice {
 public:
  /// Optional attribute setters for StridedSlice
  struct Attrs {
    /// a bitmask where a bit i being 1 means to ignore the begin
    /// value and instead use the largest interval possible. At runtime
    /// begin[i] will be replaced with `[0, n-1)` if `stride[i] > 0` or
    /// `[-1, n-1]` if `stride[i] < 0`
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs BeginMask(int64 x) {
      Attrs ret = *this;
      ret.begin_mask_ = x;
      return ret;
    }

    /// analogous to `begin_mask`
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs EndMask(int64 x) {
      Attrs ret = *this;
      ret.end_mask_ = x;
      return ret;
    }

    /// a bitmask where bit `i` being 1 means the `i`th
    /// position is actually an ellipsis. One bit at most can be 1.
    /// If `ellipsis_mask == 0`, then an implicit ellipsis mask of `1 << (m+1)`
    /// is provided. This means that `foo[3:5] == foo[3:5, ...]`. An ellipsis
    /// implicitly creates as many range specifications as necessary to fully
    /// specify the sliced range for every dimension. For example for a 4-dimensional
    /// tensor `foo` the slice `foo[2, ..., 5:8]` implies `foo[2, :, :, 5:8]`.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs EllipsisMask(int64 x) {
      Attrs ret = *this;
      ret.ellipsis_mask_ = x;
      return ret;
    }

    /// a bitmask where bit `i` being 1 means the `i`th
    /// specification creates a new shape 1 dimension. For example
    /// `foo[:4, tf.newaxis, :2]` would produce a shape `(4, 1, 2)` tensor.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NewAxisMask(int64 x) {
      Attrs ret = *this;
      ret.new_axis_mask_ = x;
      return ret;
    }

    /// a bitmask where bit `i` implies that the `i`th
    /// specification should shrink the dimensionality. begin and end
    /// must imply a slice of size 1 in the dimension. For example in
    /// python one might do `foo[:, 3, :]` which would result in
    /// `shrink_axis_mask` being 2.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs ShrinkAxisMask(int64 x) {
      Attrs ret = *this;
      ret.shrink_axis_mask_ = x;
      return ret;
    }

    int64 begin_mask_ = 0;
    int64 end_mask_ = 0;
    int64 ellipsis_mask_ = 0;
    int64 new_axis_mask_ = 0;
    int64 shrink_axis_mask_ = 0;
  };
  StridedSlice(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
             ::tensorflow::Input begin, ::tensorflow::Input end,
             ::tensorflow::Input strides);
  StridedSlice(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
             ::tensorflow::Input begin, ::tensorflow::Input end,
             ::tensorflow::Input strides, const StridedSlice::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BeginMask(int64 x) {
    return Attrs().BeginMask(x);
  }
  static Attrs EndMask(int64 x) {
    return Attrs().EndMask(x);
  }
  static Attrs EllipsisMask(int64 x) {
    return Attrs().EllipsisMask(x);
  }
  static Attrs NewAxisMask(int64 x) {
    return Attrs().NewAxisMask(x);
  }
  static Attrs ShrinkAxisMask(int64 x) {
    return Attrs().ShrinkAxisMask(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Assign `value` to the sliced l-value reference of `ref`.
///
/// The values of `value` are assigned to the positions in the variable
/// `ref` that are selected by the slice parameters. The slice parameters
/// `begin`, `end`, `strides`, etc. work exactly as in `StridedSlice`.
///
/// NOTE this op currently does not support broadcasting and so `value`'s
/// shape must be exactly the shape produced by the slice of `ref`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output_ref tensor.
class StridedSliceAssign {
 public:
  /// Optional attribute setters for StridedSliceAssign
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs BeginMask(int64 x) {
      Attrs ret = *this;
      ret.begin_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs EndMask(int64 x) {
      Attrs ret = *this;
      ret.end_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs EllipsisMask(int64 x) {
      Attrs ret = *this;
      ret.ellipsis_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NewAxisMask(int64 x) {
      Attrs ret = *this;
      ret.new_axis_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs ShrinkAxisMask(int64 x) {
      Attrs ret = *this;
      ret.shrink_axis_mask_ = x;
      return ret;
    }

    int64 begin_mask_ = 0;
    int64 end_mask_ = 0;
    int64 ellipsis_mask_ = 0;
    int64 new_axis_mask_ = 0;
    int64 shrink_axis_mask_ = 0;
  };
  StridedSliceAssign(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                   ::tensorflow::Input begin, ::tensorflow::Input end,
                   ::tensorflow::Input strides, ::tensorflow::Input value);
  StridedSliceAssign(const ::tensorflow::Scope& scope, ::tensorflow::Input ref,
                   ::tensorflow::Input begin, ::tensorflow::Input end,
                   ::tensorflow::Input strides, ::tensorflow::Input value,
                   const StridedSliceAssign::Attrs& attrs);
  operator ::tensorflow::Output() const { return output_ref; }
  operator ::tensorflow::Input() const { return output_ref; }
  ::tensorflow::Node* node() const { return output_ref.node(); }

  static Attrs BeginMask(int64 x) {
    return Attrs().BeginMask(x);
  }
  static Attrs EndMask(int64 x) {
    return Attrs().EndMask(x);
  }
  static Attrs EllipsisMask(int64 x) {
    return Attrs().EllipsisMask(x);
  }
  static Attrs NewAxisMask(int64 x) {
    return Attrs().NewAxisMask(x);
  }
  static Attrs ShrinkAxisMask(int64 x) {
    return Attrs().ShrinkAxisMask(x);
  }

  Operation operation;
  ::tensorflow::Output output_ref;
};

/// Returns the gradient of `StridedSlice`.
///
/// Since `StridedSlice` cuts out pieces of its `input` which is size
/// `shape`, its gradient will have the same shape (which is passed here
/// as `shape`). The gradient will be zero in any element that the slice
/// does not select.
///
/// Arguments are the same as StridedSliceGrad with the exception that
/// `dy` is the input gradient to be propagated and `shape` is the
/// shape of `StridedSlice`'s `input`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class StridedSliceGrad {
 public:
  /// Optional attribute setters for StridedSliceGrad
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs BeginMask(int64 x) {
      Attrs ret = *this;
      ret.begin_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs EndMask(int64 x) {
      Attrs ret = *this;
      ret.end_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs EllipsisMask(int64 x) {
      Attrs ret = *this;
      ret.ellipsis_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NewAxisMask(int64 x) {
      Attrs ret = *this;
      ret.new_axis_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs ShrinkAxisMask(int64 x) {
      Attrs ret = *this;
      ret.shrink_axis_mask_ = x;
      return ret;
    }

    int64 begin_mask_ = 0;
    int64 end_mask_ = 0;
    int64 ellipsis_mask_ = 0;
    int64 new_axis_mask_ = 0;
    int64 shrink_axis_mask_ = 0;
  };
  StridedSliceGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
                 ::tensorflow::Input begin, ::tensorflow::Input end,
                 ::tensorflow::Input strides, ::tensorflow::Input dy);
  StridedSliceGrad(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
                 ::tensorflow::Input begin, ::tensorflow::Input end,
                 ::tensorflow::Input strides, ::tensorflow::Input dy, const
                 StridedSliceGrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BeginMask(int64 x) {
    return Attrs().BeginMask(x);
  }
  static Attrs EndMask(int64 x) {
    return Attrs().EndMask(x);
  }
  static Attrs EllipsisMask(int64 x) {
    return Attrs().EllipsisMask(x);
  }
  static Attrs NewAxisMask(int64 x) {
    return Attrs().NewAxisMask(x);
  }
  static Attrs ShrinkAxisMask(int64 x) {
    return Attrs().ShrinkAxisMask(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Adds sparse `updates` to an existing tensor according to `indices`.
///
/// This operation creates a new tensor by adding sparse `updates` to the passed
/// in `tensor`.
/// This operation is very similar to `tf.compat.v1.scatter_nd_add`, except that the
/// updates are added onto an existing tensor (as opposed to a variable). If the
/// memory for the existing tensor cannot be re-used, a copy is made and updated.
///
/// `indices` is an integer tensor containing indices into a new tensor of shape
/// `tensor.shape`.  The last dimension of `indices` can be at most the rank of
/// `tensor.shape`:
///
/// ```
/// indices.shape[-1] <= tensor.shape.rank
/// ```
///
/// The last dimension of `indices` corresponds to indices into elements
/// (if `indices.shape[-1] = tensor.shape.rank`) or slices
/// (if `indices.shape[-1] < tensor.shape.rank`) along dimension
/// `indices.shape[-1]` of `tensor.shape`.  `updates` is a tensor with shape
///
/// ```
/// indices.shape[:-1] + tensor.shape[indices.shape[-1]:]
/// ```
///
/// The simplest form of `tensor_scatter_nd_add` is to add individual elements to a
/// tensor by index. For example, say we want to add 4 elements in a rank-1
/// tensor with 8 elements.
///
/// In Python, this scatter add operation would look like this:
///
/// >>> indices = tf.constant([[4], [3], [1], [7]])
/// >>> updates = tf.constant([9, 10, 11, 12])
/// >>> tensor = tf.ones([8], dtype=tf.int32)
/// >>> updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
/// >>> updated
/// <tf.Tensor: shape=(8,), dtype=int32,
/// numpy=array([ 1, 12,  1, 11, 10,  1,  1, 13], dtype=int32)>
///
/// We can also, insert entire slices of a higher rank tensor all at once. For
/// example, if we wanted to insert two slices in the first dimension of a
/// rank-3 tensor with two matrices of new values.
///
/// In Python, this scatter add operation would look like this:
///
/// >>> indices = tf.constant([[0], [2]])
/// >>> updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
/// ...                         [7, 7, 7, 7], [8, 8, 8, 8]],
/// ...                        [[5, 5, 5, 5], [6, 6, 6, 6],
/// ...                         [7, 7, 7, 7], [8, 8, 8, 8]]])
/// >>> tensor = tf.ones([4, 4, 4],dtype=tf.int32)
/// >>> updated = tf.tensor_scatter_nd_add(tensor, indices, updates)
/// >>> updated
/// <tf.Tensor: shape=(4, 4, 4), dtype=int32,
/// numpy=array([[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
///              [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
///              [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
///              [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]], dtype=int32)>
///
///
/// If `indices` contains any out-of-bound indices, depending on
/// `bad_indices_policy`, the op will either return an error or ignore the
/// out-of-bound indices. `bad_indices_policy` can be one of the following values:
/// 1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
///    historically on CPU and GPU we handle errors in different ways, and for
///    backward compatibility we keep the default behavior.
/// 2. "ERROR": raises error; GPU does not support this value.
/// 3. "IGNORE": ignore the bad indices; supported on both CPU and GPU.
///
///
/// Args:
/// * scope: A Scope object
/// * tensor: Tensor to copy/update.
/// * indices: Index tensor.
/// * updates: Updates to scatter into output.
///
/// Returns:
/// * `Output`: A new tensor copied from tensor and updates added according to the indices.
class TensorScatterAdd {
 public:
  /// Optional attribute setters for TensorScatterAdd
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    StringPiece bad_indices_policy_ = "";
  };
  TensorScatterAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
                 ::tensorflow::Input indices, ::tensorflow::Input updates);
  TensorScatterAdd(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
                 ::tensorflow::Input indices, ::tensorflow::Input updates,
                 const TensorScatterAdd::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Apply a sparse update to a tensor taking the element-wise maximum.
///
/// Returns a new tensor copied from `tensor` whose values are element-wise maximum between
/// tensor and updates according to the indices.
///
/// >>> tensor = [0, 0, 0, 0, 0, 0, 0, 0]
/// >>> indices = [[1], [4], [5]]
/// >>> updates = [1, -1, 1]
/// >>> tf.tensor_scatter_nd_max(tensor, indices, updates).numpy()
/// array([0, 1, 0, 0, 0, 1, 0, 0], dtype=int32)
///
/// Refer to `tf.tensor_scatter_nd_update` for more details.
///
/// Args:
/// * scope: A Scope object
/// * tensor: Tensor to update.
/// * indices: Index tensor.
/// * updates: Updates to scatter into output.
///
/// Returns:
/// * `Output`: A new tensor copied from tensor whose values are element-wise maximum between tensor and updates according to the indices.
class TensorScatterMax {
 public:
  /// Optional attribute setters for TensorScatterMax
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    StringPiece bad_indices_policy_ = "";
  };
  TensorScatterMax(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
                 ::tensorflow::Input indices, ::tensorflow::Input updates);
  TensorScatterMax(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
                 ::tensorflow::Input indices, ::tensorflow::Input updates,
                 const TensorScatterMax::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// TODO: add doc.
///
/// Args:
/// * scope: A Scope object
/// * tensor: Tensor to update.
/// * indices: Index tensor.
/// * updates: Updates to scatter into output.
///
/// Returns:
/// * `Output`: A new tensor copied from tensor whose values are element-wise minimum between tensor and updates according to the indices.
class TensorScatterMin {
 public:
  /// Optional attribute setters for TensorScatterMin
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    StringPiece bad_indices_policy_ = "";
  };
  TensorScatterMin(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
                 ::tensorflow::Input indices, ::tensorflow::Input updates);
  TensorScatterMin(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
                 ::tensorflow::Input indices, ::tensorflow::Input updates,
                 const TensorScatterMin::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Subtracts sparse `updates` from an existing tensor according to `indices`.
///
/// This operation creates a new tensor by subtracting sparse `updates` from the
/// passed in `tensor`.
/// This operation is very similar to `tf.scatter_nd_sub`, except that the updates
/// are subtracted from an existing tensor (as opposed to a variable). If the memory
/// for the existing tensor cannot be re-used, a copy is made and updated.
///
/// `indices` is an integer tensor containing indices into a new tensor of shape
/// `shape`.  The last dimension of `indices` can be at most the rank of `shape`:
///
///     indices.shape[-1] <= shape.rank
///
/// The last dimension of `indices` corresponds to indices into elements
/// (if `indices.shape[-1] = shape.rank`) or slices
/// (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
/// `shape`.  `updates` is a tensor with shape
///
///     indices.shape[:-1] + shape[indices.shape[-1]:]
///
/// The simplest form of tensor_scatter_sub is to subtract individual elements
/// from a tensor by index. For example, say we want to insert 4 scattered elements
/// in a rank-1 tensor with 8 elements.
///
/// In Python, this scatter subtract operation would look like this:
///
/// ```python
///     indices = tf.constant([[4], [3], [1], [7]])
///     updates = tf.constant([9, 10, 11, 12])
///     tensor = tf.ones([8], dtype=tf.int32)
///     updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
///     print(updated)
/// ```
///
/// The resulting tensor would look like this:
///
///     [1, -10, 1, -9, -8, 1, 1, -11]
///
/// We can also, insert entire slices of a higher rank tensor all at once. For
/// example, if we wanted to insert two slices in the first dimension of a
/// rank-3 tensor with two matrices of new values.
///
/// In Python, this scatter add operation would look like this:
///
/// ```python
///     indices = tf.constant([[0], [2]])
///     updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]],
///                            [[5, 5, 5, 5], [6, 6, 6, 6],
///                             [7, 7, 7, 7], [8, 8, 8, 8]]])
///     tensor = tf.ones([4, 4, 4],dtype=tf.int32)
///     updated = tf.tensor_scatter_nd_sub(tensor, indices, updates)
///     print(updated)
/// ```
///
/// The resulting tensor would look like this:
///
///     [[[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
///      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
///      [[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
///      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
///
/// Note that on CPU, if an out of bound index is found, an error is returned.
/// On GPU, if an out of bound index is found, the index is ignored.
///
/// Args:
/// * scope: A Scope object
/// * tensor: Tensor to copy/update.
/// * indices: Index tensor.
/// * updates: Updates to scatter into output.
///
/// Returns:
/// * `Output`: A new tensor copied from tensor and updates subtracted according to the indices.
class TensorScatterSub {
 public:
  /// Optional attribute setters for TensorScatterSub
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    StringPiece bad_indices_policy_ = "";
  };
  TensorScatterSub(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
                 ::tensorflow::Input indices, ::tensorflow::Input updates);
  TensorScatterSub(const ::tensorflow::Scope& scope, ::tensorflow::Input tensor,
                 ::tensorflow::Input indices, ::tensorflow::Input updates,
                 const TensorScatterSub::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Scatter `updates` into an existing tensor according to `indices`.
///
/// This operation creates a new tensor by applying sparse `updates` to the passed
/// in `tensor`.
/// This operation is very similar to `tf.scatter_nd`, except that the updates are
/// scattered onto an existing tensor (as opposed to a zero-tensor). If the memory
/// for the existing tensor cannot be re-used, a copy is made and updated.
///
/// If `indices` contains duplicates, then we pick the last update for the index.
///
/// **WARNING**: There are some GPU specific semantics for this operation.
/// - If an out of bound index is found, the index is ignored.
/// - The order in which updates are applied is nondeterministic, so the output
/// will be nondeterministic if `indices` contains duplicates.
///
/// `indices` is an integer tensor containing indices into a new tensor of shape
/// `shape`.
///
/// * `indices` must have at least 2 axes: `(num_updates, index_depth)`.
/// * The last axis of `indices` is how deep to index into `tensor` so  this index
///   depth must be less than the rank of `tensor`: `indices.shape[-1] <= tensor.ndim`
///
/// if `indices.shape[-1] = tensor.rank` this Op indexes and updates scalar elements.
/// if `indices.shape[-1] < tensor.rank` it indexes and updates slices of the input
/// `tensor`.
///
/// Each `update` has a rank of `tensor.rank - indices.shape[-1]`.
/// The overall shape of `updates` is:
///
/// ```
/// indices.shape[:-1] + tensor.shape[indices.shape[-1]:]
/// ```
///
/// If `indices` contains any out-of-bound indices, depending on
/// `bad_indices_policy`, the op will either return an error or ignore the
/// out-of-bound indices. `bad_indices_policy` can be one of the following values:
/// 1. "" or "DEFAULT": raises on CPU and ignore on GPU. This is because
///    historically on CPU and GPU we handle errors in different ways, and for
///    backward compatibility we keep the default behavior.
/// 2. "ERROR": raises error; GPU does not support this value.
/// 3. "IGNORE": ignore the bad indices; supported on both CPU and GPU.
///
/// For usage examples see the python [tf.tensor_scatter_nd_update](
/// https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update) function
///
///
/// Args:
/// * scope: A Scope object
/// * tensor: Tensor to copy/update.
/// * indices: Index tensor.
/// * updates: Updates to scatter into output.
///
/// Returns:
/// * `Output`: A new tensor with the given shape and updates applied according
/// to the indices.
class TensorScatterUpdate {
 public:
  /// Optional attribute setters for TensorScatterUpdate
  struct Attrs {
    /// Defaults to ""
    TF_MUST_USE_RESULT Attrs BadIndicesPolicy(StringPiece x) {
      Attrs ret = *this;
      ret.bad_indices_policy_ = x;
      return ret;
    }

    StringPiece bad_indices_policy_ = "";
  };
  TensorScatterUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    tensor, ::tensorflow::Input indices, ::tensorflow::Input
                    updates);
  TensorScatterUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input
                    tensor, ::tensorflow::Input indices, ::tensorflow::Input
                    updates, const TensorScatterUpdate::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BadIndicesPolicy(StringPiece x) {
    return Attrs().BadIndicesPolicy(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Assign `value` to the sliced l-value reference of `input`.
///
/// The values of `value` are assigned to the positions in the tensor `input` that
/// are selected by the slice parameters. The slice parameters `begin` `end`
/// `strides` etc. work exactly as in `StridedSlice`.
///
/// NOTE this op currently does not support broadcasting and so `value`'s shape
/// must be exactly the shape produced by the slice of `input`.
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output tensor.
class TensorStridedSliceUpdate {
 public:
  /// Optional attribute setters for TensorStridedSliceUpdate
  struct Attrs {
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs BeginMask(int64 x) {
      Attrs ret = *this;
      ret.begin_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs EndMask(int64 x) {
      Attrs ret = *this;
      ret.end_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs EllipsisMask(int64 x) {
      Attrs ret = *this;
      ret.ellipsis_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs NewAxisMask(int64 x) {
      Attrs ret = *this;
      ret.new_axis_mask_ = x;
      return ret;
    }

    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs ShrinkAxisMask(int64 x) {
      Attrs ret = *this;
      ret.shrink_axis_mask_ = x;
      return ret;
    }

    int64 begin_mask_ = 0;
    int64 end_mask_ = 0;
    int64 ellipsis_mask_ = 0;
    int64 new_axis_mask_ = 0;
    int64 shrink_axis_mask_ = 0;
  };
  TensorStridedSliceUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input, ::tensorflow::Input begin, ::tensorflow::Input
                         end, ::tensorflow::Input strides, ::tensorflow::Input
                         value);
  TensorStridedSliceUpdate(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         input, ::tensorflow::Input begin, ::tensorflow::Input
                         end, ::tensorflow::Input strides, ::tensorflow::Input
                         value, const TensorStridedSliceUpdate::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs BeginMask(int64 x) {
    return Attrs().BeginMask(x);
  }
  static Attrs EndMask(int64 x) {
    return Attrs().EndMask(x);
  }
  static Attrs EllipsisMask(int64 x) {
    return Attrs().EllipsisMask(x);
  }
  static Attrs NewAxisMask(int64 x) {
    return Attrs().NewAxisMask(x);
  }
  static Attrs ShrinkAxisMask(int64 x) {
    return Attrs().ShrinkAxisMask(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Constructs a tensor by tiling a given tensor.
///
/// This operation creates a new tensor by replicating `input` `multiples` times.
/// The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
/// and the values of `input` are replicated `multiples[i]` times along the 'i'th
/// dimension. For example, tiling `[a b c d]` by `[2]` produces
/// `[a b c d a b c d]`.
///
/// >>> a = tf.constant([[1,2,3],[4,5,6]], tf.int32)
/// >>> b = tf.constant([1,2], tf.int32)
/// >>> tf.tile(a, b)
/// <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
/// array([[1, 2, 3, 1, 2, 3],
///        [4, 5, 6, 4, 5, 6]], dtype=int32)>
/// >>> c = tf.constant([2,1], tf.int32)
/// >>> tf.tile(a, c)
/// <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
/// array([[1, 2, 3],
///        [4, 5, 6],
///        [1, 2, 3],
///        [4, 5, 6]], dtype=int32)>
/// >>> d = tf.constant([2,2], tf.int32)
/// >>> tf.tile(a, d)
/// <tf.Tensor: shape=(4, 6), dtype=int32, numpy=
/// array([[1, 2, 3, 1, 2, 3],
///        [4, 5, 6, 4, 5, 6],
///        [1, 2, 3, 1, 2, 3],
///        [4, 5, 6, 4, 5, 6]], dtype=int32)>
///
/// Args:
/// * scope: A Scope object
/// * input: Can be of any rank.
/// * multiples: 1-D. Length must be the same as the number of dimensions in `input`
///
/// Returns:
/// * `Output`: The output tensor.
class Tile {
 public:
  Tile(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
     ::tensorflow::Input multiples);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Shuffle dimensions of x according to a permutation.
///
/// The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
///   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The y tensor.
class Transpose {
 public:
  Transpose(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
          ::tensorflow::Input perm);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// Finds unique elements in a 1-D tensor.
///
/// This operation returns a tensor `y` containing all of the unique elements of `x`
/// sorted in the same order that they occur in `x`; `x` does not need to be sorted.
/// This operation also returns a tensor `idx` the same size as `x` that contains
/// the index of each value of `x` in the unique output `y`. In other words:
///
/// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
///
/// Examples:
///
/// ```
/// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
/// y, idx = unique(x)
/// y ==> [1, 2, 4, 7, 8]
/// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
/// ```
///
/// ```
/// # tensor 'x' is [4, 5, 1, 2, 3, 3, 4, 5]
/// y, idx = unique(x)
/// y ==> [4, 5, 1, 2, 3]
/// idx ==> [0, 1, 2, 3, 4, 4, 0, 1]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * x: 1-D.
///
/// Returns:
/// * `Output` y: 1-D.
/// * `Output` idx: 1-D.
class Unique {
 public:
  /// Optional attribute setters for Unique
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutIdx(DataType x) {
      Attrs ret = *this;
      ret.out_idx_ = x;
      return ret;
    }

    DataType out_idx_ = DT_INT32;
  };
  Unique(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  Unique(const ::tensorflow::Scope& scope, ::tensorflow::Input x, const
       Unique::Attrs& attrs);

  static Attrs OutIdx(DataType x) {
    return Attrs().OutIdx(x);
  }

  Operation operation;
  ::tensorflow::Output y;
  ::tensorflow::Output idx;
};

/// Finds unique elements along an axis of a tensor.
///
/// This operation either returns a tensor `y` containing unique elements
/// along the `axis` of a tensor. The returned unique elements is sorted
/// in the same order as they occur along `axis` in `x`.
/// This operation also returns a tensor `idx` that is the same size as
/// the number of the elements in `x` along the `axis` dimension. It
/// contains the index in the unique output `y`.
/// In other words, for an `1-D` tensor `x` with `axis = None:
///
/// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
///
/// For example:
///
/// ```
/// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
/// y, idx = unique(x)
/// y ==> [1, 2, 4, 7, 8]
/// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
/// ```
///
/// For an `2-D` tensor `x` with `axis = 0`:
///
/// ```
/// # tensor 'x' is [[1, 0, 0],
/// #                [1, 0, 0],
/// #                [2, 0, 0]]
/// y, idx = unique(x, axis=0)
/// y ==> [[1, 0, 0],
///        [2, 0, 0]]
/// idx ==> [0, 0, 1]
/// ```
///
/// For an `2-D` tensor `x` with `axis = 1`:
///
/// ```
/// # tensor 'x' is [[1, 0, 0],
/// #                [1, 0, 0],
/// #                [2, 0, 0]]
/// y, idx = unique(x, axis=1)
/// y ==> [[1, 0],
///        [1, 0],
///        [2, 0]]
/// idx ==> [0, 1, 1]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * x: A `Tensor`.
/// * axis: A `Tensor` of type `int32` (default: None). The axis of the Tensor to
/// find the unique elements.
///
/// Returns:
/// * `Output` y: A `Tensor`. Unique elements along the `axis` of `Tensor` x.
/// * `Output` idx: A 1-D Tensor. Has the same type as x that contains the index of each
/// value of x in the output y.
class UniqueV2 {
 public:
  /// Optional attribute setters for UniqueV2
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutIdx(DataType x) {
      Attrs ret = *this;
      ret.out_idx_ = x;
      return ret;
    }

    DataType out_idx_ = DT_INT32;
  };
  UniqueV2(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
         ::tensorflow::Input axis);
  UniqueV2(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
         ::tensorflow::Input axis, const UniqueV2::Attrs& attrs);

  static Attrs OutIdx(DataType x) {
    return Attrs().OutIdx(x);
  }

  Operation operation;
  ::tensorflow::Output y;
  ::tensorflow::Output idx;
};

/// Finds unique elements in a 1-D tensor.
///
/// This operation returns a tensor `y` containing all of the unique elements of `x`
/// sorted in the same order that they occur in `x`. This operation also returns a
/// tensor `idx` the same size as `x` that contains the index of each value of `x`
/// in the unique output `y`. Finally, it returns a third tensor `count` that
/// contains the count of each element of `y` in `x`. In other words:
///
/// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
///
/// For example:
///
/// ```
/// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
/// y, idx, count = unique_with_counts(x)
/// y ==> [1, 2, 4, 7, 8]
/// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
/// count ==> [2, 1, 3, 1, 2]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * x: 1-D.
///
/// Returns:
/// * `Output` y: 1-D.
/// * `Output` idx: 1-D.
/// * `Output` count: 1-D.
class UniqueWithCounts {
 public:
  /// Optional attribute setters for UniqueWithCounts
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutIdx(DataType x) {
      Attrs ret = *this;
      ret.out_idx_ = x;
      return ret;
    }

    DataType out_idx_ = DT_INT32;
  };
  UniqueWithCounts(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  UniqueWithCounts(const ::tensorflow::Scope& scope, ::tensorflow::Input x, const
                 UniqueWithCounts::Attrs& attrs);

  static Attrs OutIdx(DataType x) {
    return Attrs().OutIdx(x);
  }

  Operation operation;
  ::tensorflow::Output y;
  ::tensorflow::Output idx;
  ::tensorflow::Output count;
};

/// Finds unique elements along an axis of a tensor.
///
/// This operation either returns a tensor `y` containing unique elements
/// along the `axis` of a tensor. The returned unique elements is sorted
/// in the same order as they occur along `axis` in `x`.
/// This operation also returns a tensor `idx` and a tensor `count`
/// that are the same size as the number of the elements in `x` along the
/// `axis` dimension. The `idx` contains the index in the unique output `y`
/// and the `count` contains the count in the unique output `y`.
/// In other words, for an `1-D` tensor `x` with `axis = None:
///
/// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
///
/// For example:
///
/// ```
/// x = tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 8])
/// y, idx, count = tf.raw_ops.UniqueWithCountsV2(x=x, axis = [0])
/// y ==> [1, 2, 4, 7, 8]
/// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
/// count ==> [2, 1, 3, 1, 2]
/// ```
///
/// For a `2-D` tensor `x` with `axis = 0`:
///
/// ```
/// x = tf.constant([[1, 0, 0],
///                 [1, 0, 0],
///                 [2, 0, 0]])
/// y, idx, count = tf.raw_ops.UniqueWithCountsV2(x=x, axis=[0])
/// y ==> [[1, 0, 0],
///        [2, 0, 0]]
/// idx ==> [0, 0, 1]
/// count ==> [2, 1]
/// ```
///
/// For a `2-D` tensor `x` with `axis = 1`:
///
/// ```
/// x = tf.constant([[1, 0, 0],
///                 [1, 0, 0],
///                 [2, 0, 0]])
/// y, idx, count = tf.raw_ops.UniqueWithCountsV2(x=x, axis=[1])
/// y ==> [[1, 0],
///        [1, 0],
///        [2, 0]]
/// idx ==> [0, 1, 1]
/// count ==> [1, 2]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * x: A `Tensor`.
/// * axis: A `Tensor` of type `int32` (default: None). The axis of the Tensor to
/// find the unique elements.
///
/// Returns:
/// * `Output` y: A `Tensor`. Unique elements along the `axis` of `Tensor` x.
/// * `Output` idx: A 1-D Tensor. Has the same type as x that contains the index of each
/// value of x in the output y.
/// * `Output` count: A 1-D Tensor. The count of each value of x in the output y.
class UniqueWithCountsV2 {
 public:
  /// Optional attribute setters for UniqueWithCountsV2
  struct Attrs {
    /// Defaults to DT_INT32
    TF_MUST_USE_RESULT Attrs OutIdx(DataType x) {
      Attrs ret = *this;
      ret.out_idx_ = x;
      return ret;
    }

    DataType out_idx_ = DT_INT32;
  };
  UniqueWithCountsV2(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
                   ::tensorflow::Input axis);
  UniqueWithCountsV2(const ::tensorflow::Scope& scope, ::tensorflow::Input x,
                   ::tensorflow::Input axis, const UniqueWithCountsV2::Attrs&
                   attrs);

  static Attrs OutIdx(DataType x) {
    return Attrs().OutIdx(x);
  }

  Operation operation;
  ::tensorflow::Output y;
  ::tensorflow::Output idx;
  ::tensorflow::Output count;
};

/// Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.
///
/// Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
/// For example, given a tensor of shape `(A, B, C, D)`;
///
/// If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
///   and each tensor in `output` will have shape `(B, C, D)`. (Note that the
///   dimension unpacked along is gone, unlike `split`).
///
/// If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
///   and each tensor in `output` will have shape `(A, C, D)`.
/// Etc.
///
/// This is the opposite of `pack`.
///
/// Args:
/// * scope: A Scope object
/// * value: 1-D or higher, with `axis` dimension size equal to `num`.
///
/// Optional attributes (see `Attrs`):
/// * axis: Dimension along which to unpack.  Negative values wrap around, so the
/// valid range is `[-R, R)`.
///
/// Returns:
/// * `OutputList`: The list of tensors unpacked from `value`.
class Unstack {
 public:
  /// Optional attribute setters for Unstack
  struct Attrs {
    /// Dimension along which to unpack.  Negative values wrap around, so the
    /// valid range is `[-R, R)`.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Axis(int64 x) {
      Attrs ret = *this;
      ret.axis_ = x;
      return ret;
    }

    int64 axis_ = 0;
  };
  Unstack(const ::tensorflow::Scope& scope, ::tensorflow::Input value, int64 num);
  Unstack(const ::tensorflow::Scope& scope, ::tensorflow::Input value, int64 num,
        const Unstack::Attrs& attrs);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  static Attrs Axis(int64 x) {
    return Attrs().Axis(x);
  }

  Operation operation;
  ::tensorflow::OutputList output;
};

/// Converts an array of flat indices into a tuple of coordinate arrays.
///
///
/// Example:
///
/// ```
/// y = tf.unravel_index(indices=[2, 5, 7], dims=[3, 3])
/// # 'dims' represent a hypothetical (3, 3) tensor of indices:
/// # [[0, 1, *2*],
/// #  [3, 4, *5*],
/// #  [6, *7*, 8]]
/// # For each entry from 'indices', this operation returns
/// # its coordinates (marked with '*'), such as
/// # 2 ==> (0, 2)
/// # 5 ==> (1, 2)
/// # 7 ==> (2, 1)
/// y ==> [[0, 1, 2], [2, 2, 1]]
/// ```
///
/// @compatibility(numpy)
/// Equivalent to np.unravel_index
/// @end_compatibility
///
/// Args:
/// * scope: A Scope object
/// * indices: An 0-D or 1-D `int` Tensor whose elements are indices into the
/// flattened version of an array of dimensions dims.
/// * dims: An 1-D `int` Tensor. The shape of the array to use for unraveling
/// indices.
///
/// Returns:
/// * `Output`: An 2-D (or 1-D if indices is 0-D) tensor where each row has the
/// same shape as the indices array.
class UnravelIndex {
 public:
  UnravelIndex(const ::tensorflow::Scope& scope, ::tensorflow::Input indices,
             ::tensorflow::Input dims);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// Returns locations of nonzero / true values in a tensor.
///
/// This operation returns the coordinates of true elements in `condition`. The
/// coordinates are returned in a 2-D tensor where the first dimension (rows)
/// represents the number of true elements, and the second dimension (columns)
/// represents the coordinates of the true elements. Keep in mind, the shape of
/// the output tensor can vary depending on how many true values there are in
/// `condition`. Indices are output in row-major order.
///
/// For example:
///
/// ```
/// # 'input' tensor is [[True, False]
/// #                    [True, False]]
/// # 'input' has two true values, so output has two coordinates.
/// # 'input' has rank of 2, so coordinates have two indices.
/// where(input) ==> [[0, 0],
///                   [1, 0]]
///
/// # `condition` tensor is [[[True, False]
/// #                     [True, False]]
/// #                    [[False, True]
/// #                     [False, True]]
/// #                    [[False, False]
/// #                     [False, True]]]
/// # 'input' has 5 true values, so output has 5 coordinates.
/// # 'input' has rank of 3, so coordinates have three indices.
/// where(input) ==> [[0, 0, 0],
///                   [0, 1, 0],
///                   [1, 0, 1],
///                   [1, 1, 1],
///                   [2, 1, 1]]
///
/// # `condition` tensor is [[[1.5,  0.0]
/// #                     [-0.5, 0.0]]
/// #                    [[0.0,  0.25]
/// #                     [0.0,  0.75]]
/// #                    [[0.0,  0.0]
/// #                     [0.0,  0.01]]]
/// # 'input' has 5 nonzero values, so output has 5 coordinates.
/// # 'input' has rank of 3, so coordinates have three indices.
/// where(input) ==> [[0, 0, 0],
///                   [0, 1, 0],
///                   [1, 0, 1],
///                   [1, 1, 1],
///                   [2, 1, 1]]
///
/// # `condition` tensor is [[[1.5 + 0.0j, 0.0  + 0.0j]
/// #                     [0.0 + 0.5j, 0.0  + 0.0j]]
/// #                    [[0.0 + 0.0j, 0.25 + 1.5j]
/// #                     [0.0 + 0.0j, 0.75 + 0.0j]]
/// #                    [[0.0 + 0.0j, 0.0  + 0.0j]
/// #                     [0.0 + 0.0j, 0.01 + 0.0j]]]
/// # 'input' has 5 nonzero magnitude values, so output has 5 coordinates.
/// # 'input' has rank of 3, so coordinates have three indices.
/// where(input) ==> [[0, 0, 0],
///                   [0, 1, 0],
///                   [1, 0, 1],
///                   [1, 1, 1],
///                   [2, 1, 1]]
/// ```
///
/// Args:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The index tensor.
class Where {
 public:
  Where(const ::tensorflow::Scope& scope, ::tensorflow::Input condition);
  operator ::tensorflow::Output() const { return index; }
  operator ::tensorflow::Input() const { return index; }
  ::tensorflow::Node* node() const { return index.node(); }

  Operation operation;
  ::tensorflow::Output index;
};

/// Returns a tensor of zeros with the same shape and type as x.
///
/// Args:
/// * scope: A Scope object
/// * x: a tensor of type T.
///
/// Returns:
/// * `Output`: a tensor of the same shape and type as x but filled with zeros.
class ZerosLike {
 public:
  ZerosLike(const ::tensorflow::Scope& scope, ::tensorflow::Input x);
  operator ::tensorflow::Output() const { return y; }
  operator ::tensorflow::Input() const { return y; }
  ::tensorflow::Node* node() const { return y.node(); }

  Operation operation;
  ::tensorflow::Output y;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_ARRAY_OPS_H_
