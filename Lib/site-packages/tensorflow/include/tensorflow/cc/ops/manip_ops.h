// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_MANIP_OPS_H_
#define TENSORFLOW_CC_OPS_MANIP_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup manip_ops Manip Ops
/// @{

/// Rolls the elements of a tensor along an axis.
///
/// The elements are shifted positively (towards larger indices) by the offset of
/// `shift` along the dimension of `axis`. Negative `shift` values will shift
/// elements in the opposite direction. Elements that roll passed the last position
/// will wrap around to the first and vice versa. Multiple shifts along multiple
/// axes may be specified.
///
/// For example:
///
/// ```
/// # 't' is [0, 1, 2, 3, 4]
/// roll(t, shift=2, axis=0) ==> [3, 4, 0, 1, 2]
///
/// # shifting along multiple dimensions
/// # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
/// roll(t, shift=[1, -2], axis=[0, 1]) ==> [[7, 8, 9, 5, 6], [2, 3, 4, 0, 1]]
///
/// # shifting along the same axis multiple times
/// # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
/// roll(t, shift=[2, -3], axis=[1, 1]) ==> [[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * shift: Dimension must be 0-D or 1-D. `shift[i]` specifies the number of places by which
/// elements are shifted positively (towards larger indices) along the dimension
/// specified by `axis[i]`. Negative shifts will roll the elements in the opposite
/// direction.
/// * axis: Dimension must be 0-D or 1-D. `axis[i]` specifies the dimension that the shift
/// `shift[i]` should occur. If the same axis is referenced more than once, the
/// total shift for that axis will be the sum of all the shifts that belong to that
/// axis.
///
/// Returns:
/// * `Output`: Has the same shape and size as the input. The elements are shifted
/// positively (towards larger indices) by the offsets of `shift` along the
/// dimensions of `axis`.
class Roll {
 public:
  Roll(const ::tensorflow::Scope& scope, ::tensorflow::Input input,
     ::tensorflow::Input shift, ::tensorflow::Input axis);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  Operation operation;
  ::tensorflow::Output output;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_MANIP_OPS_H_
