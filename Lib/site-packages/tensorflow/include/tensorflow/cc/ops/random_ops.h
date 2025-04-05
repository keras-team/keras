// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_RANDOM_OPS_H_
#define TENSORFLOW_CC_OPS_RANDOM_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup random_ops Random Ops
/// @{

/// Draws samples from a multinomial distribution.
///
/// Args:
/// * scope: A Scope object
/// * logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
/// represents the unnormalized log probabilities for all classes.
/// * num_samples: 0-D.  Number of independent samples to draw for each row slice.
///
/// Optional attributes (see `Attrs`):
/// * seed: If either seed or seed2 is set to be non-zero, the internal random number
/// generator is seeded by the given seed.  Otherwise, a random seed is used.
/// * seed2: A second seed to avoid seed collision.
///
/// Returns:
/// * `Output`: 2-D Tensor with shape `[batch_size, num_samples]`.  Each slice `[i, :]`
/// contains the drawn class labels with range `[0, num_classes)`.
class Multinomial {
 public:
  /// Optional attribute setters for Multinomial
  struct Attrs {
    /// If either seed or seed2 is set to be non-zero, the internal random number
    /// generator is seeded by the given seed.  Otherwise, a random seed is used.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed(int64 x) {
      Attrs ret = *this;
      ret.seed_ = x;
      return ret;
    }

    /// A second seed to avoid seed collision.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed2(int64 x) {
      Attrs ret = *this;
      ret.seed2_ = x;
      return ret;
    }

    /// Defaults to DT_INT64
    TF_MUST_USE_RESULT Attrs OutputDtype(DataType x) {
      Attrs ret = *this;
      ret.output_dtype_ = x;
      return ret;
    }

    int64 seed_ = 0;
    int64 seed2_ = 0;
    DataType output_dtype_ = DT_INT64;
  };
  Multinomial(const ::tensorflow::Scope& scope, ::tensorflow::Input logits,
            ::tensorflow::Input num_samples);
  Multinomial(const ::tensorflow::Scope& scope, ::tensorflow::Input logits,
            ::tensorflow::Input num_samples, const Multinomial::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Seed(int64 x) {
    return Attrs().Seed(x);
  }
  static Attrs Seed2(int64 x) {
    return Attrs().Seed2(x);
  }
  static Attrs OutputDtype(DataType x) {
    return Attrs().OutputDtype(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Outputs random values from a normal distribution. The parameters may each be a
///
/// scalar which applies to the entire output, or a vector of length shape[0] which
/// stores the parameters for each batch.
///
/// Args:
/// * scope: A Scope object
/// * shape: The shape of the output tensor. Batches are indexed by the 0th dimension.
/// * means: The mean parameter of each batch.
/// * stdevs: The standard deviation parameter of each batch. Must be greater than 0.
/// * minvals: The minimum cutoff. May be -infinity.
/// * maxvals: The maximum cutoff. May be +infinity, and must be more than the minval
/// for each batch.
///
/// Optional attributes (see `Attrs`):
/// * seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// * seed2: A second seed to avoid seed collision.
///
/// Returns:
/// * `Output`: A matrix of shape num_batches x samples_per_batch, filled with random
/// truncated normal values using the parameters for each row.
class ParameterizedTruncatedNormal {
 public:
  /// Optional attribute setters for ParameterizedTruncatedNormal
  struct Attrs {
    /// If either `seed` or `seed2` are set to be non-zero, the random number
    /// generator is seeded by the given seed.  Otherwise, it is seeded by a
    /// random seed.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed(int64 x) {
      Attrs ret = *this;
      ret.seed_ = x;
      return ret;
    }

    /// A second seed to avoid seed collision.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed2(int64 x) {
      Attrs ret = *this;
      ret.seed2_ = x;
      return ret;
    }

    int64 seed_ = 0;
    int64 seed2_ = 0;
  };
  ParameterizedTruncatedNormal(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input shape, ::tensorflow::Input
                             means, ::tensorflow::Input stdevs,
                             ::tensorflow::Input minvals, ::tensorflow::Input
                             maxvals);
  ParameterizedTruncatedNormal(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input shape, ::tensorflow::Input
                             means, ::tensorflow::Input stdevs,
                             ::tensorflow::Input minvals, ::tensorflow::Input
                             maxvals, const
                             ParameterizedTruncatedNormal::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Seed(int64 x) {
    return Attrs().Seed(x);
  }
  static Attrs Seed2(int64 x) {
    return Attrs().Seed2(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Outputs random values from the Gamma distribution(s) described by alpha.
///
/// This op uses the algorithm by Marsaglia et al. to acquire samples via
/// transformation-rejection from pairs of uniform and normal random variables.
/// See http://dl.acm.org/citation.cfm?id=358414
///
/// Args:
/// * scope: A Scope object
/// * shape: 1-D integer tensor. Shape of independent samples to draw from each
/// distribution described by the shape parameters given in alpha.
/// * alpha: A tensor in which each scalar is a "shape" parameter describing the
/// associated gamma distribution.
///
/// Optional attributes (see `Attrs`):
/// * seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// * seed2: A second seed to avoid seed collision.
///
/// Returns:
/// * `Output`: A tensor with shape `shape + shape(alpha)`. Each slice
/// `[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
/// `alpha[i0, i1, ...iN]`. The dtype of the output matches the dtype of alpha.
class RandomGamma {
 public:
  /// Optional attribute setters for RandomGamma
  struct Attrs {
    /// If either `seed` or `seed2` are set to be non-zero, the random number
    /// generator is seeded by the given seed.  Otherwise, it is seeded by a
    /// random seed.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed(int64 x) {
      Attrs ret = *this;
      ret.seed_ = x;
      return ret;
    }

    /// A second seed to avoid seed collision.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed2(int64 x) {
      Attrs ret = *this;
      ret.seed2_ = x;
      return ret;
    }

    int64 seed_ = 0;
    int64 seed2_ = 0;
  };
  RandomGamma(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
            ::tensorflow::Input alpha);
  RandomGamma(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
            ::tensorflow::Input alpha, const RandomGamma::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Seed(int64 x) {
    return Attrs().Seed(x);
  }
  static Attrs Seed2(int64 x) {
    return Attrs().Seed2(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Outputs random values from the Poisson distribution(s) described by rate.
///
/// This op uses two algorithms, depending on rate. If rate >= 10, then
/// the algorithm by Hormann is used to acquire samples via
/// transformation-rejection.
/// See http://www.sciencedirect.com/science/article/pii/0167668793909974.
///
/// Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
/// random variables.
/// See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
/// Programming, Volume 2. Addison Wesley
///
/// Args:
/// * scope: A Scope object
/// * shape: 1-D integer tensor. Shape of independent samples to draw from each
/// distribution described by the shape parameters given in rate.
/// * rate: A tensor in which each scalar is a "rate" parameter describing the
/// associated poisson distribution.
///
/// Optional attributes (see `Attrs`):
/// * seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// * seed2: A second seed to avoid seed collision.
///
/// Returns:
/// * `Output`: A tensor with shape `shape + shape(rate)`. Each slice
/// `[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
/// `rate[i0, i1, ...iN]`.
class RandomPoissonV2 {
 public:
  /// Optional attribute setters for RandomPoissonV2
  struct Attrs {
    /// If either `seed` or `seed2` are set to be non-zero, the random number
    /// generator is seeded by the given seed.  Otherwise, it is seeded by a
    /// random seed.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed(int64 x) {
      Attrs ret = *this;
      ret.seed_ = x;
      return ret;
    }

    /// A second seed to avoid seed collision.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed2(int64 x) {
      Attrs ret = *this;
      ret.seed2_ = x;
      return ret;
    }

    /// Defaults to DT_INT64
    TF_MUST_USE_RESULT Attrs Dtype(DataType x) {
      Attrs ret = *this;
      ret.dtype_ = x;
      return ret;
    }

    int64 seed_ = 0;
    int64 seed2_ = 0;
    DataType dtype_ = DT_INT64;
  };
  RandomPoissonV2(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
                ::tensorflow::Input rate);
  RandomPoissonV2(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
                ::tensorflow::Input rate, const RandomPoissonV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Seed(int64 x) {
    return Attrs().Seed(x);
  }
  static Attrs Seed2(int64 x) {
    return Attrs().Seed2(x);
  }
  static Attrs Dtype(DataType x) {
    return Attrs().Dtype(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Randomly shuffles a tensor along its first dimension.
///
///   The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
///   to one and only one `output[i]`. For example, a mapping that might occur for a
///   3x2 tensor is:
///
/// ```
/// [[1, 2],       [[5, 6],
///  [3, 4],  ==>   [1, 2],
///  [5, 6]]        [3, 4]]
/// ```
///
/// Args:
/// * scope: A Scope object
/// * value: The tensor to be shuffled.
///
/// Optional attributes (see `Attrs`):
/// * seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// * seed2: A second seed to avoid seed collision.
///
/// Returns:
/// * `Output`: A tensor of same shape and type as `value`, shuffled along its first
/// dimension.
class RandomShuffle {
 public:
  /// Optional attribute setters for RandomShuffle
  struct Attrs {
    /// If either `seed` or `seed2` are set to be non-zero, the random number
    /// generator is seeded by the given seed.  Otherwise, it is seeded by a
    /// random seed.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed(int64 x) {
      Attrs ret = *this;
      ret.seed_ = x;
      return ret;
    }

    /// A second seed to avoid seed collision.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed2(int64 x) {
      Attrs ret = *this;
      ret.seed2_ = x;
      return ret;
    }

    int64 seed_ = 0;
    int64 seed2_ = 0;
  };
  RandomShuffle(const ::tensorflow::Scope& scope, ::tensorflow::Input value);
  RandomShuffle(const ::tensorflow::Scope& scope, ::tensorflow::Input value,
              const RandomShuffle::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Seed(int64 x) {
    return Attrs().Seed(x);
  }
  static Attrs Seed2(int64 x) {
    return Attrs().Seed2(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Outputs random values from a normal distribution.
///
/// The generated values will have mean 0 and standard deviation 1.
///
/// Args:
/// * scope: A Scope object
/// * shape: The shape of the output tensor.
/// * dtype: The type of the output.
///
/// Optional attributes (see `Attrs`):
/// * seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// * seed2: A second seed to avoid seed collision.
///
/// Returns:
/// * `Output`: A tensor of the specified shape filled with random normal values.
class RandomNormal {
 public:
  /// Optional attribute setters for RandomNormal
  struct Attrs {
    /// If either `seed` or `seed2` are set to be non-zero, the random number
    /// generator is seeded by the given seed.  Otherwise, it is seeded by a
    /// random seed.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed(int64 x) {
      Attrs ret = *this;
      ret.seed_ = x;
      return ret;
    }

    /// A second seed to avoid seed collision.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed2(int64 x) {
      Attrs ret = *this;
      ret.seed2_ = x;
      return ret;
    }

    int64 seed_ = 0;
    int64 seed2_ = 0;
  };
  RandomNormal(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
             DataType dtype);
  RandomNormal(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
             DataType dtype, const RandomNormal::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Seed(int64 x) {
    return Attrs().Seed(x);
  }
  static Attrs Seed2(int64 x) {
    return Attrs().Seed2(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Outputs random values from a uniform distribution.
///
/// The generated values follow a uniform distribution in the range `[0, 1)`. The
/// lower bound 0 is included in the range, while the upper bound 1 is excluded.
///
/// Args:
/// * scope: A Scope object
/// * shape: The shape of the output tensor.
/// * dtype: The type of the output.
///
/// Optional attributes (see `Attrs`):
/// * seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// * seed2: A second seed to avoid seed collision.
///
/// Returns:
/// * `Output`: A tensor of the specified shape filled with uniform random values.
class RandomUniform {
 public:
  /// Optional attribute setters for RandomUniform
  struct Attrs {
    /// If either `seed` or `seed2` are set to be non-zero, the random number
    /// generator is seeded by the given seed.  Otherwise, it is seeded by a
    /// random seed.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed(int64 x) {
      Attrs ret = *this;
      ret.seed_ = x;
      return ret;
    }

    /// A second seed to avoid seed collision.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed2(int64 x) {
      Attrs ret = *this;
      ret.seed2_ = x;
      return ret;
    }

    int64 seed_ = 0;
    int64 seed2_ = 0;
  };
  RandomUniform(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
              DataType dtype);
  RandomUniform(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
              DataType dtype, const RandomUniform::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Seed(int64 x) {
    return Attrs().Seed(x);
  }
  static Attrs Seed2(int64 x) {
    return Attrs().Seed2(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Outputs random integers from a uniform distribution.
///
/// The generated values are uniform integers in the range `[minval, maxval)`.
/// The lower bound `minval` is included in the range, while the upper bound
/// `maxval` is excluded.
///
/// The random integers are slightly biased unless `maxval - minval` is an exact
/// power of two.  The bias is small for values of `maxval - minval` significantly
/// smaller than the range of the output (either `2^32` or `2^64`).
///
/// Args:
/// * scope: A Scope object
/// * shape: The shape of the output tensor.
/// * minval: 0-D.  Inclusive lower bound on the generated integers.
/// * maxval: 0-D.  Exclusive upper bound on the generated integers.
///
/// Optional attributes (see `Attrs`):
/// * seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// * seed2: A second seed to avoid seed collision.
///
/// Returns:
/// * `Output`: A tensor of the specified shape filled with uniform random integers.
class RandomUniformInt {
 public:
  /// Optional attribute setters for RandomUniformInt
  struct Attrs {
    /// If either `seed` or `seed2` are set to be non-zero, the random number
    /// generator is seeded by the given seed.  Otherwise, it is seeded by a
    /// random seed.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed(int64 x) {
      Attrs ret = *this;
      ret.seed_ = x;
      return ret;
    }

    /// A second seed to avoid seed collision.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed2(int64 x) {
      Attrs ret = *this;
      ret.seed2_ = x;
      return ret;
    }

    int64 seed_ = 0;
    int64 seed2_ = 0;
  };
  RandomUniformInt(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
                 ::tensorflow::Input minval, ::tensorflow::Input maxval);
  RandomUniformInt(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
                 ::tensorflow::Input minval, ::tensorflow::Input maxval, const
                 RandomUniformInt::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Seed(int64 x) {
    return Attrs().Seed(x);
  }
  static Attrs Seed2(int64 x) {
    return Attrs().Seed2(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// Outputs random values from a truncated normal distribution.
///
/// The generated values follow a normal distribution with mean 0 and standard
/// deviation 1, except that values whose magnitude is more than 2 standard
/// deviations from the mean are dropped and re-picked.
///
/// Args:
/// * scope: A Scope object
/// * shape: The shape of the output tensor.
/// * dtype: The type of the output.
///
/// Optional attributes (see `Attrs`):
/// * seed: If either `seed` or `seed2` are set to be non-zero, the random number
/// generator is seeded by the given seed.  Otherwise, it is seeded by a
/// random seed.
/// * seed2: A second seed to avoid seed collision.
///
/// Returns:
/// * `Output`: A tensor of the specified shape filled with random truncated normal
/// values.
class TruncatedNormal {
 public:
  /// Optional attribute setters for TruncatedNormal
  struct Attrs {
    /// If either `seed` or `seed2` are set to be non-zero, the random number
    /// generator is seeded by the given seed.  Otherwise, it is seeded by a
    /// random seed.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed(int64 x) {
      Attrs ret = *this;
      ret.seed_ = x;
      return ret;
    }

    /// A second seed to avoid seed collision.
    ///
    /// Defaults to 0
    TF_MUST_USE_RESULT Attrs Seed2(int64 x) {
      Attrs ret = *this;
      ret.seed2_ = x;
      return ret;
    }

    int64 seed_ = 0;
    int64 seed2_ = 0;
  };
  TruncatedNormal(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
                DataType dtype);
  TruncatedNormal(const ::tensorflow::Scope& scope, ::tensorflow::Input shape,
                DataType dtype, const TruncatedNormal::Attrs& attrs);
  operator ::tensorflow::Output() const { return output; }
  operator ::tensorflow::Input() const { return output; }
  ::tensorflow::Node* node() const { return output.node(); }

  static Attrs Seed(int64 x) {
    return Attrs().Seed(x);
  }
  static Attrs Seed2(int64 x) {
    return Attrs().Seed2(x);
  }

  Operation operation;
  ::tensorflow::Output output;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_RANDOM_OPS_H_
