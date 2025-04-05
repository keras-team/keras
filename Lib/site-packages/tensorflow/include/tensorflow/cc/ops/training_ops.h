// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_TRAINING_OPS_H_
#define TENSORFLOW_CC_OPS_TRAINING_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup training_ops Training Ops
/// @{

/// Update '*var' according to the adadelta scheme.
///
/// accum = rho() * accum + (1 - rho()) * grad.square();
/// update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
/// update_accum = rho() * update_accum + (1 - rho()) * update.square();
/// var -= update;
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * accum_update: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * rho: Decay factor. Must be a scalar.
/// * epsilon: Constant factor. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var, accum and update_accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyAdadelta {
 public:
  /// Optional attribute setters for ApplyAdadelta
  struct Attrs {
    /// If True, updating of the var, accum and update_accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ApplyAdadelta(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
              ::tensorflow::Input accum, ::tensorflow::Input accum_update,
              ::tensorflow::Input lr, ::tensorflow::Input rho,
              ::tensorflow::Input epsilon, ::tensorflow::Input grad);
  ApplyAdadelta(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
              ::tensorflow::Input accum, ::tensorflow::Input accum_update,
              ::tensorflow::Input lr, ::tensorflow::Input rho,
              ::tensorflow::Input epsilon, ::tensorflow::Input grad, const
              ApplyAdadelta::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the adagrad scheme.
///
/// accum += grad * grad
/// var -= lr * grad * (1 / sqrt(accum))
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyAdagrad {
 public:
  /// Optional attribute setters for ApplyAdagrad
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UpdateSlots(bool x) {
      Attrs ret = *this;
      ret.update_slots_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool update_slots_ = true;
  };
  ApplyAdagrad(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
             ::tensorflow::Input accum, ::tensorflow::Input lr,
             ::tensorflow::Input grad);
  ApplyAdagrad(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
             ::tensorflow::Input accum, ::tensorflow::Input lr,
             ::tensorflow::Input grad, const ApplyAdagrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UpdateSlots(bool x) {
    return Attrs().UpdateSlots(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the proximal adagrad scheme.
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * gradient_accumulator: Should be from a Variable().
/// * gradient_squared_accumulator: Should be from a Variable().
/// * grad: The gradient.
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * global_step: Training step number. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyAdagradDA {
 public:
  /// Optional attribute setters for ApplyAdagradDA
  struct Attrs {
    /// If True, updating of the var and accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ApplyAdagradDA(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
               ::tensorflow::Input gradient_accumulator, ::tensorflow::Input
               gradient_squared_accumulator, ::tensorflow::Input grad,
               ::tensorflow::Input lr, ::tensorflow::Input l1,
               ::tensorflow::Input l2, ::tensorflow::Input global_step);
  ApplyAdagradDA(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
               ::tensorflow::Input gradient_accumulator, ::tensorflow::Input
               gradient_squared_accumulator, ::tensorflow::Input grad,
               ::tensorflow::Input lr, ::tensorflow::Input l1,
               ::tensorflow::Input l2, ::tensorflow::Input global_step, const
               ApplyAdagradDA::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the Adam algorithm.
///
/// $$\text{lr}_t := \mathrm{lr} \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$$
/// $$m_t := \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g$$
/// $$v_t := \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g^2$$
/// $$\text{var} := \begin{cases} \text{var} - (m_t \beta_1 + g \cdot (1 - \beta_1))\cdot\text{lr}_t/(\sqrt{v_t} + \epsilon), &\text{if use_nesterov}\\\\  \text{var} - m_t \cdot \text{lr}_t /(\sqrt{v_t} + \epsilon), &\text{otherwise} \end{cases}$$
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * m: Should be from a Variable().
/// * v: Should be from a Variable().
/// * beta1_power: Must be a scalar.
/// * beta2_power: Must be a scalar.
/// * lr: Scaling factor. Must be a scalar.
/// * beta1: Momentum factor. Must be a scalar.
/// * beta2: Momentum factor. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, m, and v tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// * use_nesterov: If `True`, uses the nesterov update.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyAdam {
 public:
  /// Optional attribute setters for ApplyAdam
  struct Attrs {
    /// If `True`, updating of the var, m, and v tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// If `True`, uses the nesterov update.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNesterov(bool x) {
      Attrs ret = *this;
      ret.use_nesterov_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool use_nesterov_ = false;
  };
  ApplyAdam(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
          ::tensorflow::Input m, ::tensorflow::Input v, ::tensorflow::Input
          beta1_power, ::tensorflow::Input beta2_power, ::tensorflow::Input lr,
          ::tensorflow::Input beta1, ::tensorflow::Input beta2,
          ::tensorflow::Input epsilon, ::tensorflow::Input grad);
  ApplyAdam(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
          ::tensorflow::Input m, ::tensorflow::Input v, ::tensorflow::Input
          beta1_power, ::tensorflow::Input beta2_power, ::tensorflow::Input lr,
          ::tensorflow::Input beta1, ::tensorflow::Input beta2,
          ::tensorflow::Input epsilon, ::tensorflow::Input grad, const
          ApplyAdam::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UseNesterov(bool x) {
    return Attrs().UseNesterov(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the AddSign update.
///
/// m_t <- beta1 * m_{t-1} + (1 - beta1) * g
/// update <- (alpha + sign_decay * sign(g) *sign(m)) * g
/// variable <- variable - lr_t * update
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * m: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * alpha: Must be a scalar.
/// * sign_decay: Must be a scalar.
/// * beta: Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and m tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyAddSign {
 public:
  /// Optional attribute setters for ApplyAddSign
  struct Attrs {
    /// If `True`, updating of the var and m tensors is
    /// protected by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ApplyAddSign(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
             ::tensorflow::Input m, ::tensorflow::Input lr, ::tensorflow::Input
             alpha, ::tensorflow::Input sign_decay, ::tensorflow::Input beta,
             ::tensorflow::Input grad);
  ApplyAddSign(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
             ::tensorflow::Input m, ::tensorflow::Input lr, ::tensorflow::Input
             alpha, ::tensorflow::Input sign_decay, ::tensorflow::Input beta,
             ::tensorflow::Input grad, const ApplyAddSign::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the centered RMSProp algorithm.
///
/// The centered RMSProp algorithm uses an estimate of the centered second moment
/// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
/// uses the (uncentered) second moment. This often helps with training, but is
/// slightly more expensive in terms of computation and memory.
///
/// Note that in dense implementation of this algorithm, mg, ms, and mom will
/// update even if the grad is zero, but in this sparse implementation, mg, ms,
/// and mom will not update in iterations during which the grad is zero.
///
/// mean_square = decay * mean_square + (1-decay) * gradient ** 2
/// mean_grad = decay * mean_grad + (1-decay) * gradient
///
/// Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)
///
/// mg <- rho * mg_{t-1} + (1-rho) * grad
/// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
/// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
/// var <- var - mom
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * mg: Should be from a Variable().
/// * ms: Should be from a Variable().
/// * mom: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * rho: Decay rate. Must be a scalar.
/// * momentum: Momentum Scale. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, mg, ms, and mom tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyCenteredRMSProp {
 public:
  /// Optional attribute setters for ApplyCenteredRMSProp
  struct Attrs {
    /// If `True`, updating of the var, mg, ms, and mom tensors is
    /// protected by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ApplyCenteredRMSProp(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input mg, ::tensorflow::Input ms,
                     ::tensorflow::Input mom, ::tensorflow::Input lr,
                     ::tensorflow::Input rho, ::tensorflow::Input momentum,
                     ::tensorflow::Input epsilon, ::tensorflow::Input grad);
  ApplyCenteredRMSProp(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input mg, ::tensorflow::Input ms,
                     ::tensorflow::Input mom, ::tensorflow::Input lr,
                     ::tensorflow::Input rho, ::tensorflow::Input momentum,
                     ::tensorflow::Input epsilon, ::tensorflow::Input grad,
                     const ApplyCenteredRMSProp::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the Ftrl-proximal scheme.
///
/// accum_new = accum + grad * grad
/// linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
/// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
/// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * linear: Should be from a Variable().
/// * grad: The gradient.
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * lr_power: Scaling factor. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyFtrl {
 public:
  /// Optional attribute setters for ApplyFtrl
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs MultiplyLinearByLr(bool x) {
      Attrs ret = *this;
      ret.multiply_linear_by_lr_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool multiply_linear_by_lr_ = false;
  };
  ApplyFtrl(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
          ::tensorflow::Input accum, ::tensorflow::Input linear,
          ::tensorflow::Input grad, ::tensorflow::Input lr, ::tensorflow::Input
          l1, ::tensorflow::Input l2, ::tensorflow::Input lr_power);
  ApplyFtrl(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
          ::tensorflow::Input accum, ::tensorflow::Input linear,
          ::tensorflow::Input grad, ::tensorflow::Input lr, ::tensorflow::Input
          l1, ::tensorflow::Input l2, ::tensorflow::Input lr_power, const
          ApplyFtrl::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs MultiplyLinearByLr(bool x) {
    return Attrs().MultiplyLinearByLr(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the Ftrl-proximal scheme.
///
/// grad_with_shrinkage = grad + 2 * l2_shrinkage * var
/// accum_new = accum + grad * grad
/// linear += grad_with_shrinkage -
///     (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
/// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
/// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * linear: Should be from a Variable().
/// * grad: The gradient.
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 shrinkage regularization. Must be a scalar.
/// * lr_power: Scaling factor. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyFtrlV2 {
 public:
  /// Optional attribute setters for ApplyFtrlV2
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs MultiplyLinearByLr(bool x) {
      Attrs ret = *this;
      ret.multiply_linear_by_lr_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool multiply_linear_by_lr_ = false;
  };
  ApplyFtrlV2(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
            ::tensorflow::Input accum, ::tensorflow::Input linear,
            ::tensorflow::Input grad, ::tensorflow::Input lr,
            ::tensorflow::Input l1, ::tensorflow::Input l2, ::tensorflow::Input
            l2_shrinkage, ::tensorflow::Input lr_power);
  ApplyFtrlV2(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
            ::tensorflow::Input accum, ::tensorflow::Input linear,
            ::tensorflow::Input grad, ::tensorflow::Input lr,
            ::tensorflow::Input l1, ::tensorflow::Input l2, ::tensorflow::Input
            l2_shrinkage, ::tensorflow::Input lr_power, const
            ApplyFtrlV2::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs MultiplyLinearByLr(bool x) {
    return Attrs().MultiplyLinearByLr(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' by subtracting 'alpha' * 'delta' from it.
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * alpha: Scaling factor. Must be a scalar.
/// * delta: The change.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyGradientDescent {
 public:
  /// Optional attribute setters for ApplyGradientDescent
  struct Attrs {
    /// If `True`, the subtraction will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ApplyGradientDescent(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input alpha, ::tensorflow::Input delta);
  ApplyGradientDescent(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input alpha, ::tensorflow::Input delta,
                     const ApplyGradientDescent::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the momentum scheme.
///
/// Set use_nesterov = True if you want to use Nesterov momentum.
///
/// accum = accum * momentum + grad
/// var -= lr * accum
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * grad: The gradient.
/// * momentum: Momentum. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// * use_nesterov: If `True`, the tensor passed to compute grad will be
/// var - lr * momentum * accum, so in the end, the var you get is actually
/// var - lr * momentum * accum.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyMomentum {
 public:
  /// Optional attribute setters for ApplyMomentum
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// If `True`, the tensor passed to compute grad will be
    /// var - lr * momentum * accum, so in the end, the var you get is actually
    /// var - lr * momentum * accum.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNesterov(bool x) {
      Attrs ret = *this;
      ret.use_nesterov_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool use_nesterov_ = false;
  };
  ApplyMomentum(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
              ::tensorflow::Input accum, ::tensorflow::Input lr,
              ::tensorflow::Input grad, ::tensorflow::Input momentum);
  ApplyMomentum(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
              ::tensorflow::Input accum, ::tensorflow::Input lr,
              ::tensorflow::Input grad, ::tensorflow::Input momentum, const
              ApplyMomentum::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UseNesterov(bool x) {
    return Attrs().UseNesterov(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the AddSign update.
///
/// m_t <- beta1 * m_{t-1} + (1 - beta1) * g
/// update <- exp(logbase * sign_decay * sign(g) * sign(m_t)) * g
/// variable <- variable - lr_t * update
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * m: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * logbase: Must be a scalar.
/// * sign_decay: Must be a scalar.
/// * beta: Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and m tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyPowerSign {
 public:
  /// Optional attribute setters for ApplyPowerSign
  struct Attrs {
    /// If `True`, updating of the var and m tensors is
    /// protected by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ApplyPowerSign(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
               ::tensorflow::Input m, ::tensorflow::Input lr,
               ::tensorflow::Input logbase, ::tensorflow::Input sign_decay,
               ::tensorflow::Input beta, ::tensorflow::Input grad);
  ApplyPowerSign(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
               ::tensorflow::Input m, ::tensorflow::Input lr,
               ::tensorflow::Input logbase, ::tensorflow::Input sign_decay,
               ::tensorflow::Input beta, ::tensorflow::Input grad, const
               ApplyPowerSign::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.
///
/// accum += grad * grad
/// prox_v = var - lr * grad * (1 / sqrt(accum))
/// var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyProximalAdagrad {
 public:
  /// Optional attribute setters for ApplyProximalAdagrad
  struct Attrs {
    /// If True, updating of the var and accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ApplyProximalAdagrad(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input accum, ::tensorflow::Input lr,
                     ::tensorflow::Input l1, ::tensorflow::Input l2,
                     ::tensorflow::Input grad);
  ApplyProximalAdagrad(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input accum, ::tensorflow::Input lr,
                     ::tensorflow::Input l1, ::tensorflow::Input l2,
                     ::tensorflow::Input grad, const
                     ApplyProximalAdagrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' as FOBOS algorithm with fixed learning rate.
///
/// prox_v = var - alpha * delta
/// var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * alpha: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * delta: The change.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyProximalGradientDescent {
 public:
  /// Optional attribute setters for ApplyProximalGradientDescent
  struct Attrs {
    /// If True, the subtraction will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ApplyProximalGradientDescent(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input
                             alpha, ::tensorflow::Input l1, ::tensorflow::Input
                             l2, ::tensorflow::Input delta);
  ApplyProximalGradientDescent(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input
                             alpha, ::tensorflow::Input l1, ::tensorflow::Input
                             l2, ::tensorflow::Input delta, const
                             ApplyProximalGradientDescent::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the RMSProp algorithm.
///
/// Note that in dense implementation of this algorithm, ms and mom will
/// update even if the grad is zero, but in this sparse implementation, ms
/// and mom will not update in iterations during which the grad is zero.
///
/// mean_square = decay * mean_square + (1-decay) * gradient ** 2
/// Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
///
/// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
/// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
/// var <- var - mom
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * ms: Should be from a Variable().
/// * mom: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * rho: Decay rate. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, ms, and mom tensors is protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyRMSProp {
 public:
  /// Optional attribute setters for ApplyRMSProp
  struct Attrs {
    /// If `True`, updating of the var, ms, and mom tensors is protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ApplyRMSProp(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
             ::tensorflow::Input ms, ::tensorflow::Input mom,
             ::tensorflow::Input lr, ::tensorflow::Input rho,
             ::tensorflow::Input momentum, ::tensorflow::Input epsilon,
             ::tensorflow::Input grad);
  ApplyRMSProp(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
             ::tensorflow::Input ms, ::tensorflow::Input mom,
             ::tensorflow::Input lr, ::tensorflow::Input rho,
             ::tensorflow::Input momentum, ::tensorflow::Input epsilon,
             ::tensorflow::Input grad, const ApplyRMSProp::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the adadelta scheme.
///
/// accum = rho() * accum + (1 - rho()) * grad.square();
/// update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
/// update_accum = rho() * update_accum + (1 - rho()) * update.square();
/// var -= update;
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * accum_update: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * rho: Decay factor. Must be a scalar.
/// * epsilon: Constant factor. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var, accum and update_accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyAdadelta {
 public:
  /// Optional attribute setters for ResourceApplyAdadelta
  struct Attrs {
    /// If True, updating of the var, accum and update_accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceApplyAdadelta(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      var, ::tensorflow::Input accum, ::tensorflow::Input
                      accum_update, ::tensorflow::Input lr, ::tensorflow::Input
                      rho, ::tensorflow::Input epsilon, ::tensorflow::Input
                      grad);
  ResourceApplyAdadelta(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      var, ::tensorflow::Input accum, ::tensorflow::Input
                      accum_update, ::tensorflow::Input lr, ::tensorflow::Input
                      rho, ::tensorflow::Input epsilon, ::tensorflow::Input
                      grad, const ResourceApplyAdadelta::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update '*var' according to the adagrad scheme.
///
/// accum += grad * grad
/// var -= lr * grad * (1 / sqrt(accum))
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyAdagrad {
 public:
  /// Optional attribute setters for ResourceApplyAdagrad
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UpdateSlots(bool x) {
      Attrs ret = *this;
      ret.update_slots_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool update_slots_ = true;
  };
  ResourceApplyAdagrad(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input accum, ::tensorflow::Input lr,
                     ::tensorflow::Input grad);
  ResourceApplyAdagrad(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input accum, ::tensorflow::Input lr,
                     ::tensorflow::Input grad, const
                     ResourceApplyAdagrad::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UpdateSlots(bool x) {
    return Attrs().UpdateSlots(x);
  }

  Operation operation;
};

/// Update '*var' according to the proximal adagrad scheme.
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * gradient_accumulator: Should be from a Variable().
/// * gradient_squared_accumulator: Should be from a Variable().
/// * grad: The gradient.
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * global_step: Training step number. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyAdagradDA {
 public:
  /// Optional attribute setters for ResourceApplyAdagradDA
  struct Attrs {
    /// If True, updating of the var and accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceApplyAdagradDA(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       var, ::tensorflow::Input gradient_accumulator,
                       ::tensorflow::Input gradient_squared_accumulator,
                       ::tensorflow::Input grad, ::tensorflow::Input lr,
                       ::tensorflow::Input l1, ::tensorflow::Input l2,
                       ::tensorflow::Input global_step);
  ResourceApplyAdagradDA(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       var, ::tensorflow::Input gradient_accumulator,
                       ::tensorflow::Input gradient_squared_accumulator,
                       ::tensorflow::Input grad, ::tensorflow::Input lr,
                       ::tensorflow::Input l1, ::tensorflow::Input l2,
                       ::tensorflow::Input global_step, const
                       ResourceApplyAdagradDA::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update '*var' according to the Adam algorithm.
///
/// $$\text{lr}_t := \mathrm{lr} \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$$
/// $$m_t := \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g$$
/// $$v_t := \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g^2$$
/// $$\text{var} := \begin{cases} \text{var} - (m_t \beta_1 + g \cdot (1 - \beta_1))\cdot\text{lr}_t/(\sqrt{v_t} + \epsilon), &\text{if use_nesterov}\\\\  \text{var} - m_t \cdot \text{lr}_t /(\sqrt{v_t} + \epsilon), &\text{otherwise} \end{cases}$$
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * m: Should be from a Variable().
/// * v: Should be from a Variable().
/// * beta1_power: Must be a scalar.
/// * beta2_power: Must be a scalar.
/// * lr: Scaling factor. Must be a scalar.
/// * beta1: Momentum factor. Must be a scalar.
/// * beta2: Momentum factor. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, m, and v tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// * use_nesterov: If `True`, uses the nesterov update.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyAdam {
 public:
  /// Optional attribute setters for ResourceApplyAdam
  struct Attrs {
    /// If `True`, updating of the var, m, and v tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// If `True`, uses the nesterov update.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNesterov(bool x) {
      Attrs ret = *this;
      ret.use_nesterov_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool use_nesterov_ = false;
  };
  ResourceApplyAdam(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                  ::tensorflow::Input m, ::tensorflow::Input v,
                  ::tensorflow::Input beta1_power, ::tensorflow::Input
                  beta2_power, ::tensorflow::Input lr, ::tensorflow::Input
                  beta1, ::tensorflow::Input beta2, ::tensorflow::Input
                  epsilon, ::tensorflow::Input grad);
  ResourceApplyAdam(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                  ::tensorflow::Input m, ::tensorflow::Input v,
                  ::tensorflow::Input beta1_power, ::tensorflow::Input
                  beta2_power, ::tensorflow::Input lr, ::tensorflow::Input
                  beta1, ::tensorflow::Input beta2, ::tensorflow::Input
                  epsilon, ::tensorflow::Input grad, const
                  ResourceApplyAdam::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UseNesterov(bool x) {
    return Attrs().UseNesterov(x);
  }

  Operation operation;
};

/// Update '*var' according to the Adam algorithm.
///
/// $$\text{lr}_t := \mathrm{learning_rate} * \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
/// $$m_t := \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
/// $$v_t := \beta_2 * v_{t-1} + (1 - \beta_2) * g * g$$
/// $$\hat{v}_t := max{\hat{v}_{t-1}, v_t}$$
/// $$\text{variable} := \text{variable} - \text{lr}_t * m_t / (\sqrt{\hat{v}_t} + \epsilon)$$
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * m: Should be from a Variable().
/// * v: Should be from a Variable().
/// * vhat: Should be from a Variable().
/// * beta1_power: Must be a scalar.
/// * beta2_power: Must be a scalar.
/// * lr: Scaling factor. Must be a scalar.
/// * beta1: Momentum factor. Must be a scalar.
/// * beta2: Momentum factor. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, m, and v tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyAdamWithAmsgrad {
 public:
  /// Optional attribute setters for ResourceApplyAdamWithAmsgrad
  struct Attrs {
    /// If `True`, updating of the var, m, and v tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceApplyAdamWithAmsgrad(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input m,
                             ::tensorflow::Input v, ::tensorflow::Input vhat,
                             ::tensorflow::Input beta1_power,
                             ::tensorflow::Input beta2_power,
                             ::tensorflow::Input lr, ::tensorflow::Input beta1,
                             ::tensorflow::Input beta2, ::tensorflow::Input
                             epsilon, ::tensorflow::Input grad);
  ResourceApplyAdamWithAmsgrad(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input m,
                             ::tensorflow::Input v, ::tensorflow::Input vhat,
                             ::tensorflow::Input beta1_power,
                             ::tensorflow::Input beta2_power,
                             ::tensorflow::Input lr, ::tensorflow::Input beta1,
                             ::tensorflow::Input beta2, ::tensorflow::Input
                             epsilon, ::tensorflow::Input grad, const
                             ResourceApplyAdamWithAmsgrad::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update '*var' according to the AddSign update.
///
/// m_t <- beta1 * m_{t-1} + (1 - beta1) * g
/// update <- (alpha + sign_decay * sign(g) *sign(m)) * g
/// variable <- variable - lr_t * update
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * m: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * alpha: Must be a scalar.
/// * sign_decay: Must be a scalar.
/// * beta: Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and m tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyAddSign {
 public:
  /// Optional attribute setters for ResourceApplyAddSign
  struct Attrs {
    /// If `True`, updating of the var and m tensors is
    /// protected by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceApplyAddSign(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input m, ::tensorflow::Input lr,
                     ::tensorflow::Input alpha, ::tensorflow::Input sign_decay,
                     ::tensorflow::Input beta, ::tensorflow::Input grad);
  ResourceApplyAddSign(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input m, ::tensorflow::Input lr,
                     ::tensorflow::Input alpha, ::tensorflow::Input sign_decay,
                     ::tensorflow::Input beta, ::tensorflow::Input grad, const
                     ResourceApplyAddSign::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update '*var' according to the centered RMSProp algorithm.
///
/// The centered RMSProp algorithm uses an estimate of the centered second moment
/// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
/// uses the (uncentered) second moment. This often helps with training, but is
/// slightly more expensive in terms of computation and memory.
///
/// Note that in dense implementation of this algorithm, mg, ms, and mom will
/// update even if the grad is zero, but in this sparse implementation, mg, ms,
/// and mom will not update in iterations during which the grad is zero.
///
/// mean_square = decay * mean_square + (1-decay) * gradient ** 2
/// mean_grad = decay * mean_grad + (1-decay) * gradient
///
/// Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)
///
/// mg <- rho * mg_{t-1} + (1-rho) * grad
/// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
/// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
/// var <- var - mom
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * mg: Should be from a Variable().
/// * ms: Should be from a Variable().
/// * mom: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * rho: Decay rate. Must be a scalar.
/// * momentum: Momentum Scale. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, mg, ms, and mom tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyCenteredRMSProp {
 public:
  /// Optional attribute setters for ResourceApplyCenteredRMSProp
  struct Attrs {
    /// If `True`, updating of the var, mg, ms, and mom tensors is
    /// protected by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceApplyCenteredRMSProp(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input mg,
                             ::tensorflow::Input ms, ::tensorflow::Input mom,
                             ::tensorflow::Input lr, ::tensorflow::Input rho,
                             ::tensorflow::Input momentum, ::tensorflow::Input
                             epsilon, ::tensorflow::Input grad);
  ResourceApplyCenteredRMSProp(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input mg,
                             ::tensorflow::Input ms, ::tensorflow::Input mom,
                             ::tensorflow::Input lr, ::tensorflow::Input rho,
                             ::tensorflow::Input momentum, ::tensorflow::Input
                             epsilon, ::tensorflow::Input grad, const
                             ResourceApplyCenteredRMSProp::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update '*var' according to the Ftrl-proximal scheme.
///
/// accum_new = accum + grad * grad
/// linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
/// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
/// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * linear: Should be from a Variable().
/// * grad: The gradient.
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * lr_power: Scaling factor. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyFtrl {
 public:
  /// Optional attribute setters for ResourceApplyFtrl
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs MultiplyLinearByLr(bool x) {
      Attrs ret = *this;
      ret.multiply_linear_by_lr_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool multiply_linear_by_lr_ = false;
  };
  ResourceApplyFtrl(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                  ::tensorflow::Input accum, ::tensorflow::Input linear,
                  ::tensorflow::Input grad, ::tensorflow::Input lr,
                  ::tensorflow::Input l1, ::tensorflow::Input l2,
                  ::tensorflow::Input lr_power);
  ResourceApplyFtrl(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                  ::tensorflow::Input accum, ::tensorflow::Input linear,
                  ::tensorflow::Input grad, ::tensorflow::Input lr,
                  ::tensorflow::Input l1, ::tensorflow::Input l2,
                  ::tensorflow::Input lr_power, const ResourceApplyFtrl::Attrs&
                  attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs MultiplyLinearByLr(bool x) {
    return Attrs().MultiplyLinearByLr(x);
  }

  Operation operation;
};

/// Update '*var' according to the Ftrl-proximal scheme.
///
/// accum_new = accum + grad * grad
/// grad_with_shrinkage = grad + 2 * l2_shrinkage * var
/// linear += grad_with_shrinkage +
///     (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
/// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
/// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * linear: Should be from a Variable().
/// * grad: The gradient.
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 shrinkage regularization. Must be a scalar.
/// * lr_power: Scaling factor. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyFtrlV2 {
 public:
  /// Optional attribute setters for ResourceApplyFtrlV2
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs MultiplyLinearByLr(bool x) {
      Attrs ret = *this;
      ret.multiply_linear_by_lr_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool multiply_linear_by_lr_ = false;
  };
  ResourceApplyFtrlV2(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                    ::tensorflow::Input accum, ::tensorflow::Input linear,
                    ::tensorflow::Input grad, ::tensorflow::Input lr,
                    ::tensorflow::Input l1, ::tensorflow::Input l2,
                    ::tensorflow::Input l2_shrinkage, ::tensorflow::Input
                    lr_power);
  ResourceApplyFtrlV2(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                    ::tensorflow::Input accum, ::tensorflow::Input linear,
                    ::tensorflow::Input grad, ::tensorflow::Input lr,
                    ::tensorflow::Input l1, ::tensorflow::Input l2,
                    ::tensorflow::Input l2_shrinkage, ::tensorflow::Input
                    lr_power, const ResourceApplyFtrlV2::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs MultiplyLinearByLr(bool x) {
    return Attrs().MultiplyLinearByLr(x);
  }

  Operation operation;
};

/// Update '*var' by subtracting 'alpha' * 'delta' from it.
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * alpha: Scaling factor. Must be a scalar.
/// * delta: The change.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyGradientDescent {
 public:
  /// Optional attribute setters for ResourceApplyGradientDescent
  struct Attrs {
    /// If `True`, the subtraction will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceApplyGradientDescent(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input
                             alpha, ::tensorflow::Input delta);
  ResourceApplyGradientDescent(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input
                             alpha, ::tensorflow::Input delta, const
                             ResourceApplyGradientDescent::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update '*var' according to the momentum scheme.
///
/// Set use_nesterov = True if you want to use Nesterov momentum.
///
/// accum = accum * momentum - lr * grad
/// var += accum
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * grad: The gradient.
/// * momentum: Momentum. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// * use_nesterov: If `True`, the tensor passed to compute grad will be
/// var + momentum * accum, so in the end, the var you get is actually
/// var + momentum * accum.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyKerasMomentum {
 public:
  /// Optional attribute setters for ResourceApplyKerasMomentum
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// If `True`, the tensor passed to compute grad will be
    /// var + momentum * accum, so in the end, the var you get is actually
    /// var + momentum * accum.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNesterov(bool x) {
      Attrs ret = *this;
      ret.use_nesterov_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool use_nesterov_ = false;
  };
  ResourceApplyKerasMomentum(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input var, ::tensorflow::Input accum,
                           ::tensorflow::Input lr, ::tensorflow::Input grad,
                           ::tensorflow::Input momentum);
  ResourceApplyKerasMomentum(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input var, ::tensorflow::Input accum,
                           ::tensorflow::Input lr, ::tensorflow::Input grad,
                           ::tensorflow::Input momentum, const
                           ResourceApplyKerasMomentum::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UseNesterov(bool x) {
    return Attrs().UseNesterov(x);
  }

  Operation operation;
};

/// Update '*var' according to the momentum scheme.
///
/// Set use_nesterov = True if you want to use Nesterov momentum.
///
/// accum = accum * momentum + grad
/// var -= lr * accum
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * grad: The gradient.
/// * momentum: Momentum. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// * use_nesterov: If `True`, the tensor passed to compute grad will be
/// var - lr * momentum * accum, so in the end, the var you get is actually
/// var - lr * momentum * accum.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyMomentum {
 public:
  /// Optional attribute setters for ResourceApplyMomentum
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// If `True`, the tensor passed to compute grad will be
    /// var - lr * momentum * accum, so in the end, the var you get is actually
    /// var - lr * momentum * accum.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNesterov(bool x) {
      Attrs ret = *this;
      ret.use_nesterov_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool use_nesterov_ = false;
  };
  ResourceApplyMomentum(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      var, ::tensorflow::Input accum, ::tensorflow::Input lr,
                      ::tensorflow::Input grad, ::tensorflow::Input momentum);
  ResourceApplyMomentum(const ::tensorflow::Scope& scope, ::tensorflow::Input
                      var, ::tensorflow::Input accum, ::tensorflow::Input lr,
                      ::tensorflow::Input grad, ::tensorflow::Input momentum,
                      const ResourceApplyMomentum::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UseNesterov(bool x) {
    return Attrs().UseNesterov(x);
  }

  Operation operation;
};

/// Update '*var' according to the AddSign update.
///
/// m_t <- beta1 * m_{t-1} + (1 - beta1) * g
/// update <- exp(logbase * sign_decay * sign(g) * sign(m_t)) * g
/// variable <- variable - lr_t * update
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * m: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * logbase: Must be a scalar.
/// * sign_decay: Must be a scalar.
/// * beta: Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and m tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyPowerSign {
 public:
  /// Optional attribute setters for ResourceApplyPowerSign
  struct Attrs {
    /// If `True`, updating of the var and m tensors is
    /// protected by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceApplyPowerSign(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       var, ::tensorflow::Input m, ::tensorflow::Input lr,
                       ::tensorflow::Input logbase, ::tensorflow::Input
                       sign_decay, ::tensorflow::Input beta,
                       ::tensorflow::Input grad);
  ResourceApplyPowerSign(const ::tensorflow::Scope& scope, ::tensorflow::Input
                       var, ::tensorflow::Input m, ::tensorflow::Input lr,
                       ::tensorflow::Input logbase, ::tensorflow::Input
                       sign_decay, ::tensorflow::Input beta,
                       ::tensorflow::Input grad, const
                       ResourceApplyPowerSign::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.
///
/// accum += grad * grad
/// prox_v = var - lr * grad * (1 / sqrt(accum))
/// var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyProximalAdagrad {
 public:
  /// Optional attribute setters for ResourceApplyProximalAdagrad
  struct Attrs {
    /// If True, updating of the var and accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceApplyProximalAdagrad(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input
                             accum, ::tensorflow::Input lr, ::tensorflow::Input
                             l1, ::tensorflow::Input l2, ::tensorflow::Input
                             grad);
  ResourceApplyProximalAdagrad(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input
                             accum, ::tensorflow::Input lr, ::tensorflow::Input
                             l1, ::tensorflow::Input l2, ::tensorflow::Input
                             grad, const ResourceApplyProximalAdagrad::Attrs&
                             attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update '*var' as FOBOS algorithm with fixed learning rate.
///
/// prox_v = var - alpha * delta
/// var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * alpha: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * delta: The change.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyProximalGradientDescent {
 public:
  /// Optional attribute setters for ResourceApplyProximalGradientDescent
  struct Attrs {
    /// If True, the subtraction will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceApplyProximalGradientDescent(const ::tensorflow::Scope& scope,
                                     ::tensorflow::Input var,
                                     ::tensorflow::Input alpha,
                                     ::tensorflow::Input l1,
                                     ::tensorflow::Input l2,
                                     ::tensorflow::Input delta);
  ResourceApplyProximalGradientDescent(const ::tensorflow::Scope& scope,
                                     ::tensorflow::Input var,
                                     ::tensorflow::Input alpha,
                                     ::tensorflow::Input l1,
                                     ::tensorflow::Input l2,
                                     ::tensorflow::Input delta, const
                                     ResourceApplyProximalGradientDescent::Attrs&
                                     attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update '*var' according to the RMSProp algorithm.
///
/// Note that in dense implementation of this algorithm, ms and mom will
/// update even if the grad is zero, but in this sparse implementation, ms
/// and mom will not update in iterations during which the grad is zero.
///
/// mean_square = decay * mean_square + (1-decay) * gradient ** 2
/// Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
///
/// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
/// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
/// var <- var - mom
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * ms: Should be from a Variable().
/// * mom: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * rho: Decay rate. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, ms, and mom tensors is protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyRMSProp {
 public:
  /// Optional attribute setters for ResourceApplyRMSProp
  struct Attrs {
    /// If `True`, updating of the var, ms, and mom tensors is protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceApplyRMSProp(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input ms, ::tensorflow::Input mom,
                     ::tensorflow::Input lr, ::tensorflow::Input rho,
                     ::tensorflow::Input momentum, ::tensorflow::Input epsilon,
                     ::tensorflow::Input grad);
  ResourceApplyRMSProp(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input ms, ::tensorflow::Input mom,
                     ::tensorflow::Input lr, ::tensorflow::Input rho,
                     ::tensorflow::Input momentum, ::tensorflow::Input epsilon,
                     ::tensorflow::Input grad, const
                     ResourceApplyRMSProp::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// var: Should be from a Variable().
///
/// Args:
/// * scope: A Scope object
/// * accum: Should be from a Variable().
/// * accum_update: : Should be from a Variable().
/// * lr: Learning rate. Must be a scalar.
/// * rho: Decay factor. Must be a scalar.
/// * epsilon: Constant factor. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceSparseApplyAdadelta {
 public:
  /// Optional attribute setters for ResourceSparseApplyAdadelta
  struct Attrs {
    /// If True, updating of the var and accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceSparseApplyAdadelta(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input var, ::tensorflow::Input accum,
                            ::tensorflow::Input accum_update,
                            ::tensorflow::Input lr, ::tensorflow::Input rho,
                            ::tensorflow::Input epsilon, ::tensorflow::Input
                            grad, ::tensorflow::Input indices);
  ResourceSparseApplyAdadelta(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input var, ::tensorflow::Input accum,
                            ::tensorflow::Input accum_update,
                            ::tensorflow::Input lr, ::tensorflow::Input rho,
                            ::tensorflow::Input epsilon, ::tensorflow::Input
                            grad, ::tensorflow::Input indices, const
                            ResourceSparseApplyAdadelta::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update relevant entries in '*var' and '*accum' according to the adagrad scheme.
///
/// That is for rows we have grad for, we update var and accum as follows:
/// accum += grad * grad
/// var -= lr * grad * (1 / sqrt(accum))
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Learning rate. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceSparseApplyAdagrad {
 public:
  /// Optional attribute setters for ResourceSparseApplyAdagrad
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UpdateSlots(bool x) {
      Attrs ret = *this;
      ret.update_slots_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool update_slots_ = true;
  };
  ResourceSparseApplyAdagrad(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input var, ::tensorflow::Input accum,
                           ::tensorflow::Input lr, ::tensorflow::Input grad,
                           ::tensorflow::Input indices);
  ResourceSparseApplyAdagrad(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input var, ::tensorflow::Input accum,
                           ::tensorflow::Input lr, ::tensorflow::Input grad,
                           ::tensorflow::Input indices, const
                           ResourceSparseApplyAdagrad::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UpdateSlots(bool x) {
    return Attrs().UpdateSlots(x);
  }

  Operation operation;
};

/// Update entries in '*var' and '*accum' according to the proximal adagrad scheme.
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * gradient_accumulator: Should be from a Variable().
/// * gradient_squared_accumulator: Should be from a Variable().
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
/// * lr: Learning rate. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * global_step: Training step number. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceSparseApplyAdagradDA {
 public:
  /// Optional attribute setters for ResourceSparseApplyAdagradDA
  struct Attrs {
    /// If True, updating of the var and accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceSparseApplyAdagradDA(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input
                             gradient_accumulator, ::tensorflow::Input
                             gradient_squared_accumulator, ::tensorflow::Input
                             grad, ::tensorflow::Input indices,
                             ::tensorflow::Input lr, ::tensorflow::Input l1,
                             ::tensorflow::Input l2, ::tensorflow::Input
                             global_step);
  ResourceSparseApplyAdagradDA(const ::tensorflow::Scope& scope,
                             ::tensorflow::Input var, ::tensorflow::Input
                             gradient_accumulator, ::tensorflow::Input
                             gradient_squared_accumulator, ::tensorflow::Input
                             grad, ::tensorflow::Input indices,
                             ::tensorflow::Input lr, ::tensorflow::Input l1,
                             ::tensorflow::Input l2, ::tensorflow::Input
                             global_step, const
                             ResourceSparseApplyAdagradDA::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update '*var' according to the centered RMSProp algorithm.
///
/// The centered RMSProp algorithm uses an estimate of the centered second moment
/// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
/// uses the (uncentered) second moment. This often helps with training, but is
/// slightly more expensive in terms of computation and memory.
///
/// Note that in dense implementation of this algorithm, mg, ms, and mom will
/// update even if the grad is zero, but in this sparse implementation, mg, ms,
/// and mom will not update in iterations during which the grad is zero.
///
/// mean_square = decay * mean_square + (1-decay) * gradient ** 2
/// mean_grad = decay * mean_grad + (1-decay) * gradient
/// Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)
///
/// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
/// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
/// var <- var - mom
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * mg: Should be from a Variable().
/// * ms: Should be from a Variable().
/// * mom: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * rho: Decay rate. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var, ms and mom.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, mg, ms, and mom tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceSparseApplyCenteredRMSProp {
 public:
  /// Optional attribute setters for ResourceSparseApplyCenteredRMSProp
  struct Attrs {
    /// If `True`, updating of the var, mg, ms, and mom tensors is
    /// protected by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceSparseApplyCenteredRMSProp(const ::tensorflow::Scope& scope,
                                   ::tensorflow::Input var, ::tensorflow::Input
                                   mg, ::tensorflow::Input ms,
                                   ::tensorflow::Input mom, ::tensorflow::Input
                                   lr, ::tensorflow::Input rho,
                                   ::tensorflow::Input momentum,
                                   ::tensorflow::Input epsilon,
                                   ::tensorflow::Input grad,
                                   ::tensorflow::Input indices);
  ResourceSparseApplyCenteredRMSProp(const ::tensorflow::Scope& scope,
                                   ::tensorflow::Input var, ::tensorflow::Input
                                   mg, ::tensorflow::Input ms,
                                   ::tensorflow::Input mom, ::tensorflow::Input
                                   lr, ::tensorflow::Input rho,
                                   ::tensorflow::Input momentum,
                                   ::tensorflow::Input epsilon,
                                   ::tensorflow::Input grad,
                                   ::tensorflow::Input indices, const
                                   ResourceSparseApplyCenteredRMSProp::Attrs&
                                   attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update relevant entries in '*var' according to the Ftrl-proximal scheme.
///
/// That is for rows we have grad for, we update var, accum and linear as follows:
/// accum_new = accum + grad * grad
/// linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
/// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
/// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * linear: Should be from a Variable().
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * lr_power: Scaling factor. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceSparseApplyFtrl {
 public:
  /// Optional attribute setters for ResourceSparseApplyFtrl
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs MultiplyLinearByLr(bool x) {
      Attrs ret = *this;
      ret.multiply_linear_by_lr_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool multiply_linear_by_lr_ = false;
  };
  ResourceSparseApplyFtrl(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        var, ::tensorflow::Input accum, ::tensorflow::Input
                        linear, ::tensorflow::Input grad, ::tensorflow::Input
                        indices, ::tensorflow::Input lr, ::tensorflow::Input
                        l1, ::tensorflow::Input l2, ::tensorflow::Input
                        lr_power);
  ResourceSparseApplyFtrl(const ::tensorflow::Scope& scope, ::tensorflow::Input
                        var, ::tensorflow::Input accum, ::tensorflow::Input
                        linear, ::tensorflow::Input grad, ::tensorflow::Input
                        indices, ::tensorflow::Input lr, ::tensorflow::Input
                        l1, ::tensorflow::Input l2, ::tensorflow::Input
                        lr_power, const ResourceSparseApplyFtrl::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs MultiplyLinearByLr(bool x) {
    return Attrs().MultiplyLinearByLr(x);
  }

  Operation operation;
};

/// Update relevant entries in '*var' according to the Ftrl-proximal scheme.
///
/// That is for rows we have grad for, we update var, accum and linear as follows:
/// grad_with_shrinkage = grad + 2 * l2_shrinkage * var
/// accum_new = accum + grad_with_shrinkage * grad_with_shrinkage
/// linear += grad_with_shrinkage +
///     (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
/// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
/// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * linear: Should be from a Variable().
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 shrinkage regularization. Must be a scalar.
/// * lr_power: Scaling factor. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceSparseApplyFtrlV2 {
 public:
  /// Optional attribute setters for ResourceSparseApplyFtrlV2
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs MultiplyLinearByLr(bool x) {
      Attrs ret = *this;
      ret.multiply_linear_by_lr_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool multiply_linear_by_lr_ = false;
  };
  ResourceSparseApplyFtrlV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          var, ::tensorflow::Input accum, ::tensorflow::Input
                          linear, ::tensorflow::Input grad, ::tensorflow::Input
                          indices, ::tensorflow::Input lr, ::tensorflow::Input
                          l1, ::tensorflow::Input l2, ::tensorflow::Input
                          l2_shrinkage, ::tensorflow::Input lr_power);
  ResourceSparseApplyFtrlV2(const ::tensorflow::Scope& scope, ::tensorflow::Input
                          var, ::tensorflow::Input accum, ::tensorflow::Input
                          linear, ::tensorflow::Input grad, ::tensorflow::Input
                          indices, ::tensorflow::Input lr, ::tensorflow::Input
                          l1, ::tensorflow::Input l2, ::tensorflow::Input
                          l2_shrinkage, ::tensorflow::Input lr_power, const
                          ResourceSparseApplyFtrlV2::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs MultiplyLinearByLr(bool x) {
    return Attrs().MultiplyLinearByLr(x);
  }

  Operation operation;
};

/// Update relevant entries in '*var' and '*accum' according to the momentum scheme.
///
/// Set use_nesterov = True if you want to use Nesterov momentum.
///
/// That is for rows we have grad for, we update var and accum as follows:
///
/// accum = accum * momentum - lr * grad
/// var += accum
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Learning rate. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
/// * momentum: Momentum. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// * use_nesterov: If `True`, the tensor passed to compute grad will be
/// var + momentum * accum, so in the end, the var you get is actually
/// var + momentum * accum.
///
/// Returns:
/// * the created `Operation`
class ResourceSparseApplyKerasMomentum {
 public:
  /// Optional attribute setters for ResourceSparseApplyKerasMomentum
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// If `True`, the tensor passed to compute grad will be
    /// var + momentum * accum, so in the end, the var you get is actually
    /// var + momentum * accum.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNesterov(bool x) {
      Attrs ret = *this;
      ret.use_nesterov_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool use_nesterov_ = false;
  };
  ResourceSparseApplyKerasMomentum(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input var, ::tensorflow::Input
                                 accum, ::tensorflow::Input lr,
                                 ::tensorflow::Input grad, ::tensorflow::Input
                                 indices, ::tensorflow::Input momentum);
  ResourceSparseApplyKerasMomentum(const ::tensorflow::Scope& scope,
                                 ::tensorflow::Input var, ::tensorflow::Input
                                 accum, ::tensorflow::Input lr,
                                 ::tensorflow::Input grad, ::tensorflow::Input
                                 indices, ::tensorflow::Input momentum, const
                                 ResourceSparseApplyKerasMomentum::Attrs&
                                 attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UseNesterov(bool x) {
    return Attrs().UseNesterov(x);
  }

  Operation operation;
};

/// Update relevant entries in '*var' and '*accum' according to the momentum scheme.
///
/// Set use_nesterov = True if you want to use Nesterov momentum.
///
/// That is for rows we have grad for, we update var and accum as follows:
///
/// accum = accum * momentum + grad
/// var -= lr * accum
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Learning rate. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
/// * momentum: Momentum. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// * use_nesterov: If `True`, the tensor passed to compute grad will be
/// var - lr * momentum * accum, so in the end, the var you get is actually
/// var - lr * momentum * accum.
///
/// Returns:
/// * the created `Operation`
class ResourceSparseApplyMomentum {
 public:
  /// Optional attribute setters for ResourceSparseApplyMomentum
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// If `True`, the tensor passed to compute grad will be
    /// var - lr * momentum * accum, so in the end, the var you get is actually
    /// var - lr * momentum * accum.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNesterov(bool x) {
      Attrs ret = *this;
      ret.use_nesterov_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool use_nesterov_ = false;
  };
  ResourceSparseApplyMomentum(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input var, ::tensorflow::Input accum,
                            ::tensorflow::Input lr, ::tensorflow::Input grad,
                            ::tensorflow::Input indices, ::tensorflow::Input
                            momentum);
  ResourceSparseApplyMomentum(const ::tensorflow::Scope& scope,
                            ::tensorflow::Input var, ::tensorflow::Input accum,
                            ::tensorflow::Input lr, ::tensorflow::Input grad,
                            ::tensorflow::Input indices, ::tensorflow::Input
                            momentum, const ResourceSparseApplyMomentum::Attrs&
                            attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UseNesterov(bool x) {
    return Attrs().UseNesterov(x);
  }

  Operation operation;
};

/// Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.
///
/// That is for rows we have grad for, we update var and accum as follows:
/// accum += grad * grad
/// prox_v = var
/// prox_v -= lr * grad * (1 / sqrt(accum))
/// var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Learning rate. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceSparseApplyProximalAdagrad {
 public:
  /// Optional attribute setters for ResourceSparseApplyProximalAdagrad
  struct Attrs {
    /// If True, updating of the var and accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceSparseApplyProximalAdagrad(const ::tensorflow::Scope& scope,
                                   ::tensorflow::Input var, ::tensorflow::Input
                                   accum, ::tensorflow::Input lr,
                                   ::tensorflow::Input l1, ::tensorflow::Input
                                   l2, ::tensorflow::Input grad,
                                   ::tensorflow::Input indices);
  ResourceSparseApplyProximalAdagrad(const ::tensorflow::Scope& scope,
                                   ::tensorflow::Input var, ::tensorflow::Input
                                   accum, ::tensorflow::Input lr,
                                   ::tensorflow::Input l1, ::tensorflow::Input
                                   l2, ::tensorflow::Input grad,
                                   ::tensorflow::Input indices, const
                                   ResourceSparseApplyProximalAdagrad::Attrs&
                                   attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Sparse update '*var' as FOBOS algorithm with fixed learning rate.
///
/// That is for rows we have grad for, we update var as follows:
/// prox_v = var - alpha * grad
/// var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * alpha: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * the created `Operation`
class ResourceSparseApplyProximalGradientDescent {
 public:
  /// Optional attribute setters for ResourceSparseApplyProximalGradientDescent
  struct Attrs {
    /// If True, the subtraction will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceSparseApplyProximalGradientDescent(const ::tensorflow::Scope& scope,
                                           ::tensorflow::Input var,
                                           ::tensorflow::Input alpha,
                                           ::tensorflow::Input l1,
                                           ::tensorflow::Input l2,
                                           ::tensorflow::Input grad,
                                           ::tensorflow::Input indices);
  ResourceSparseApplyProximalGradientDescent(const ::tensorflow::Scope& scope,
                                           ::tensorflow::Input var,
                                           ::tensorflow::Input alpha,
                                           ::tensorflow::Input l1,
                                           ::tensorflow::Input l2,
                                           ::tensorflow::Input grad,
                                           ::tensorflow::Input indices, const
                                           ResourceSparseApplyProximalGradientDescent::Attrs&
                                           attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// Update '*var' according to the RMSProp algorithm.
///
/// Note that in dense implementation of this algorithm, ms and mom will
/// update even if the grad is zero, but in this sparse implementation, ms
/// and mom will not update in iterations during which the grad is zero.
///
/// mean_square = decay * mean_square + (1-decay) * gradient ** 2
/// Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
///
/// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
/// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
/// var <- var - mom
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * ms: Should be from a Variable().
/// * mom: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * rho: Decay rate. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var, ms and mom.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, ms, and mom tensors is protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceSparseApplyRMSProp {
 public:
  /// Optional attribute setters for ResourceSparseApplyRMSProp
  struct Attrs {
    /// If `True`, updating of the var, ms, and mom tensors is protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceSparseApplyRMSProp(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input var, ::tensorflow::Input ms,
                           ::tensorflow::Input mom, ::tensorflow::Input lr,
                           ::tensorflow::Input rho, ::tensorflow::Input
                           momentum, ::tensorflow::Input epsilon,
                           ::tensorflow::Input grad, ::tensorflow::Input
                           indices);
  ResourceSparseApplyRMSProp(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input var, ::tensorflow::Input ms,
                           ::tensorflow::Input mom, ::tensorflow::Input lr,
                           ::tensorflow::Input rho, ::tensorflow::Input
                           momentum, ::tensorflow::Input epsilon,
                           ::tensorflow::Input grad, ::tensorflow::Input
                           indices, const ResourceSparseApplyRMSProp::Attrs&
                           attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

/// var: Should be from a Variable().
///
/// Args:
/// * scope: A Scope object
/// * accum: Should be from a Variable().
/// * accum_update: : Should be from a Variable().
/// * lr: Learning rate. Must be a scalar.
/// * rho: Decay factor. Must be a scalar.
/// * epsilon: Constant factor. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as "var".
class SparseApplyAdadelta {
 public:
  /// Optional attribute setters for SparseApplyAdadelta
  struct Attrs {
    /// If True, updating of the var and accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  SparseApplyAdadelta(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                    ::tensorflow::Input accum, ::tensorflow::Input
                    accum_update, ::tensorflow::Input lr, ::tensorflow::Input
                    rho, ::tensorflow::Input epsilon, ::tensorflow::Input grad,
                    ::tensorflow::Input indices);
  SparseApplyAdadelta(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                    ::tensorflow::Input accum, ::tensorflow::Input
                    accum_update, ::tensorflow::Input lr, ::tensorflow::Input
                    rho, ::tensorflow::Input epsilon, ::tensorflow::Input grad,
                    ::tensorflow::Input indices, const
                    SparseApplyAdadelta::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update relevant entries in '*var' and '*accum' according to the adagrad scheme.
///
/// That is for rows we have grad for, we update var and accum as follows:
/// $$accum += grad * grad$$
/// $$var -= lr * grad * (1 / sqrt(accum))$$
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Learning rate. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class SparseApplyAdagrad {
 public:
  /// Optional attribute setters for SparseApplyAdagrad
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to true
    TF_MUST_USE_RESULT Attrs UpdateSlots(bool x) {
      Attrs ret = *this;
      ret.update_slots_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool update_slots_ = true;
  };
  SparseApplyAdagrad(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                   ::tensorflow::Input accum, ::tensorflow::Input lr,
                   ::tensorflow::Input grad, ::tensorflow::Input indices);
  SparseApplyAdagrad(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                   ::tensorflow::Input accum, ::tensorflow::Input lr,
                   ::tensorflow::Input grad, ::tensorflow::Input indices, const
                   SparseApplyAdagrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UpdateSlots(bool x) {
    return Attrs().UpdateSlots(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update entries in '*var' and '*accum' according to the proximal adagrad scheme.
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * gradient_accumulator: Should be from a Variable().
/// * gradient_squared_accumulator: Should be from a Variable().
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
/// * lr: Learning rate. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * global_step: Training step number. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as "var".
class SparseApplyAdagradDA {
 public:
  /// Optional attribute setters for SparseApplyAdagradDA
  struct Attrs {
    /// If True, updating of the var and accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  SparseApplyAdagradDA(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input gradient_accumulator,
                     ::tensorflow::Input gradient_squared_accumulator,
                     ::tensorflow::Input grad, ::tensorflow::Input indices,
                     ::tensorflow::Input lr, ::tensorflow::Input l1,
                     ::tensorflow::Input l2, ::tensorflow::Input global_step);
  SparseApplyAdagradDA(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                     ::tensorflow::Input gradient_accumulator,
                     ::tensorflow::Input gradient_squared_accumulator,
                     ::tensorflow::Input grad, ::tensorflow::Input indices,
                     ::tensorflow::Input lr, ::tensorflow::Input l1,
                     ::tensorflow::Input l2, ::tensorflow::Input global_step,
                     const SparseApplyAdagradDA::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the centered RMSProp algorithm.
///
/// The centered RMSProp algorithm uses an estimate of the centered second moment
/// (i.e., the variance) for normalization, as opposed to regular RMSProp, which
/// uses the (uncentered) second moment. This often helps with training, but is
/// slightly more expensive in terms of computation and memory.
///
/// Note that in dense implementation of this algorithm, mg, ms, and mom will
/// update even if the grad is zero, but in this sparse implementation, mg, ms,
/// and mom will not update in iterations during which the grad is zero.
///
/// mean_square = decay * mean_square + (1-decay) * gradient ** 2
/// mean_grad = decay * mean_grad + (1-decay) * gradient
/// Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)
///
/// $$ms <- rho * ms_{t-1} + (1-rho) * grad * grad$$
/// $$mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)$$
/// $$var <- var - mom$$
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * mg: Should be from a Variable().
/// * ms: Should be from a Variable().
/// * mom: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * rho: Decay rate. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var, ms and mom.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, mg, ms, and mom tensors is
/// protected by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class SparseApplyCenteredRMSProp {
 public:
  /// Optional attribute setters for SparseApplyCenteredRMSProp
  struct Attrs {
    /// If `True`, updating of the var, mg, ms, and mom tensors is
    /// protected by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  SparseApplyCenteredRMSProp(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input var, ::tensorflow::Input mg,
                           ::tensorflow::Input ms, ::tensorflow::Input mom,
                           ::tensorflow::Input lr, ::tensorflow::Input rho,
                           ::tensorflow::Input momentum, ::tensorflow::Input
                           epsilon, ::tensorflow::Input grad,
                           ::tensorflow::Input indices);
  SparseApplyCenteredRMSProp(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input var, ::tensorflow::Input mg,
                           ::tensorflow::Input ms, ::tensorflow::Input mom,
                           ::tensorflow::Input lr, ::tensorflow::Input rho,
                           ::tensorflow::Input momentum, ::tensorflow::Input
                           epsilon, ::tensorflow::Input grad,
                           ::tensorflow::Input indices, const
                           SparseApplyCenteredRMSProp::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update relevant entries in '*var' according to the Ftrl-proximal scheme.
///
/// That is for rows we have grad for, we update var, accum and linear as follows:
/// $$accum_new = accum + grad * grad$$
/// $$linear += grad + (accum_{new}^{-lr_{power}} - accum^{-lr_{power}} / lr * var$$
/// $$quadratic = 1.0 / (accum_{new}^{lr_{power}} * lr) + 2 * l2$$
/// $$var = (sign(linear) * l1 - linear) / quadratic\ if\ |linear| > l1\ else\ 0.0$$
/// $$accum = accum_{new}$$
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * linear: Should be from a Variable().
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * lr_power: Scaling factor. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class SparseApplyFtrl {
 public:
  /// Optional attribute setters for SparseApplyFtrl
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs MultiplyLinearByLr(bool x) {
      Attrs ret = *this;
      ret.multiply_linear_by_lr_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool multiply_linear_by_lr_ = false;
  };
  SparseApplyFtrl(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                ::tensorflow::Input accum, ::tensorflow::Input linear,
                ::tensorflow::Input grad, ::tensorflow::Input indices,
                ::tensorflow::Input lr, ::tensorflow::Input l1,
                ::tensorflow::Input l2, ::tensorflow::Input lr_power);
  SparseApplyFtrl(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                ::tensorflow::Input accum, ::tensorflow::Input linear,
                ::tensorflow::Input grad, ::tensorflow::Input indices,
                ::tensorflow::Input lr, ::tensorflow::Input l1,
                ::tensorflow::Input l2, ::tensorflow::Input lr_power, const
                SparseApplyFtrl::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs MultiplyLinearByLr(bool x) {
    return Attrs().MultiplyLinearByLr(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update relevant entries in '*var' according to the Ftrl-proximal scheme.
///
/// That is for rows we have grad for, we update var, accum and linear as follows:
/// grad_with_shrinkage = grad + 2 * l2_shrinkage * var
/// accum_new = accum + grad * grad
/// linear += grad_with_shrinkage -
///     (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
/// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
/// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
/// accum = accum_new
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * linear: Should be from a Variable().
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
/// * lr: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 shrinkage regularization. Must be a scalar.
/// * lr_power: Scaling factor. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class SparseApplyFtrlV2 {
 public:
  /// Optional attribute setters for SparseApplyFtrlV2
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// Defaults to false
    TF_MUST_USE_RESULT Attrs MultiplyLinearByLr(bool x) {
      Attrs ret = *this;
      ret.multiply_linear_by_lr_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool multiply_linear_by_lr_ = false;
  };
  SparseApplyFtrlV2(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                  ::tensorflow::Input accum, ::tensorflow::Input linear,
                  ::tensorflow::Input grad, ::tensorflow::Input indices,
                  ::tensorflow::Input lr, ::tensorflow::Input l1,
                  ::tensorflow::Input l2, ::tensorflow::Input l2_shrinkage,
                  ::tensorflow::Input lr_power);
  SparseApplyFtrlV2(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                  ::tensorflow::Input accum, ::tensorflow::Input linear,
                  ::tensorflow::Input grad, ::tensorflow::Input indices,
                  ::tensorflow::Input lr, ::tensorflow::Input l1,
                  ::tensorflow::Input l2, ::tensorflow::Input l2_shrinkage,
                  ::tensorflow::Input lr_power, const SparseApplyFtrlV2::Attrs&
                  attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs MultiplyLinearByLr(bool x) {
    return Attrs().MultiplyLinearByLr(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update relevant entries in '*var' and '*accum' according to the momentum scheme.
///
/// Set use_nesterov = True if you want to use Nesterov momentum.
///
/// That is for rows we have grad for, we update var and accum as follows:
///
/// $$accum = accum * momentum + grad$$
/// $$var -= lr * accum$$
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Learning rate. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
/// * momentum: Momentum. Must be a scalar.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var and accum tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
/// * use_nesterov: If `True`, the tensor passed to compute grad will be
/// var - lr * momentum * accum, so in the end, the var you get is actually
/// var - lr * momentum * accum.
///
/// Returns:
/// * `Output`: Same as "var".
class SparseApplyMomentum {
 public:
  /// Optional attribute setters for SparseApplyMomentum
  struct Attrs {
    /// If `True`, updating of the var and accum tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    /// If `True`, the tensor passed to compute grad will be
    /// var - lr * momentum * accum, so in the end, the var you get is actually
    /// var - lr * momentum * accum.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseNesterov(bool x) {
      Attrs ret = *this;
      ret.use_nesterov_ = x;
      return ret;
    }

    bool use_locking_ = false;
    bool use_nesterov_ = false;
  };
  SparseApplyMomentum(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                    ::tensorflow::Input accum, ::tensorflow::Input lr,
                    ::tensorflow::Input grad, ::tensorflow::Input indices,
                    ::tensorflow::Input momentum);
  SparseApplyMomentum(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                    ::tensorflow::Input accum, ::tensorflow::Input lr,
                    ::tensorflow::Input grad, ::tensorflow::Input indices,
                    ::tensorflow::Input momentum, const
                    SparseApplyMomentum::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }
  static Attrs UseNesterov(bool x) {
    return Attrs().UseNesterov(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.
///
/// That is for rows we have grad for, we update var and accum as follows:
/// $$accum += grad * grad$$
/// $$prox_v = var$$
/// $$prox_v -= lr * grad * (1 / sqrt(accum))$$
/// $$var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}$$
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * accum: Should be from a Variable().
/// * lr: Learning rate. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, updating of the var and accum tensors will be protected by
/// a lock; otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as "var".
class SparseApplyProximalAdagrad {
 public:
  /// Optional attribute setters for SparseApplyProximalAdagrad
  struct Attrs {
    /// If True, updating of the var and accum tensors will be protected by
    /// a lock; otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  SparseApplyProximalAdagrad(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input var, ::tensorflow::Input accum,
                           ::tensorflow::Input lr, ::tensorflow::Input l1,
                           ::tensorflow::Input l2, ::tensorflow::Input grad,
                           ::tensorflow::Input indices);
  SparseApplyProximalAdagrad(const ::tensorflow::Scope& scope,
                           ::tensorflow::Input var, ::tensorflow::Input accum,
                           ::tensorflow::Input lr, ::tensorflow::Input l1,
                           ::tensorflow::Input l2, ::tensorflow::Input grad,
                           ::tensorflow::Input indices, const
                           SparseApplyProximalAdagrad::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Sparse update '*var' as FOBOS algorithm with fixed learning rate.
///
/// That is for rows we have grad for, we update var as follows:
/// $$prox_v = var - alpha * grad$$
/// $$var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}$$
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * alpha: Scaling factor. Must be a scalar.
/// * l1: L1 regularization. Must be a scalar.
/// * l2: L2 regularization. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var and accum.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If True, the subtraction will be protected by a lock;
/// otherwise the behavior is undefined, but may exhibit less contention.
///
/// Returns:
/// * `Output`: Same as "var".
class SparseApplyProximalGradientDescent {
 public:
  /// Optional attribute setters for SparseApplyProximalGradientDescent
  struct Attrs {
    /// If True, the subtraction will be protected by a lock;
    /// otherwise the behavior is undefined, but may exhibit less contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  SparseApplyProximalGradientDescent(const ::tensorflow::Scope& scope,
                                   ::tensorflow::Input var, ::tensorflow::Input
                                   alpha, ::tensorflow::Input l1,
                                   ::tensorflow::Input l2, ::tensorflow::Input
                                   grad, ::tensorflow::Input indices);
  SparseApplyProximalGradientDescent(const ::tensorflow::Scope& scope,
                                   ::tensorflow::Input var, ::tensorflow::Input
                                   alpha, ::tensorflow::Input l1,
                                   ::tensorflow::Input l2, ::tensorflow::Input
                                   grad, ::tensorflow::Input indices, const
                                   SparseApplyProximalGradientDescent::Attrs&
                                   attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// Update '*var' according to the RMSProp algorithm.
///
/// Note that in dense implementation of this algorithm, ms and mom will
/// update even if the grad is zero, but in this sparse implementation, ms
/// and mom will not update in iterations during which the grad is zero.
///
/// mean_square = decay * mean_square + (1-decay) * gradient ** 2
/// Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
///
/// $$ms <- rho * ms_{t-1} + (1-rho) * grad * grad$$
/// $$mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)$$
/// $$var <- var - mom$$
///
/// Args:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * ms: Should be from a Variable().
/// * mom: Should be from a Variable().
/// * lr: Scaling factor. Must be a scalar.
/// * rho: Decay rate. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
/// * indices: A vector of indices into the first dimension of var, ms and mom.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, ms, and mom tensors is protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class SparseApplyRMSProp {
 public:
  /// Optional attribute setters for SparseApplyRMSProp
  struct Attrs {
    /// If `True`, updating of the var, ms, and mom tensors is protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  SparseApplyRMSProp(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                   ::tensorflow::Input ms, ::tensorflow::Input mom,
                   ::tensorflow::Input lr, ::tensorflow::Input rho,
                   ::tensorflow::Input momentum, ::tensorflow::Input epsilon,
                   ::tensorflow::Input grad, ::tensorflow::Input indices);
  SparseApplyRMSProp(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                   ::tensorflow::Input ms, ::tensorflow::Input mom,
                   ::tensorflow::Input lr, ::tensorflow::Input rho,
                   ::tensorflow::Input momentum, ::tensorflow::Input epsilon,
                   ::tensorflow::Input grad, ::tensorflow::Input indices, const
                   SparseApplyRMSProp::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
  ::tensorflow::Output out;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_TRAINING_OPS_H_
