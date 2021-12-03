"""Library of helper classes of optimizer."""


class GradientsClipOption:
  """Gradients clip option for optimizer class.

  Attributes:
    clipnorm: float. If set, the gradient of each weight is individually clipped
      so that its norm is no higher than this value.
    clipvalue: float. If set, the gradient of each weight is clipped to be no
      higher than this value.
    global_clipnorm: float. If set, the gradient of all weights is clipped so
      that their global norm is no higher than this value.
  """

  def __init__(self, clipnorm=None, clipvalue=None, global_clipnorm=None):
    if clipnorm is not None and global_clipnorm is not None:
      raise ValueError(f"At most one of `clipnorm` and `global_clipnorm` can "
                       f"be set. Received: clipnorm={clipnorm}, "
                       f"global_clipnorm={global_clipnorm}.")

    if clipnorm and clipnorm <= 0:
      raise ValueError("Clipnorm should be a positive number, but received "
                       f"clipnorm={clipnorm}.")
    if global_clipnorm and global_clipnorm <= 0:
      raise ValueError("global_clipnorm should be a positive number, but "
                       f"received global_clipnorm={global_clipnorm}.")
    if clipvalue and clipvalue <= 0:
      raise ValueError("clipvalue should be a positive number, but received "
                       f"clipvalue={clipvalue}.")
    self.clipnorm = clipnorm
    self.global_clipnorm = global_clipnorm
    self.clipvalue = clipvalue

  def get_config(self):
    return {
        "clipnorm": self.clipnorm,
        "global_clipnorm": self.global_clipnorm,
        "clipvalue": self.clipvalue,
    }


class EMAOption:
  # TODO(b/207532340): Add examples on how to use this EMAOption.
  """EMA option for optimizer class.

  Attributes:
    use_ema: boolean, default to False. If True, exponential moving average
      (EMA) is applied. EMA consists of computing an exponential moving average
      of the weights of the model (as the weight values change after each
      training batch), and periodically overwriting the weights with their
      moving average.
    ema_momentum: float, default to 0.99. Only used if `use_ema=True`. This is
      the momentum to use when computing the EMA of the model's weights:
        `new_average = ema_momentum * old_average + (1 - ema_momentum) *
        current_variable_value`.
    ema_overwrite_frequency: int or None, default to 100. Only used if
      `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations, we
      overwrite the model variable by its stored moving average. If None, we do
      not overwrite model variables in the middle of training, and users need to
      explicitly overwrite the model variable by calling
      `finalize_variable_update()`.
  """

  def __init__(self,
               use_ema=False,
               ema_momentum=0.99,
               ema_overwrite_frequency=100):
    self.use_ema = use_ema
    if use_ema:
      # Verify the arguments related to EMA.
      if ema_momentum > 1 or ema_momentum < 0:
        raise ValueError("`ema_momentum` must be in the range [0, 1]. "
                         f"Received: ema_momentum={ema_momentum}")
      if ema_overwrite_frequency and (not isinstance(
          ema_overwrite_frequency, int) or ema_overwrite_frequency < 1):
        raise ValueError(
            "`ema_overwrite_frequency` must be an integer > 1 or None. "
            f"Received: ema_overwrite_frequency={ema_overwrite_frequency}")
    self.ema_momentum = ema_momentum
    self.ema_overwrite_frequency = ema_overwrite_frequency

  def get_config(self):
    return {
        "use_ema": self.use_ema,
        "ema_momentum": self.ema_momentum,
        "ema_overwrite_frequency": self.ema_overwrite_frequency,
    }
