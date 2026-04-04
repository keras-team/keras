import torch

from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class Adam(torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Adam):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        # Use ._value directly to skip _maybe_autocast overhead
        # (optimizer state doesn't need autocast).
        _get_idx = self._get_variable_index
        var_tensors = [v._value for v in variables]

        dtype = var_tensors[0].dtype
        device = var_tensors[0].device

        lr = (
            learning_rate._value
            if hasattr(learning_rate, "_value")
            else learning_rate
        )
        if isinstance(lr, torch.Tensor):
            lr = lr.to(dtype=dtype, device=device)
        else:
            lr = torch.as_tensor(lr, dtype=dtype, device=device)
        local_step = self._iterations._value.to(dtype=dtype, device=device) + 1

        beta_1 = self.beta_1
        beta_2 = self.beta_2
        beta_1_power = torch.pow(beta_1, local_step)
        beta_2_power = torch.pow(beta_2, local_step)
        alpha = lr * torch.sqrt(1.0 - beta_2_power) / (1.0 - beta_1_power)

        m_list = [self._momentums[_get_idx(v)]._value for v in variables]
        v_list = [self._velocities[_get_idx(v)]._value for v in variables]

        torch._foreach_mul_(m_list, beta_1)
        torch._foreach_add_(m_list, grads, alpha=1 - beta_1)

        torch._foreach_mul_(v_list, beta_2)
        torch._foreach_add_(
            v_list, torch._foreach_mul(grads, grads), alpha=1 - beta_2
        )

        if self.amsgrad:
            v_hat_list = [
                self._velocity_hats[_get_idx(v)]._value for v in variables
            ]
            torch._foreach_maximum_(v_hat_list, v_list)
            v_list = v_hat_list

        torch._foreach_add_(
            var_tensors,
            torch._foreach_div(
                torch._foreach_mul(m_list, alpha),
                torch._foreach_add(torch._foreach_sqrt(v_list), self.epsilon),
            ),
            alpha=-1,
        )
