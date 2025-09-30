import numpy as np
from typing import List, Dict, Any, Tuple
import keras
from keras import optimizers
import logging
import re

from keras.src.backend.distributed import backend_resolver
from keras.src.backend.distributed.base import DistributedBackend


import numpy as np
from typing import List, Dict, Any
import keras
from keras import optimizers
import logging
import re 


logger = logging.getLogger(__name__)


class CoordinatedOptimizer:
    def __init__(self, base_optimizer: optimizers.Optimizer, world_size: int,
                 distributed_backend: str = 'auto', rank: int = 0, shard_optimizer_states: bool = True, tensor_parallel_config=None):

        self.base_optimizer = base_optimizer
        self.world_size = world_size
        self.rank = rank
        self.shard_optimizer_states = shard_optimizer_states
        self.tensor_parallel_config = tensor_parallel_config
        self.sharded_states = {}

        if backend_resolver.get_distributed_backend:
            try:
                self.distributed_backend = backend_resolver.get_distributed_backend(distributed_backend, world_size, rank)
                logger.info(f"Using distributed backend: {type(self.distributed_backend).__name__}")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed backend: {e}")
                self.distributed_backend = None
        else:
            self.distributed_backend = None
            logger.warning("Distributed backend not available, using fallback")

        if self.shard_optimizer_states:
            if not getattr(self.base_optimizer, 'built', False):
                logger.error("Optimizer state sharding requires a pre-built base_optimizer. "
                             "The base optimizer has no variables to shard.")
                self.shard_optimizer_states = False
            else:
                self._initialize_sharded_states()

    def _get_actual_optimizer_state(self) -> Dict[str, Any]:
        state_dict = {}

        for var in self.base_optimizer.variables:
            raw_name = getattr(var, 'name', None)
            raw_path = getattr(var, 'path', None)
            identifier = (raw_path or raw_name or str(var))
            parts = identifier.split('/')
            tail = parts[-1]
            tail_lower = tail.lower()

            if 'iteration' in tail_lower or tail_lower in {'iter', 't'}:
                state_dict['t'] = var
                continue
            if 'learning_rate' in tail_lower or tail_lower in {'lr'}:
                state_dict['lr'] = var
                continue

            state_name = None
            param_name = None
            if tail_lower.endswith('_momentum'):
                state_name = 'momentum'
                param_name = tail[: -len('_momentum')]
            elif tail_lower.endswith('_velocity'):
                state_name = 'velocity'
                param_name = tail[: -len('_velocity')]
            elif tail_lower.endswith('_m'):
                state_name = 'm'
                param_name = tail[: -len('_m')]
            elif tail_lower.endswith('_v'):
                state_name = 'v'
                param_name = tail[: -len('_v')]
            else:
                state_name = 'state'
                param_name = tail

            if state_name not in state_dict:
                state_dict[state_name] = {}
            state_dict[state_name][param_name] = var

        return state_dict

    def _initialize_sharded_states(self):
        logger.info("Initializing sharded optimizer states...")
        if not self.shard_optimizer_states:
            logger.warning("Sharding is disabled; skipping initialization.")
            return

        try:
            base_state = self._get_actual_optimizer_state()
            if not base_state:
                logger.error("Failed to get optimizer state. Aborting sharding.")
                self.shard_optimizer_states = False
                return

            self.sharded_states = {}

            for state_name, state_value in base_state.items():
                if isinstance(state_value, dict):
                    self.sharded_states[state_name] = {}
                    for param_name, param_state_var in state_value.items():
                        sharding_dim = 0
                        rule_found = False

                        if self.tensor_parallel_config:
                            
                            # --- ADD THIS LINE ---
                            # Normalize path separator from '/' (Keras) to '.' (our config)
                            normalized_param_name = param_name.replace('/', '.')
                            # --- END ADD ---

                            for pattern, action in self.tensor_parallel_config.state_rules.items():
                                
                                # --- CHANGE THIS LINE ---
                                if re.search(pattern, normalized_param_name):
                                # --- END CHANGE ---
                                    if hasattr(action, 'dim'):
                                        sharding_dim = action.dim
                                        rule_found = True
                                        logger.debug(
                                            f"Found rule for '{param_name}': "
                                            f"sharding state on dim={sharding_dim}."
                                        )
                                        break
                        
                        if not rule_found:
                            logger.debug(
                                f"No specific rule for '{param_name}'. "
                                f"Defaulting to sharding state on dim={sharding_dim}."
                            )

                        self.sharded_states[state_name][param_name] = self._partition_state_across_shards(
                            param_state_var,
                            dim=sharding_dim
                        )

                else:
                    self.sharded_states[state_name] = self._partition_state_across_shards(
                        state_value,
                        dim=0
                    )
            
            logger.info(f"✅ Sharded optimizer states initialized successfully: {list(self.sharded_states.keys())}")

        except Exception as e:
            logger.error(
                f"Failed during sharded state initialization: {e}. "
                "Falling back to replicated (non-sharded) optimizer states.",
                exc_info=True
            )
            self.shard_optimizer_states = False
            self.sharded_states = {}
    
    def _get_base_optimizer_state_structure(self):
        """Get the structure of the base optimizer's state."""
        try:
            import numpy as np
            dummy_var = keras.Variable(np.array([1.0]))
            
            if hasattr(self.base_optimizer, 'get_updates'):
                updates = self.base_optimizer.get_updates([dummy_var], [np.array([0.0])])
                state_structure = {}
                
                for update in updates:
                    if hasattr(update, 'name') and 'm' in update.name:
                        state_structure['m'] = {'dummy': np.array([0.0])}
                    elif hasattr(update, 'name') and 'v' in update.name:
                        state_structure['v'] = {'dummy': np.array([0.0])}
                
                return state_structure
            else:
                if isinstance(self.base_optimizer, optimizers.Adam):
                    return {
                        'm': {'dummy': np.array([0.0])},  # First moment
                        'v': {'dummy': np.array([0.0])},  # Second moment
                        't': 0  # Time step
                    }
                elif isinstance(self.base_optimizer, optimizers.SGD):
                    return {
                        'momentum': {'dummy': np.array([0.0])}  # Momentum buffer
                    }
                else:
                    return {'dummy': np.array([0.0])}
                    
        except Exception as e:
            logger.warning(f"Could not determine optimizer state structure: {e}")
            return {'dummy': np.array([0.0])}

    def _partition_state_across_shards(self, state_variable: any, dim: int):
        try:
            shape = tuple(getattr(state_variable, 'shape', ()))
            dtype = np.float32
            if hasattr(state_variable, 'dtype'):
                try:
                    dt = state_variable.dtype
                    if hasattr(dt, 'as_numpy_dtype'):
                        dtype = np.dtype(dt.as_numpy_dtype)
                    else:
                        dtype = np.dtype(dt)
                except TypeError:
                    logger.debug("Could not convert tensor dtype to numpy, using float32.")
                    dtype = np.float32

            state_array = np.zeros(shape, dtype=dtype)

            if state_array.ndim > dim and shape[dim] > 0:
                logger.debug(f"Partitioning state of shape {shape} along axis {dim}.")
                return np.array_split(state_array, self.world_size, axis=dim)
            else:
                if state_array.ndim > 0 and shape[0] > 0:
                    logger.debug(f"Cannot split shape {shape} on axis {dim}. Falling back to axis 0.")
                    return np.array_split(state_array, self.world_size, axis=0)
                else:
                    logger.debug(f"State is a scalar of shape {shape}. Replicating.")
                    return [np.copy(state_array) for _ in range(self.world_size)]

        except Exception as e:
            param_name = getattr(state_variable, 'name', 'N/A')
            logger.warning(
                f"Failed to partition optimizer state '{param_name}': {e}. "
                "Replicating the state variable across all devices as a fallback."
            )
            return [state_variable] * self.world_size

    def get_config(self):
        return {
            'base_optimizer': self.base_optimizer.get_config(),
            'world_size': self.world_size,
            'shard_optimizer_states': self.shard_optimizer_states
        }
    
    def apply_gradients(self, gradients_and_vars: List[List[tuple]], shard_models: List):
        if len(gradients_and_vars) != self.world_size:
            raise ValueError(f"Expected {self.world_size} gradient sets, got {len(gradients_and_vars)}")
        
        synchronized_gradients = self._synchronize_gradients(gradients_and_vars)
        
        if self.shard_optimizer_states and self.sharded_states:
            logger.info("Applying gradients with SHARDED optimizer states")
            self._apply_gradients_with_sharded_states(synchronized_gradients, shard_models)
        else:
            logger.info("Applying gradients with REPLICATED optimizer states")
            self._apply_gradients_with_replicated_states(synchronized_gradients, shard_models)
    
    def _apply_gradients_with_sharded_states(self, synchronized_gradients: List[List[tuple]], shard_models: List):
        try:
            for shard_idx, (shard_grads, shard_model) in enumerate(zip(synchronized_gradients, shard_models)):
                logger.info(f"Updating shard {shard_idx} with sharded optimizer states")
                
                local_states = self._get_local_optimizer_states(shard_idx)
                self._update_shard_with_local_states(shard_idx, shard_grads, shard_model, local_states)
                
        except Exception as e:
            logger.error(f"Failed to apply gradients with sharded states: {e}")
            self._apply_gradients_with_replicated_states(synchronized_gradients, shard_models)
    
    def _apply_gradients_with_replicated_states(self, synchronized_gradients: List[List[tuple]], shard_models: List):
        for i, (shard_opt, shard_model) in enumerate(zip(self.shard_optimizers, shard_models)):
            shard_opt.apply_gradients(synchronized_gradients[i])
    
    def _get_local_optimizer_states(self, shard_idx: int):
        local_states = {}
        
        for state_name, state_value in self.sharded_states.items():
            if isinstance(state_value, dict):
                local_states[state_name] = {}
                for param_name, param_states in state_value.items():
                    if shard_idx < len(param_states):
                        local_states[state_name][param_name] = param_states[shard_idx]
                    else:
                        local_states[state_name][param_name] = param_states[0]
            else:
                if shard_idx < len(state_value):
                    local_states[state_name] = state_value[shard_idx]
                else:
                    local_states[state_name] = state_value[0]
        
        return local_states
    
    def _update_shard_with_local_states(self, shard_idx: int, shard_grads: List[tuple], 
                                      shard_model, local_states: dict):
        try:
            shard_opt = self.shard_optimizers[shard_idx]
            self._update_optimizer_internal_state(shard_opt, local_states)
            shard_opt.apply_gradients(shard_grads)
            logger.info(f"Shard {shard_idx} updated successfully with local states")
            
        except Exception as e:
            logger.error(f"Failed to update shard {shard_idx} with local states: {e}")
            shard_opt.apply_gradients(shard_grads)

    def _update_optimizer_internal_state(self, optimizer, local_states: dict):
        try:
            if not hasattr(optimizer, 'variables') or not optimizer.variables:
                logger.warning(f"Optimizer '{optimizer.name}' has no variables to update. It may not be built yet.")
                return

            optimizer_var_map = {}
            for var in optimizer.variables:
                identifier = getattr(var, 'path', getattr(var, 'name', str(var)))
                parts = identifier.split('/')
                tail = parts[-1]
                tail_lower = tail.lower()

                if 'iteration' in tail_lower or tail_lower in {'iter', 't'}:
                    optimizer_var_map[('t', None)] = var
                    continue
                if 'learning_rate' in tail_lower or tail_lower in {'lr'}:
                    optimizer_var_map[('lr', None)] = var
                    continue

                state_name = None
                param_name_in_opt = None
                
                if tail_lower.endswith('_momentum'):
                    state_name = 'momentum'
                    param_name_in_opt = tail[:-len('_momentum')]
                elif tail_lower.endswith('_velocity'):
                    state_name = 'velocity'
                    param_name_in_opt = tail[:-len('_velocity')]
                elif tail_lower.endswith('_m'):
                    state_name = 'm'
                    param_name_in_opt = tail[:-len('_m')]
                elif tail_lower.endswith('_v'):
                    state_name = 'v'
                    param_name_in_opt = tail[:-len('_v')]

                if state_name and param_name_in_opt:
                    optimizer_var_map[(state_name, param_name_in_opt)] = var

            updated_vars_count = 0
            for state_name, state_value in local_states.items():
                if isinstance(state_value, dict):
                    for param_name, local_param_state in state_value.items():
                        key = (state_name, param_name)
                        if key in optimizer_var_map:
                            optimizer_var_map[key].assign(local_param_state)
                            updated_vars_count += 1
                        else:
                            logger.warning(f"Could not find matching variable in optimizer for local state '{state_name}/{param_name}'.")
                else:
                    key = (state_name, None)
                    if key in optimizer_var_map:
                        optimizer_var_map[key].assign(state_value)
                        updated_vars_count += 1
                    else:
                        logger.warning(f"Could not find matching scalar variable in optimizer for local state '{state_name}'.")

            if updated_vars_count > 0:
                logger.info(f"Successfully updated {updated_vars_count} internal state variables in optimizer '{optimizer.name}'.")

        except Exception as e:
            logger.error(f"Failed to update optimizer internal state for '{optimizer.name}': {e}", exc_info=True)
    
    def _synchronize_gradients(self, gradients_and_vars: List[List[tuple]]) -> List[List[tuple]]:
        if not self.tensor_parallel_config:
            return gradients_and_vars

        column_parallel_patterns = set()
        for pattern, action in self.tensor_parallel_config.state_rules.items():
            if hasattr(action, 'sharding_type') and action.sharding_type == 'column':
                column_parallel_patterns.add(pattern)
        
        num_weights = len(gradients_and_vars[0])
        
        for i in range(num_weights):
            variable = gradients_and_vars[0][i][1]
            
            needs_sync = False
            import re
            for pattern in column_parallel_patterns:
                 if re.search(pattern.strip('$').strip('^'), variable.name):
                     needs_sync = True
                     break
            
            if needs_sync:
                grads_to_reduce = [gradients_and_vars[shard_idx][i][0] 
                                   for shard_idx in range(self.world_size)]
                
                if any(g is not None for g in grads_to_reduce):
                    synced_grad = self._allreduce_gradients(grads_to_reduce)
                
                    for shard_idx in range(self.world_size):
                        original_var = gradients_and_vars[shard_idx][i][1]
                        gradients_and_vars[shard_idx][i] = (synced_grad[shard_idx], original_var)

        return gradients_and_vars
    
    def _allreduce_gradients(self, gradients: List[Any]) -> List[Any]:
        if self.distributed_backend is not None and self.distributed_backend.is_initialized:
            try:
                logger.info("Using REAL distributed backend for AllReduce")
                numpy_gradient = keras.ops.convert_to_numpy(gradients[0])
                
                synchronized_numpy = self.distributed_backend.allreduce(
                    numpy_gradient, op='mean'
                )
                synchronized_tensor = keras.ops.convert_to_tensor(synchronized_numpy)
                synchronized_gradients = [synchronized_tensor for _ in range(self.world_size)]
            
                logger.info(f"REAL AllReduce completed using {type(self.distributed_backend).__name__}")
                return synchronized_gradients
                
            except Exception as e:
                logger.warning(f"Real distributed AllReduce failed: {e}. Falling back to simulation.")        
        if not gradients:
            return []
            
        keras_gradients = [keras.ops.convert_to_tensor(grad) for grad in gradients]
        stacked_gradients = keras.ops.stack(keras_gradients, axis=0)
        
        total_gradients = keras.ops.sum(stacked_gradients, axis=0)
        mean_grad = total_gradients / self.world_size
        
        synchronized_gradients = [keras.ops.copy(mean_grad) for _ in range(self.world_size)]
        
        logger.info(f"SIMULATION AllReduce completed for gradients with shape {keras.ops.shape(mean_grad)}")
        return synchronized_gradients
    
    def get_weights(self):
        weights = []
        for opt in self.shard_optimizers:
            weights.extend(opt.get_weights())
        return weights
    
    def set_weights(self, weights):
        weights_per_shard = len(weights) // self.world_size
        for i, opt in enumerate(self.shard_optimizers):
            start_idx = i * weights_per_shard
            end_idx = start_idx + weights_per_shard
            shard_weights = weights[start_idx:end_idx]
            opt.set_weights(shard_weights)
    
    def get_memory_usage(self):
        try:
            def _bytes_for_var(var) -> int:
                try:
                    shape = getattr(var, 'shape', ())
                    if hasattr(shape, 'as_list'):
                        shape = tuple(d if d is not None else 0 for d in shape.as_list())
                    else:
                        shape = tuple(int(d) for d in shape) if shape else ()
                    numel = int(np.prod(shape)) if len(shape) > 0 else 1
                    itemsize = 4
                    if hasattr(var, 'dtype'):
                        dt = var.dtype
                        try:
                            if hasattr(dt, 'as_numpy_dtype'):
                                itemsize = np.dtype(dt.as_numpy_dtype).itemsize
                            else:
                                itemsize = np.dtype(dt).itemsize
                        except Exception:
                            itemsize = 4
                    return numel * itemsize
                except Exception:
                    return 4

            unsharded_total_bytes = 0
            if hasattr(self.base_optimizer, 'variables'):
                for var in self.base_optimizer.variables:
                    unsharded_total_bytes += _bytes_for_var(var)

            if not hasattr(self, 'sharded_states') or not self.sharded_states:
                total_memory_bytes = unsharded_total_bytes * max(self.world_size, 1)
                sharded_memory_bytes = total_memory_bytes

                total_memory_mb = total_memory_bytes / (1024 * 1024)
                return {
                    'sharding_enabled': False,
                    'world_size': self.world_size,
                    'total_memory': f"{total_memory_mb:.2f} MB",
                    'sharded_memory': f"{total_memory_mb:.2f} MB",
                    'memory_savings': '0.00%',
                    'total_memory_bytes': total_memory_bytes,
                    'sharded_memory_bytes': sharded_memory_bytes
                }

            total_memory_bytes = 0
            sharded_memory_bytes = 0
            
            for state_name, state_info in self.sharded_states.items():
                if isinstance(state_info, dict):
                    for var_name, var_states in state_info.items():
                        if isinstance(var_states, list):
                            for shard_state in var_states:
                                if hasattr(shard_state, 'numel'):
                                    shard_memory = shard_state.numel() * shard_state.element_size()
                                    total_memory_bytes += shard_memory
                                    sharded_memory_bytes += shard_memory
                                elif hasattr(shard_state, 'nbytes'):
                                    shard_memory = shard_state.nbytes
                                    total_memory_bytes += shard_memory
                                    sharded_memory_bytes += shard_memory
                                elif hasattr(shard_state, 'shape'):
                                    shard_memory = np.prod(shard_state.shape) * 4
                                    total_memory_bytes += shard_memory
                                    sharded_memory_bytes += shard_memory
                else:
                    if isinstance(state_info, list):
                        for shard_state in state_info:
                            if hasattr(shard_state, 'numel'):
                                shard_memory = shard_state.numel() * shard_state.element_size()
                                total_memory_bytes += shard_memory
                                sharded_memory_bytes += shard_memory
                            elif hasattr(shard_state, 'nbytes'):
                                shard_memory = shard_state.nbytes
                                total_memory_bytes += shard_memory
                                sharded_memory_bytes += shard_memory
                            elif hasattr(shard_state, 'shape'):
                                shard_memory = np.prod(shard_state.shape) * 4
                                total_memory_bytes += shard_memory
                                sharded_memory_bytes += shard_memory
            
            if unsharded_total_bytes > 0:
                replicated_memory_bytes = unsharded_total_bytes * max(self.world_size, 1)
                savings_bytes = replicated_memory_bytes - sharded_memory_bytes
                savings_percentage = (savings_bytes / replicated_memory_bytes) * 100
                
                memory_savings = f"{savings_percentage:.2f}%"
            else:
                memory_savings = "0.00%"

            total_memory_mb = sharded_memory_bytes / (1024 * 1024)
            sharded_memory_mb = sharded_memory_bytes / (1024 * 1024)
            
            return {
                'sharding_enabled': True,
                'total_memory': f"{total_memory_mb:.2f} MB",
                'sharded_memory': f"{sharded_memory_mb:.2f} MB",
                'memory_savings': memory_savings,
                'world_size': self.world_size,
                'total_memory_bytes': sharded_memory_bytes,
                'sharded_memory_bytes': sharded_memory_bytes
            }
            
        except Exception as e:
            logger.warning(f"Memory calculation failed: {e}")
            return {
                'sharding_enabled': False,
                'total_memory': '0.00 MB',
                'sharded_memory': '0.00 MB',
                'memory_savings': '0.00%',
                'error': str(e)
            }

    
    def enable_optimizer_state_sharding(self):
        if not self.shard_optimizer_states:
            logger.info("Enabling optimizer state sharding...")
            self.shard_optimizer_states = True
            self._initialize_sharded_states()
            logger.info("Optimizer state sharding enabled")
        else:
            logger.info("Optimizer state sharding already enabled")
    
    def disable_optimizer_state_sharding(self):
        if self.shard_optimizer_states:
            logger.info("Disabling optimizer state sharding...")
            self.shard_optimizer_states = False
            self.sharded_states = {}
            logger.info("Optimizer state sharding disabled, using replicated states")
        else:
            logger.info("Optimizer state sharding already disabled")

    def _get_sharded_states_structure(self):
        if not hasattr(self, 'sharded_states') or not self.sharded_states:
            return {'error': 'States are not sharded or initialized.'}
        
        structure = {}
        for state_name, state_info in self.sharded_states.items():
            structure[state_name] = {}
            if isinstance(state_info, dict):
                for var_name, var_shards in state_info.items():
                    structure[state_name][var_name] = {
                        'num_shards': len(var_shards),
                        'shard_shapes': [s.shape for s in var_shards]
                    }
            else:
                 structure[state_name] = {
                    'num_shards': len(state_info),
                    'shard_shapes': [s.shape for s in state_info]
                }
        return structure


class TensorParallelOptimizer(optimizers.Optimizer):
    
    def __init__(self, base_optimizer: optimizers.Optimizer, world_size: int, distributed_backend: str = 'auto', tensor_parallel_config=None):
        lr = 0.001
        if hasattr(base_optimizer, 'learning_rate'):
            try:
                if hasattr(base_optimizer.learning_rate, 'numpy'):
                    lr = float(base_optimizer.learning_rate.numpy())
                else:
                    lr = float(base_optimizer.learning_rate)
            except:
                lr = 0.001
        
        if isinstance(base_optimizer, str):
            optimizer_name = base_optimizer
        else:
            optimizer_name = getattr(base_optimizer, 'name', 'unknown')
            
        super().__init__(
            learning_rate=lr,
            name=f"TensorParallel_{optimizer_name}"
        )
        
        if isinstance(base_optimizer, str):
            if base_optimizer.lower() == 'adam':
                actual_optimizer = optimizers.Adam(learning_rate=lr)
            elif base_optimizer.lower() == 'sgd':
                actual_optimizer = optimizers.SGD(learning_rate=lr)
            elif base_optimizer.lower() == 'rmsprop':
                actual_optimizer = optimizers.RMSprop(learning_rate=lr)
            else:
                actual_optimizer = optimizers.Adam(learning_rate=lr)
            self.base_optimizer = actual_optimizer
        else:
            self.base_optimizer = base_optimizer
            
        self.coordinated_optimizer = CoordinatedOptimizer(
            self.base_optimizer, world_size, 
            distributed_backend=distributed_backend,
            tensor_parallel_config=tensor_parallel_config 
        )
        self.world_size = world_size
        self.base_optimizer = base_optimizer
    
    def apply_gradients(self, gradients_and_vars, **kwargs):
        if isinstance(gradients_and_vars, list) and len(gradients_and_vars) > 0:
            if isinstance(gradients_and_vars[0], list):
                shard_models = kwargs.get('shard_models', [])
                return self.coordinated_optimizer.apply_gradients(gradients_and_vars, shard_models)
            else:
                return self._apply_standard_gradients(gradients_and_vars)
        else:
            return self._apply_standard_gradients(gradients_and_vars)
    
    def _apply_standard_gradients(self, gradients_and_vars):
        try:
            self.base_optimizer.apply_gradients(gradients_and_vars)
            return gradients_and_vars
        except Exception as e:
            return gradients_and_vars
           
    def get_config(self):
        config = super().get_config()
        config.update({
            'base_optimizer': self.base_optimizer.get_config(),
            'world_size': self.world_size
        })
        return config
    
    def get_weights(self):
        return self.coordinated_optimizer.get_weights()
    
    def set_weights(self, weights):
        return self.coordinated_optimizer.set_weights(weights)
    
    def update_step(self, gradient, variable, *args, **kwargs):
        if hasattr(self.base_optimizer, 'update_step'):
            try:
                return self.base_optimizer.update_step(gradient, variable, *args, **kwargs)
            except TypeError:
                return self.base_optimizer.update_step(gradient, variable)
        try:
            return super().update_step(gradient, variable, *args, **kwargs)
        except TypeError:
            return super().update_step(gradient, variable)

    def build(self, variables):
        try:
            if hasattr(self.base_optimizer, 'build'):
                self.base_optimizer.build(variables)
        except Exception:
            pass
        try:
            return super().build(variables)
        except Exception:
            return None

    def apply(self, gradients, variables=None, *args, **kwargs):
        if variables is None and gradients and isinstance(gradients[0], tuple):
            return self._apply_standard_gradients(gradients)
        if hasattr(self.base_optimizer, 'apply'):
            try:
                return self.base_optimizer.apply(gradients, variables, *args, **kwargs)
            except TypeError:
                pass
        if variables is not None:
            gv = list(zip(gradients, variables))
            return self._apply_standard_gradients(gv)
        return self._apply_standard_gradients(gradients)
    