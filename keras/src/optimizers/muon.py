from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer



class Muon(optimizer.Optimizer):
    """Optimizer that implements the Adam algorithm.

    Adam optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments.

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    - This optimizer should not be used for the embedding layer, 
    the final fully connected layer, or any {0,1}-D parameters; 
    those should all be optimized by AdamW.
   

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
            It should be noted that lr is one-tenth when using adamw.
        adam_beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates. Defaults to
            `0.9`.
        adam_beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates. Defaults to
            `0.999`.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults
            to `1e-7`.
        output_layer_key_word:List, which stores the keyword. 
            All layers with the keyword in the path will use adamw.
            In general, the embedding layer and the final layer should use adamw.
        adam_lr_ratio:float,The ratio of the learning rate when 
                using Adam to the main learning rate
        momentum: The momentum used by the internal SGD. 
        ns_steps: The number of Newton-Schulz iterations to run. 
        nesterov: Whether to use Nesterov-style momentum in the internal SGD
        {{base_optimizer_keyword_args}}
    """

    def __init__(
        self,
        learning_rate=0.001,
        adam_beta_1=0.9,
        adam_beta_2=0.999,
        epsilon=1e-7,
        weight_decay=0.1,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="muon",
        output_layer_key_word:list = ["embedding"],
        muon_a = 3.4445,
        muon_b = -4.7750,
        muon_c =  2.0315,
        adam_lr_ratio=0.1,
        momentum=0.95,
        ns_steps=6,
        nesterov = True,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.epsilon = epsilon
        self.output_layer_key_word = output_layer_key_word
        self.muon_a = muon_a
        self.muon_b = muon_b
        self.muon_c = muon_c
        self.adam_lr_ratio = adam_lr_ratio
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.nesterov = nesterov
    def weather_use_adamw(self,variable):
        #To use it with 4D convolutional filters, 
        #it works well to just flatten their last 3 dimensions.
        #any {0,1}-D parameters should all be optimized by adam
        if not 1<len(variable.shape)<4:
            return True
        for keyword in self.output_layer_key_word:
            if keyword in variable.path:
                return True
        return False
    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
            var_list: list of model variables to build Adam variables on.
        """
        if self.built:
            return
        super().build(var_list)
        self.adam_momentums = {}
        self.adam_velocities = {}
        
        self.muon_momentums = {}
        self.muon_velocities = {}
        
        for var in var_list:
            self.adam_momentums[var.path] =\
                    self.add_variable_from_reference(
                        reference_variable=var, name="momentum"
                    )
            if self.weather_use_adamw(var):
                self.adam_velocities[var.path] =\
                    self.add_variable_from_reference(
                        reference_variable=var, name="velocity"
                    )
                    


    def update_step(self, gradient, variable, learning_rate):
        if self.weather_use_adamw(variable):
            #It should be noted that lr is one-tenth when using adamw.
            self.adamw_update_step(gradient, variable, 
                                   learning_rate*self.adam_lr_ratio)
        else:
            self.muon_update_step(gradient, variable, learning_rate)
    def muon_update_step(self, gradient, variable, lr):
        m = self.adam_momentums[variable.path]
        self.assign_add(
            m, ops.add(gradient,m*(self.momentum-1))
        )
        shape = variable.shape
        if self.nesterov:
            g = ops.add(gradient,self.momentum*m)
        else:
            g = m
        
        self.assign_sub(
            variable,
            lr*self.zeropower_via_newtonschulz5(g,self.ns_steps)*
                 max(1, shape[0]/shape[1])**0.5    
        )
        
        
        
        
        
            
            
    def adamw_update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        adam_beta_1_power = ops.power(
            ops.cast(self.adam_beta_1, variable.dtype), local_step
        )
        adam_beta_2_power = ops.power(
            ops.cast(self.adam_beta_2, variable.dtype), local_step
        )

        m = self.adam_momentums[variable.path]
        v = self.adam_velocities[variable.path]

        alpha = lr * ops.sqrt(1 - adam_beta_2_power) / (1 - adam_beta_1_power)

        self.assign_add(
            m, ops.multiply(ops.subtract(gradient, m), 1 - self.adam_beta_1)
        )
        self.assign_add(
            v,
            ops.multiply(
                ops.subtract(ops.square(gradient), v), 1 - self.adam_beta_2
            ),
        )
        self.assign_sub(
            variable,
            ops.divide(
                ops.multiply(m, alpha), ops.add(ops.sqrt(v), self.epsilon)
            ),
        )
    def transpose_last_axis(self,X):
        shape = ops.shape(X)
        temp_order = list(range(len(shape)))
        temp_order[-2] = temp_order[-1]
        temp_order[-1] = len(shape)-2
        X = ops.transpose(X,temp_order)
        return X
    def zeropower_via_newtonschulz5(self,X, steps: int) :
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
        quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
        of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
        zero even beyond the point where the iteration no longer converges all the way to one everywhere
        on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
        where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
        performance at all relative to UV^T, where USV^T = G is the SVD.
        """
        shape = ops.shape(X)
        assert len(shape) >= 2 
        
        a, b, c = self.muon_a, self.muon_b, self.muon_c
        if shape[-2] > shape[-1]:
            X = self.transpose_last_axis(X)

        # Ensure spectral norm is at most 1
        X = X / (ops.norm(X,axis=(-2, -1), keepdims=True) + 1e-7)
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A 
            X = a * X + B @ X
        
        if shape[-2] > shape[-1]:
            X = self.transpose_last_axis(X)
        return X

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "adam_beta_1": self.adam_beta_1,
                "adam_beta_2": self.adam_beta_2,
                "epsilon": self.epsilon,
                "output_layer_key_word":self.output_layer_key_word,
                "muon_a":self.muon_a,
                "muon_b":self.muon_b,
                "muon_c":self.muon_c,
                "adam_lr_ratio":self.adam_lr_ratio,
                "momentum":self.momentum,
                "ns_steps":self.ns_steps,
                "nesterov":self.nesterov
            }
        )
        return config


