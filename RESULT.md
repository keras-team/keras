# Result

Configuration:

- MNIST
- model
  - `Dense` or `EinsumDense`
  - `BatchNormalization`
  - `ReLU`
- fine-tuning with `enable_lora(rank=2)`
- inference time: batch size=1024
  - float: `self.lora_enabled=True`
  - int8: `self.lora_enabled=False` (merged)

|backend|dtype_policy|layer|float acc.|int8 acc.|float inference time|int8 inference time|inference time ratio|
|-|-|-|-|-|-|-|-|
|tensorflow|float32|`Dense`|0.95990|0.96000|0.00395s|0.00198s|0.501|
|tensorflow|mixed_bfloat16|`Dense`|0.96110|0.96110|0.00265s|0.00200s|0.755|
|jax|float32|`Dense`|0.96130|0.96160|0.00304s|0.00132s|0.434|
|jax|mixed_bfloat16|`Dense`|0.95290|0.95300|0.00177s|0.00133s|0.751|
|tensorflow|float32|`EinsumDense`|0.95950|0.95920|0.00384s|0.00188s|0.490|
|tensorflow|mixed_bfloat16|`EinsumDense`|0.95980|0.95970|0.00258s|0.00200s|0.775|
|jax|float32|`EinsumDense`|0.96170|0.96160|0.00302s|0.00132s|0.437|
|jax|mixed_bfloat16|`EinsumDense`|0.95720|0.95680|0.00176s|0.00125s|0.710|
