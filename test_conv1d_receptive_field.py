import keras
import numpy as np

print('=== 测试1: 默认情况 ===')
conv1 = keras.layers.Conv1D(filters=32, kernel_size=3)
print(f'conv1.return_receptive_field = {conv1.return_receptive_field}')
has_attr = hasattr(conv1, 'current_receptive_field')
print(f'conv1 是否有 current_receptive_field 属性: {has_attr}')

print()
print('=== 测试2: return_receptive_field=True ===')
conv2 = keras.layers.Conv1D(filters=32, kernel_size=3, return_receptive_field=True)
print(f'conv2.return_receptive_field = {conv2.return_receptive_field}')
print(f'conv2.current_receptive_field = {conv2.current_receptive_field}')

print()
print('=== 测试3: 不同参数的感受野计算 ===')

conv3 = keras.layers.Conv1D(filters=32, kernel_size=5, dilation_rate=1, return_receptive_field=True)
print(f'kernel_size=5, dilation_rate=1: 感受野 = {conv3.current_receptive_field} (预期: 5)')

conv4 = keras.layers.Conv1D(filters=32, kernel_size=3, dilation_rate=2, return_receptive_field=True)
print(f'kernel_size=3, dilation_rate=2: 感受野 = {conv4.current_receptive_field} (预期: 5)')

conv5 = keras.layers.Conv1D(filters=32, kernel_size=5, dilation_rate=3, return_receptive_field=True)
print(f'kernel_size=5, dilation_rate=3: 感受野 = {conv5.current_receptive_field} (预期: 13)')

print()
print('=== 测试4: get_config 序列化 ===')
config = conv2.get_config()
print(f'config 中是否包含 return_receptive_field: {"return_receptive_field" in config}')
print(f'config["return_receptive_field"] = {config.get("return_receptive_field")}')

print()
print('=== 测试5: 从 config 重建层 ===')
conv2_reconstructed = keras.layers.Conv1D.from_config(config)
print(f'重建层的 return_receptive_field = {conv2_reconstructed.return_receptive_field}')
print(f'重建层的 current_receptive_field = {conv2_reconstructed.current_receptive_field}')

print()
print('所有测试完成!')
