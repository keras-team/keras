"""Quick check of _fast_call status for custom layers."""
import os, sys
os.environ["KERAS_BACKEND"] = "torch"

# Add the benchmark dir so we can import
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
from bench_transformer import build_keras_transformer, Config

cfg = Config()
model = build_keras_transformer(cfg)
x = np.ones((cfg.batch, cfg.seq_len), dtype=np.int32)
_ = model(x)  # build

print("=== _fast_call status ===")
for layer in model._flatten_layers():
    fc = getattr(layer, '_fast_call', '?')
    has_sublayers = bool(getattr(layer, '_layers', []))
    ctx = getattr(layer, '_accepts_context_arg', '?')
    trn = getattr(layer, '_call_has_training_arg', '?')
    mask = getattr(layer, '_call_has_mask_arg', '?')
    built = getattr(layer, 'built', '?')
    print(f"  {layer.name:30s}  _fast_call={fc!s:5s}  built={built!s:5s}  "
          f"has_sublayers={has_sublayers!s:5s}  ctx={ctx!s:5s}  "
          f"training_arg={trn!s:5s}  mask_arg={mask!s:5s}")
