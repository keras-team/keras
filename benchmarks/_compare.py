import json, os

combined = {}
for fname in ["bench_torch.json", "bench_jax.json"]:
    if os.path.exists(fname):
        combined.update(json.load(open(fname)))

groups = {
    "Pure JAX":     sorted(k for k in combined if k.startswith("jax ")),
    "Pure PyTorch": sorted(k for k in combined if k.startswith("torch ")),
    "Keras[torch]": sorted(k for k in combined if "keras[torch]" in k),
    "Keras[jax]":   sorted(k for k in combined if "keras[jax]" in k),
}

for name, keys in groups.items():
    if not keys:
        continue
    print("\n  " + name)
    print("  " + "-" * 64)
    for k in keys:
        print("    {:<52s}  {:8.2f} ms".format(k, combined[k]))

torch_pairs = [
    ("torch  CNN  eager",                      "keras[torch]  CNN  eager",                  "CNN eager      "),
    ("torch  CNN  compile",                    "keras[torch]  CNN  compile",                "CNN compile    "),
    ("torch  LLM  forward  eager",             "keras[torch]  LLM  forward  eager",         "LLM fwd eager  "),
    ("torch  LLM  forward  compile",           "keras[torch]  LLM  forward  compile",       "LLM fwd compile"),
    ("torch  LLM  generate 32tok  eager",      "keras[torch]  LLM  generate 32tok  eager",  "LLM gen eager  "),
    ("torch  LLM  generate 32tok  compile",    "keras[torch]  LLM  generate 32tok  compile","LLM gen compile"),
]

print("\n\n  Keras[torch] overhead vs raw PyTorch")
print("  " + "-" * 66)
print("  {:<18s}  {:>10s}  {:>14s}  {:>10s}".format("Op", "PyTorch", "Keras[torch]", "Overhead"))
print("  {:<18s}  {:>10s}  {:>14s}  {:>10s}".format("-"*18, "-"*10, "-"*14, "-"*10))
for pt_k, kt_k, label in torch_pairs:
    if pt_k in combined and kt_k in combined:
        pt_v, kt_v = combined[pt_k], combined[kt_k]
        print("  {}  {:8.2f} ms  {:12.2f} ms  {:8.2f}x".format(label, pt_v, kt_v, kt_v / pt_v))

jax_pairs = [
    ("jax  CNN  eager",                     "keras[jax]  CNN  eager",                "CNN eager      "),
    ("jax  CNN  jit",                       "keras[jax]  CNN  jit",                  "CNN jit        "),
    ("jax  LLM  forward  eager",            "keras[jax]  LLM  forward  eager",       "LLM fwd eager  "),
    ("jax  LLM  forward  jit",              "keras[jax]  LLM  forward  jit",         "LLM fwd jit    "),
    ("jax  LLM  generate 32tok  eager",     "keras[jax]  LLM  generate 32tok  eager","LLM gen eager  "),
    ("jax  LLM  generate 32tok  jit+scan",  "keras[jax]  LLM  generate 32tok  jit",  "LLM gen jit    "),
]

print("\n\n  Keras[jax] overhead vs pure JAX")
print("  " + "-" * 66)
print("  {:<18s}  {:>10s}  {:>14s}  {:>10s}".format("Op", "Pure JAX", "Keras[jax]", "Overhead"))
print("  {:<18s}  {:>10s}  {:>14s}  {:>10s}".format("-"*18, "-"*10, "-"*14, "-"*10))
for jax_k, kj_k, label in jax_pairs:
    if jax_k in combined and kj_k in combined:
        jax_v, kj_v = combined[jax_k], combined[kj_k]
        print("  {}  {:8.2f} ms  {:12.2f} ms  {:8.2f}x".format(label, jax_v, kj_v, kj_v / jax_v))
