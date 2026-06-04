## 2025-02-27 - Explicitly disable pickle in np.load
**Vulnerability:** `np.load` was called without explicitly specifying `allow_pickle=False` in `keras/src/saving/saving_lib.py`.
**Learning:** While modern NumPy versions default to `allow_pickle=False`, it is a good security practice to explicitly declare it to prevent insecure deserialization, which could lead to arbitrary code execution if a maliciously crafted `.npz` file is loaded.
**Prevention:** Always explicitly use `allow_pickle=False` when calling `np.load` unless pickling is strictly required and the source is trusted.
