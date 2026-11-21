## Code Review Guardrails (High-Signal & Anti-False-Positive Rules)

When performing code reviews on pull requests in `keras-team/keras`, strictly adhere to these top-level rules before applying any guideline below:

1. **Check End-to-End Consistency Without Critiquing Untouched Code**:
   - **DO check**: When a PR updates a contract (such as changing `compute_output_spec` dtype/shape promotion, decoupling a component from an optional dependency, or adding an argument to `Layer.add_weight`), verify that sibling methods, helpers, and public aliases (`_<op>()` fallback, `call()`, `Layer.add_variable`, lifecycle hooks) stay synchronized, and flag genuine spelling typos in newly added identifiers/docstrings/tests.
   - **Do NOT flag**: Untouched context lines, moved/renamed code, or pre-existing logic adjacent to a `+`/`-` line. Never widen `start_line..end_line` across a diff hunk just to critique pre-existing code when the PR only made a narrow fix, rename, or migration nearby.
2. **Verify Imports & Base Classes Before Claiming `NameError` or `AttributeError`**: Large files like `keras/src/ops/numpy.py` define imports thousands of lines above the diff, and unchanged `*_test.py` files plus `keras/src/testing/test_case.py` and `test_utils.py` are omitted from auxiliary `<FILES>` context (`keras.src.testing.TestCase` defines `assertAllClose`, `assertNotAllClose`, `assertAllEqual`, `assertSparse`, `run_layer_test`, and `get_temp_dir`). **Do NOT flag** unverified `NameError` or missing-method/attribute claims unless verified against full file imports and base classes.
3. **Catch Real Bugs, Not Speculative Defensive Boilerplate**: Check for uninitialized/`None` attributes dereferenced across control-flow paths, `issubclass(x, ...)` without `inspect.isclass(x)` when `x` can be a function/callable, and reading private `_*` attributes on user-supplied objects instead of public attributes or `getattr`. **Do NOT flag** hypothetical `None` returns from standard constructors/factories, extra runtime checks contradicting a hook's documented signature, state rollback on objects discarded upon an unhandled exception, `RuntimeWarning` wrappers in tests (`pyproject.toml` ignores `RuntimeWarning`), auto-filling mandatory user arguments, expanding an API's contract beyond its docstring, English i18n string nags, or micro-optimizations/caching outside per-step hot paths (`Layer.__call__`, `convert_to_tensor`, dtype resolution) such as trace-time helpers, one-time setup, host data loading, or standalone benchmark scripts.
4. **Skip Checks Enforced by CI**: Pre-commit and CI run `ruff check` (`E`, `F`, `I`, `S101`), `ruff format --check`, `shell/api_gen.sh`, and `zizmor`. **Do NOT flag** formatting, line length, import sorting, `assert` (`S101`), or unused imports (`F401`).
5. **Respect Existing Repo Patterns**: Before suggesting review comments (like asking users to add boilerplate or specific patterns), actively check for existing design patterns across the repository. Do not suggest adding useless code or structures that contradict or fall outside the established Keras repo coding style.

---

# Keras API design guidelines

These guidelines apply when a PR introduces or modifies a public `@keras_export` API (**Do NOT flag** argument counts or names dictated by NumPy/JAX/PyTorch/ONNX specs or existing base-class signatures).

## Seek to minimize cognitive load for our users.

Always seek to minimize the cognitive load imposed on our users in the course of using our APIs.

At a high level:

- **Minimize the actions & choices required from the user.** Make sure default values for arguments are sensible and reflect best practices (so that users usually wouldn't have to manually configure these). Don't expose options that are not important or do not match real use cases, "just in case".
- **Design simple and consistent workflows that reflect simple and consistent mental models.**

Here are a few practical rules:

- **No API should deal with internal implementation details.** An API is a language for our users to talk about the problem they care about -- and they don't care about our internal hacks. For instance, an option like `use_locking` in an optimizer should be avoided. If an argument requires users to understand the implementation (not just what the code is supposed to implement, like SGD in this case), then the argument should not be included in the public API. **An API is all about the problem it solves, not about how the code works in the background.**
- **Introduce as few new concepts as possible.** It's not just that additional data structures require more effort in order to learn about their methods and properties, it's that they multiply the number of **mental models** that are necessary to grok your API. Ideally, you should only need **a single universal mental model around which everything is organized** (in Keras, that's the `Layer`). Definitely avoid having more than 2 or 3 mental models underlying the workflows you design. Likewise, avoid having concepts that are mostly overlapping but subtly different, since the difference will be difficult to convey clearly and will confuse our users (like, say, `Network` and `Model` -- this is why we don't export `Network` as a public API).
- **Objects that do interchangeable things should have identical or very close APIs.** In particular they should have the same positional arguments. For example, it should be possible to swap one optimizer for another in user code (when leaving all arguments to their default value) without editing the arguments.
- **If you find yourself proposing a signature with more than 6-7 arguments, consider whether all of these arguments are useful.** How many people and use cases would be affected if you removed one argument? How much would they be affected -- would they be able to easily extend the API (e.g. via subclassing) to support their use case without that built-in argument? Could this API be broken up into smaller, modular objects?
- **Best-practices should come baked into your API.** The simplest way to use your API (leaving all arguments to their default value, using the most obvious tool for the task, etc) should be as close as possible to the best way of solving the problem. In particular, all arguments that can be given a default value should be given a default value, and that default should match the most common use case.
- **Plain Python types are preferable to custom types.** Use tuples, strings, ints... A custom type requires more knowledge and effort on the part of the user (e.g. `TensorShape`, which is also breaking established conventions of scientific Python). **When using enums, make sure that their values are strings**, so as to make it possible for users to pass plain strings (example: `data_format="channels_last"`, `padding="valid"`).
- **Explicit, single-level configuration arguments are preferable to nested, hidden configuration arguments.** Avoid something like: `MyLayer(hyperparameter_dict)`, instead use `MyLayer(units, activation=None, ...)`.

In particular, naming is important and difficult:

- **The meaning of an argument should be clear from its name and should not require knowledge that only the implementers have.** In particular, argument names should only involve recognized terms of art ("L1 norm" is a term of art), and should not involve implementation-related vocabulary (e.g. "fused batchnorm").
- **Avoid `OverlyLongAndSpecificNamingPatterns`.** If you find yourself with argument names with involve more than 3 subparts (e.g. "squared_operator_norm"), reconsider. Argument names should be intuitive and easy to remember.
- Avoid overly generic names (e.g. `variable`, `parameter` on high-level APIs; **Do NOT flag** `x` in `keras.ops` or math/tensor functions).
- **Make sure you are consistent in your naming choices.** Naming consistency means both **internal naming consistency** (don't call `dim` what is called `axis` in other places, don't call `ndims` what is called `ndim` elsewhere) and **consistency with established conventions for the problem domain (terms of art)**. Before settling on a name, make sure to look up existing names used by domain experts (or other APIs). In our case, argument names should be consistent with the broader scientific Python conventions, in particular NumPy.

Note that Keras uses the following naming rules:

- We use the convention `num_*` for counters, though omitting an explicit counter is nicer when there is no ambiguity (e.g. `units`, `epochs`, `filters`).
- The rank of a tensor is its `ndim`. A specific dimension index is an `axis`. The number of dimensions in a linear projection (or similar) is `units`.
- By convention Keras layers are named with nouns rather than verbs (e.g. `Normalization` and not `Normalize`, `Convolution` and not `Convolve`).
- Following Python conventions, classes use capitalized parts (e.g. `ClassName`) and functions and methods use snake case (e.g. `function_name`).
- If an argument name has a numerical suffix (e.g. `alpha_1`), we put an underscore before the suffix in snake case. The capitalized equivalent would be e.g. `Alpha1`.
- We used fully spelled-out names, e.g. `attention_scores` and not `attn_scores`. There are a couple standardized exceptions to this rule, in particular `dim` for "dimension" and `num` for "number". These are sufficiently common that they are not ambiguous to a first-time reader.

---

## Balance expressivity vs. user-friendliness.

### Simple use cases should be simple, advanced use cases should be possible:

**Don't increase the cognitive load of common use cases for the sake of niche use cases**, even minimally.
**Make sure that advanced users have a path to support their use case**, even if this path requires the users to roll out plugins or other API extensions (in particular via subclassing). **It is ok for advanced use cases not to be directly supported in the built-in API options.**

### Keep our APIs modular.

**Complex objects should be achievable by composing simple objects with few arguments, that do one thing reliably.** There is a balance to strike between having complex signatures on fewer objects, and having more objects with simpler signatures. A good API has a reasonable number of objects, with reasonably simple signatures (see also: avoiding signatures with more than 6-7 arguments).

**Things that create state or side-effects should be classes. Functions should be stateless.**
For instance, layers that create weights should not be cast as functions, since it makes the weights (and other elements of state) hard to access, impossible to update, and forces reliance on a global state capturing the side effects of layer-functions.

### APIs should be strictly compartmentalized.

For instance, the optimizer API or the layers API should not contain arguments for configuring distributed training. That should go into the distribution API.

---

## Don't neglect error messages, docstrings, and documentation.

Documentation and error messages are an integral part of the API. Good docs and helpful error messages are key to a delightful user experience.

- **Catch user errors early and anticipate common mistakes.** Do user input validation as soon as possible. Actively keep track of common mistakes that people make (by screening GitHub and StackOverflow), and either solve them by simplifying our API, adding targeted error messages for these mistakes, or having a "solutions to common issues" page in our docs. Consider adding automated fallback behaviors (e.g. casting a wrongly-typed input) instead of raising errors, when applicable. Be nice to our users.
- **Provide detailed feedback messages upon user error.** To validate user-facing input, always use `ValueError` (not `AttributeError`/`TypeError` or silent fallback). Error messages should be contextual, informative, and actionable. Every error message that transparently provides the user with the solution to their problem means one less support ticket, multiplied by how many times users run into the same issue. A good error message should answer:
    - What happened, in what context? (Include all relevant hyperparameters, e.g., `strides`, `padding`, `pool_size`, not just the input shape. Always include the *received* values like `original_shape` or `output_mode`).
    - What did the software expect? (Explicitly mention the expected vs. actual path or value for complex models).
    - How can the user fix it?
- **Use plain Python representations in error messages.** Shapes and sizes should be represented as Python tuples (e.g., `(2, 2)`) rather than raw NumPy array representations (e.g., `[2 2]`). Use `.name` attributes for objects instead of raw object strings. (**Do NOT** nitpick wording or demand a rigid `"Received: arg=..."` format when a `ValueError` message already clearly states the invalid input, **Do NOT** ask to wrap `TensorShape` in `tuple()`, and **Do NOT** demand extra user-input validation inside internal backend fast paths.)
- **A docstring should answer the question: what is this about, and why & how should I use it?** It should assume as little context as possible, and it shouldn't mention specialized terms without first introducing them (for example, "num_blocks: Number of blocks in the kernel" is not a good argument description if this is the first time you mention "blocks" in your docstring).
- **Show, don't tell: your documentation should not talk about how the software works, it should show how to use it.** Show code examples for end-to-end workflows; show code examples for each and every common use case and key feature of your API. **All public `@keras_export` docstrings should include code examples** (**Do NOT flag** missing `Examples:` on internal helpers, backend functions, private methods, or `Operation` subclasses).

Note that Keras uses the following rules for writing docstrings:

- For class docstrings, document arguments in an `Args:` section in the class docstring, not in `__init__`.
    - When a user creates a class, they are not calling the `MyLayer.__init__()` method as if it were a regular method, they are calling `MyLayer`. We don't want to generate documentation for the `__init__()` method as a standalone method that needs to be called directly, that would be confusing. We also don't need `__init__()` docstrings that always start with "Initializes a MyLayer class.", which is useless information. Leaving `__init__()` without a docstring is the best practice.
    - If constructor arguments are documented in `__init__`, it forces us to programmatically copy the `__init__` docstring when generating docs and concatenate it to the class docstring. This means that the `Args` section becomes the last thing in the docstring, which is bad.
- The order of information in a class docstring should be:
    - One-line description of the class, that gives initial context to the user. e.g. `Applies Dropout to the input.` Make sure the one-line description is useful. No `Intantiates an ObscureName class instance.`
    - Paragraph(s) of more detailed information that tells the user what the object is for and when they need to use it. e.g. `The Dropout layer randomly sets input units to 0 with a frequency of "rate" at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by "1/(1 - rate)" such that the sum over all inputs is unchanged. [...]`
    - If there is a reference paper, cite it here.
    - `Args` section.
    - If it's a layer that has arguments in `call`, the `Call arguments` section.
    - If it's a `Layer`, `Input shape` and `Output shape` sections.
    - Example(s).
    - Lastly, addendum. Information that isn't very important and that most users don't need, but that should be documented somewhere.
        - e.g. warnings about edge cases or compatibility issues.
        - e.g. pointers to further guides and tutorials.

---

## Implementation Best Practices (Multi-Backend & Symbolic Tensors)

Keras supports multiple backends (JAX, TensorFlow, PyTorch) and uses symbolic execution. Follow these patterns to ensure compatibility and robustness:

### Multi-Backend Compatibility

- **Prefer `convert_to_tensor` over backend-specific methods**: Use `ops.convert_to_tensor(x)` (or `backend.ops.convert_to_tensor(x)` inside `keras/src/ops/`, or the backend's own `convert_to_tensor`) instead of direct calls like `torch.as_tensor(x)` on user inputs. This ensures proper handling of various input types, including Keras objects such as variables, and consistent detection of the dtype. Outside `keras/src/ops/`, **both** `ops.*` and `backend.ops.*` are valid—**Do NOT** ask authors to switch one to the other, and **Do NOT flag** internal backend constant constructors like `torch.tensor(..., dtype=..., device=...)`.
- **Backend-Agnostic Shape Handling**: In shared code (`keras/src/ops/`, layers, metrics) and `keras/src/backend/tensorflow/**`, prefer using `ops.shape(inputs)` (or `backend.ops.shape(inputs)` inside `keras/src/ops/`, `tf.shape(inputs)` in `backend/tensorflow/`) over `.shape` when an axis length is used in runtime tensor computation, padding, or slicing (`reshape`, `broadcast_to`, `split`, `conv` padding), because `x.shape[i]` is `None` on symbolic TensorFlow tensors (note TensorFlow's `ops.shape(x)` keeps static dims as Python ints and only returns tensors for unknown dims, so **Do NOT flag** `ops.shape(x)[axis]` as losing static shapes, and never branch with Python `if`/`else` on a TensorFlow dynamic tensor dim from `ops.shape(x)[i]`).
- **Support Dynamic Dimensions**:
    - Use `len(shape)` or `ops.ndim(x)` to access shapes in a backend-agnostic way.
    - Use `isinstance(d, int)` to detect that a dimension from `ops.shape(x)` or a layer `input_shape` tuple is static and not dynamic. `None` is not the only representation for dynamic dimensions.
    - Use plain Python operators (e.g. `+`, `-`, `*`, `//`, `builtins.abs`) to perform math on the dimensions of a shape, not Keras ops. This works seamlessly for static dimensions, symbolic dimensions (JAX, Torch) and tensor dimensions (TensorFlow). For instance, `math.prod(shape)` is the correct way to determine the size of an array.
    - **Do NOT flag `x.shape`, `x.ndim`, or `is not None` when**: (1) inside `keras/src/backend/{jax,torch,numpy}/**` (where `.shape` dimensions are concrete integers and JAX ops like `jnp.repeat` require static integer sizes—never claim `.shape` arithmetic or comparisons fail under tracing/export there), (2) checking rank (`len(x.shape)` or `x.ndim`) or reading `KerasTensor.shape` inside `compute_output_spec`, `compute_output_shape`, or `build(self, input_shape)`, or (3) guarding an optional static bounds check (`size = x.shape[axis]; if size is not None:`).

### Backend-Agnostic Implementation for New Ops

Applies **only** when a PR adds a new op or adds a backend-agnostic fallback `_<op>()` to an existing op under `keras/src/ops/`. Every op must have
an implementation written purely with other Keras ops, in addition to any
backend-specific fast path.

Flag the following in the diff:

- **No agnostic path**: the op is only implemented in `keras/src/backend/<backend>/ops/`
  and the `keras/src/ops/` function just forwards to backend-specific implementation.
  The op then breaks on any backend that does not implement it.
- **Logic not centralized**: the implementation is inlined in the public
  function or in `Operation.call()`, and the other one duplicates it. Both must
  delegate to a single private module-level `_my_op()` helper that owns the
  dispatch and the fallback.
- **Only one path tested**: the test is not parameterized with
  `@parameterized.named_parameters(named_product(BACKEND_AGNOSTIC_OPS))`.
- **Leaked test flag**: a test calls `_set_use_backend_agnostic_ops(...)`
  without immediately registering
  `self.addCleanup(backend.config._set_use_backend_agnostic_ops, False)`.
  A failing assertion then silently changes the behavior of every subsequent
  test in the process. Also flag new tests that reset the flag in a
  `try`/`finally` block and suggest `self.addCleanup` instead.
- **Public `ops` API in the fallback**: inside `keras/src/ops/`, the fallback
  must call backend ops directly — `backend.ops.<op>` for core ops (e.g.
  `backend.ops.cast`, `backend.ops.convert_to_tensor`, `backend.ops.shape` — note `backend.ops.cast` accepts Python scalars too) and `backend.ops.<module>.<op>` otherwise (e.g.
  `backend.ops.numpy.sqrt`, `backend.ops.numpy.where`, `backend.ops.numpy.equal`), or the dependency's `_<dep>()` helper if it has
  one (e.g. `_divide_no_nan`). Flag calls to the public `ops.<op>` / `keras.ops` API. Never suggest
  replacing `backend.ops.*` with `ops.*`, and never suggest non-existent flat names like `backend.ops.divide_no_nan`, `backend.ops.equal`, or `backend.convert_to_numpy`. Both `config.<fn>()` (`from keras.src.backend import config`, including `config._use_backend_agnostic_ops()`) and `backend.<fn>()` (`backend.backend()`, `backend.floatx()`, `backend.epsilon()`) are valid; `standardize_dtype`, `standardize_shape`, `is_int_dtype`, and `is_float_dtype` live in `keras.src.backend.common` (not `keras.src.backend`).
- **Do NOT flag on existing-op fallbacks**: when adding a fallback `_<op>()` to an **existing** op, do NOT ask to add new test methods, new dtype cases, or extra `Operation()(x)` assertions not already present in the existing test.

When flagging, cite "Backend-agnostic implementation for new ops" in
`CONTRIBUTING.md` and point to `vsplit`/`_vsplit` in `keras/src/ops/numpy.py`
as the reference implementation.

### Optimization & Numeric Stability

- **Division by Zero**: Use `ops.divide_no_nan` (or `_divide_no_nan` / `backend.ops.numpy.divide_no_nan` inside `keras/src/ops/`) for mask weight calculations or any situation where a zero divisor is possible.
- **Arithmetic Masking**: Use `ops.where(mask, x, 0)` or `backend.ops.numpy.where(mask, x, 0)` instead of multiplication `x * mask` to clear values outside of a boolean mask to save memory on intermediary values (**Do NOT flag** `x * mask` for continuous weights, and never simplify the gradient-safe "double `where`" pattern `safe_x2 = where(equal(x2, 0), 1, x2)` into a single `where`, which causes `0 * inf = NaN` gradients).

### API Design & Validation

- **Early Validation**: Perform axis canonicalization (using `canonicalize_axis` or `canonicalize_axes`) early to catch errors: in `compute_output_spec` for symbolic inputs and in the backend implementation of the op for eager inputs, if the backend native op doesn't validate the axis (**Do NOT** demand `canonicalize_axis` inside a backend op when the native op or tuple indexing already validates negative axes). Validate axis uniqueness *after* canonicalization. In `keras/src/backend/numpy/`, normalize `list` `axis` to `tuple` (`np.var`/`np.squeeze` raise `TypeError` on a list); PyTorch ops (`torch.sum`, `torch.mean`, `torch.var`, `torch.squeeze`) accept a list `dim`, so **Do NOT flag** list `dim` in `keras/src/backend/torch/`. Conversely, in `Layer` classes, normalize static axis/reduction attributes (`TrackedList` -> `tuple`) **once in `__init__` or `build()`**, never via per-call `list(...)`/`tuple(...)` conversions inside `Layer.call()`.

### Testing

- **Maximize Test Coverage**: Use specific test marks (like `@pytest.mark.requires_trainable_backend`) only when strictly necessary. Ensure training-related components are tested with a full `model.fit()` loop.
- **Verify Error Clarity**: Use `self.assertRaisesRegex` to verify that the error message contains the expected guidance (**Do NOT** require `assertRaisesRegex` in shared multi-backend op tests when native backend error messages differ).
- **Avoid Dynamic Shapes**: In layers, metrics, and losses avoid operations that have an output with a dynamic shape where possible as they are not compilable with JAX. This includes `ops.nonzero`, `ops.unique` without a `size`.

---

## Conventions from maintainer reviews

Enforce these guidelines on newly added or modified code even when older code elsewhere in the file doesn't follow them, **but do NOT flag** untouched context lines, moved/renamed code, or pre-existing logic adjacent to a `+`/`-` line.

### Scope of a pull request

- **One concern per PR**: Flag changes unrelated to the stated goal, such as drive-by refactors, reformatting of untouched lines, files from another PR, or stray debug scripts. Ask the author to revert them or open a separate PR. Breaking changes always go in their own PR. (**Do NOT** flag a modified test in `*_test.py` as unrelated when unchanged lines of the test file are omitted from context.)
- **Fix the root cause in the right place**: A bug in one backend's implementation is fixed in that backend, not worked around in shared code. Disabling the feature is not a fix.
- **Performance changes need evidence**: Ask for measured speedups, and compile-time cost if relevant, in the PR description. Push back on micro-optimizations that make core code like `Layer.__call__` harder to read without a measurable gain.
- **Tutorials and demos go to keras-io**: Notebooks, guides and demo scripts don't belong in this repo. Ask for a Colab link in the PR description instead.

### Simplicity

- **Remove dead and redundant code**: Flag unused parameters and helpers; branches that only re-pass default values; checks the caller already did; `hasattr` or `getattr(self, "_x", None)` for attributes always set in `__init__`; overrides that repeat what the superclass already does (**Do NOT flag** dummy test subclasses overriding `super()` to test serialization/override detection); and caches without a measured benefit.
- **Reuse existing utilities**: Before accepting a new helper, check for an existing one (e.g. `canonicalize_axis`/`canonicalize_axes` instead of `axis % ndim`, `standardize_tuple`/`to_tuple_or_list`, `ops.convert_to_numpy`, the `tree` API for nested structures, `serialization_lib`). When the same logic appears twice, factor it into one helper.
- **Readable control flow**: Prefer early `return`/`continue`, named intermediate values and at most one level of nesting. Avoid unexplained arithmetic constants and expressions spread over many lines.
- **No broad `try`/`except`**: Don't catch broad exceptions or use `try`/`except` for control flow when an explicit check is possible. It hides real errors.
- **Keep hot paths cheap**: In code that runs on every op or every layer call (`convert_to_tensor`, `Layer.__call__`, dtype resolution), flag redundant conversions and copies (a no-op `np.asarray`, a `convert_to_tensor` whose result is discarded, converting the same value twice). Do cheap early-exit checks before expensive conversions.

### Imports

- **Import from `keras.src`**: Never `import keras` inside the library or its unit tests, because it can cause circular imports (**Do NOT flag** `integration_tests/**` or tests explicitly verifying public `keras.*` API exports like `ops_test.py`).
- **Imports at the top of the module**: Flag function-local imports unless they are needed to break a circular import (**Do NOT flag** `module_utils` `LazyModule` or local imports of backend/optional packages like `jax`, `torch`, `tensorflow`, `scipy`, `onnx`, `orbax`, `pil`, `pandas` in shared tests, exporters, callbacks, or adapters).

### Comments, docstrings and naming

- **Comments describe the code, not the PR**: Flag comments that mention "previously", the old behavior, the bug history, issue links, or how the bug was found. That context belongs in the commit message and can be found with `git blame`. Ask to cut long explanatory comments to one line, or remove them when the code is self-explanatory. Don't add file-level comments, and don't remove useful existing comments (**Do NOT flag** concise "why" comments explaining non-obvious workarounds or invariants, or intentional removal of obsolete comments).
- **No comments in unit tests**: Unless what the test exercises is really not obvious.
- **Docstrings match the behavior**: Flag docstrings that describe features that weren't implemented or make claims the code doesn't back up. Document new arguments the same way sibling classes do, including the default value, and update the docstring when a contract changes.
- **Accurate and consistent names**: Names must not overclaim (e.g. `fused_*` when fusion isn't guaranteed) and must reuse the term already used for the same concept nearby, internal names included (e.g. `num_processes`/`process_id`, not `world_size`/`rank`). Method names start with a verb. Keep wording neutral (no `victim`/`attacker` in tests).

### Multi-backend code and ops

- **Validation goes in the backend implementations**: Input validation, argument normalization and `convert_to_tensor` must not live only in the public function in `keras/src/ops/`. Keras internals call `backend.*` directly and skip it. Likewise, validation placed only in `compute_output_spec` is skipped for eager inputs, because `compute_output_spec` is only called on symbolic inputs. The symbolic and eager paths must agree on shapes, dtypes and errors.
- **Same behavior on every backend**: When a fix or feature lands in one backend, ask for it in all backends (including NumPy), or for a reason why not (**Do NOT flag** a bugfix in a single backend when other backends already pass the shared test). When backends disagree, JAX is the reference.
- **No backend-specific code in shared code**: Layers, losses, metrics and trainers must not branch on `backend.backend()`. Use an op or a hook that a backend can override. Never hardcode a list of backend names; use capability flags such as `backend.SUPPORTS_GRADIENT` so that pluggable backends keep working (**Do NOT flag** backend dispatch in `keras/src/backend/**`, `keras/src/export/**`, `keras/src/saving/**`, `keras/src/trainers/data_adapters/**`, `keras/src/distribute/**`, or `keras/src/callbacks/tensorboard.py`).
- **Ops run natively**: An op implementation must not forward to NumPy or SciPy. Ops must work on backend tensors, on GPU/TPU, and under compilation (**Do NOT flag** NumPy in `keras/src/backend/numpy/**`, in tests, or on static shapes and Python values).
- **Keras ops for tensor comparisons**: On tensors, use `ops.less`, `ops.greater`, `ops.equal`, etc. (or `backend.ops.numpy.<op>` inside `keras/src/ops/`) instead of `<`, `>`, `==` (**Do NOT flag** Python comparisons on scalars, `axis`, `ndim`, `len(...)`, static shape ints, dtype strings, or NumPy arrays in `*_test.py`).
- **Plain Python for shapes**: Do not use Keras ops to manipulate shapes and dimensions. Instead, use plain Python operators (`+`, `*`) and `math.prod`. This handles both ints and symbolic dimensions properly.
- **`ops.shape(x)` in backend-independent code**: Use `ops.shape(x)` instead of `x.shape` in backend-independent code when an axis length may be dynamic / symbolic (see the **Do NOT flag** cases under `Support Dynamic Dimensions` above).

### Dtypes

- **Only promote integer and bool inputs**: When an op needs a float input, cast only integer and bool dtypes (`"int" in dtype or dtype == "bool"` or `dtypes.promote_to_float_dtype(dtype)`—never write `dtype not in (...)` or `np.dtype(dtype).kind == "f"`, since `ml_dtypes.bfloat16` has `kind == "V"`, and note `ml_dtypes.finfo` supports `bfloat16`). Leave float inputs in their dtype: don't upcast `bfloat16`/`float16` to `float32`, and don't downcast `float64`. Follow the promotion pattern of similar ops. **Do NOT flag weak Python scalars or `float` in `dtypes.result_type`**: Python `int` and `float` (`0`, `1`, `2.0`, `float`) are **weak types** in `dtypes.result_type` and tensor ops (`result_type("float16", float) == "float16"`, `power(2.0, x)` keeps `bfloat16`)—never suggest replacing `float` with `"float32"` in `result_type`, and never call `convert_to_tensor` on Python `(int, float)` scalars before `result_type`.
- **Dtype tests cover the edge dtypes**: Dtype tests for new ops or dtype changes should include `bfloat16`, `float64`, integer and bool inputs, not only `FLOAT_DTYPES` (except when only adding a `_<op>()` fallback to an existing op).

### Tests

- **Tests must be able to fail**: Check values, not just shapes and types. A test that only checks `assertIsNotNone` or that the code doesn't raise is not enough. Use non-trivial inputs (not just `0`). Ask whether the new test would pass on the code before the fix. Every bug fix and every new branch needs a test that exercises it (**Do NOT flag** missing tests for docs-only, comment-only, `.gemini/`, or `.github/` changes).
- **No redundant tests**: Flag tests that duplicate existing coverage, e.g. the multi-backend tests in `keras/src/ops/*_test.py`. When an existing test misses a case, augment that test instead of adding a new one.
- **No new test files or classes**: You almost never need to add a new test file or a new test class. There should be a `_test.py` file for the module you modified (a new `foo_test.py` for a brand-new `foo.py` and separate `test_*_invalid_*` error methods are fine).
- **Extend and parameterize existing tests**: Add cases to the existing test or its parameterization (e.g. `NNOpsDtypeTest`) instead of new test functions. Use `absl.testing.parameterized` (`named_parameters`, `product`, `testing.named_product`) instead of loops, `self.subTest`, copy-pasted variants or `pytest.mark.parametrize`.
- **Shared tests stay shared**: A test with nothing backend-specific goes in the shared test module, without a backend `skipif`, even if the bug fixed was backend-specific. Shared test helpers go in `test_utils`, not in another `_test.py` module.
- **Skip properly, or not at all**: Use `@pytest.mark.skipif` on the test or class, not `self.skipTest`. Exclude a test for a backend in that backend's `excluded_tests.txt`, not an `if backend.backend() == ...` check inside a test. Don't skip a test to hide a failure on one backend; fix the backend. Don't guard dependencies listed in the requirements files with `pytest.importorskip` or `try`/`except ImportError` (except optional TensorFlow via `module_utils.tensorflow` and `.github/workflows/config/tf_free_ignored_tests.txt`).
- **Prefer real objects over mocks**: Mock only what can't be done for real. Test with real arrays, real packages and real distributions, and don't build impossible situations with mocks.
- **No leaked global state**: Restore any global config a test changes (image data format, distribution, environment variables, flags) with `self.addCleanup`. Never modify `sys.modules` or the import system, or call `os.environ.clear()`. Call `super().setUp()` first.
- **Write files only to the temp dir**: Assume the current folder is read-only. Only write files under `self.get_temp_dir()`, which automatically cleans up the directory.

### Compatibility, serialization and TensorFlow decoupling

- **No silent breaking changes**: Flag changes to public method names, exception types, default numerics, `keras_export` names or their order, and the saved-model format. Renames need a backward-compatible alias. Intentional breaking changes need a release-notes entry under "Breaking changes".
- **Serialization**: `get_config` returns what the user passed, not resolved values. Use `serialization_lib`. Save/load round trips must be idempotent.
- **TensorFlow decoupling**: Flag new `import tensorflow`, `tf.*`, or `tensorflow.python.*` usage **outside**: (1) `keras/src/backend/tensorflow/**`, (2) `tf.data` handling in `data_adapters/` and `keras/src/utils/*dataset_utils.py`, (3) `keras/src/export/{saved_model,saved_model_export_archive,tfsm_layer,litert,tf2onnx_lib,xla_saved_model_archive}.py` and `keras/src/legacy/saving/**`, and (4) TF-specific tests, including lazy `from keras.src.utils.module_utils import tensorflow as tf` in `*_test.py` (+ `.github/workflows/config/tf_free_ignored_tests.txt`). Outside these paths, the lazy `module_utils` TensorFlow import is still a TensorFlow dependency; when adding a TF-free path, ensure all methods reachable without TensorFlow guard any remaining `tf.*` calls.
