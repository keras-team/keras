"""Keras common starlark macros."""

# Macro to run Keras py_tests against pip installation.
def py_test(deps = [], data = [], kernels = [], **kwargs):
    native.py_test(
        deps = select({
            "//conditions:default": deps,
            "//keras:no_keras_py_deps": [],
        }),
        data = data + kernels,
        **kwargs
    )

# This is a trimmed down version of tf_py_test since a lot of internal
# features are just not available to OSS build, and also not applicable to Keras.
# So far xla, grpc and tfrt are ignored.
def tf_py_test(
        name,
        srcs,
        size = "medium",
        data = [],
        deps = [],
        main = None,
        args = [],
        tags = [],
        shard_count = 1,
        additional_visibility = [],
        kernels = [],
        flaky = 0,
        xla_enable_strict_auto_jit = False,
        xla_enabled = False,
        grpc_enabled = False,
        tfrt_enabled = False,
        tfrt_enabled_internal = False,
        **kwargs):
    kwargs.setdefault("python_version", "PY3")
    kwargs.setdefault("srcs_version", "PY3")
    py_test(
        name = name,
        size = size,
        srcs = srcs,
        args = args,
        data = data,
        flaky = flaky,
        kernels = kernels,
        main = main,
        shard_count = shard_count,
        tags = tags,
        deps = deps,
        **kwargs
    )

# This is a trimmed down version of cuda_py_test since a lot of internal
# features are just not available to OSS build, and also not applicable to Keras.
# So far xla, grpc and tfrt are ignored.
def cuda_py_test(
        name,
        srcs,
        size = "medium",
        data = [],
        main = None,
        args = [],
        shard_count = 1,
        kernels = [],
        tags = [],
        flaky = 0,
        xla_enable_strict_auto_jit = False,
        xla_enabled = False,
        grpc_enabled = False,
        xla_tags = [],  # additional tags for xla_gpu tests
        **kwargs):
    if main == None:
        main = name + ".py"
    for config in ["cpu", "gpu"]:
        test_name = name
        test_tags = tags
        if config == "gpu":
            test_tags = test_tags + ["requires-gpu-nvidia", "gpu"]
        if xla_enable_strict_auto_jit:
            tf_py_test(
                name = test_name + "_xla_" + config,
                size = size,
                srcs = srcs,
                args = args,
                data = data,
                flaky = flaky,
                grpc_enabled = grpc_enabled,
                kernels = kernels,
                main = main,
                shard_count = shard_count,
                tags = test_tags + xla_tags + ["xla", "manual"],
                xla_enabled = xla_enabled,
                xla_enable_strict_auto_jit = True,
                **kwargs
            )
        if config == "gpu":
            test_name += "_gpu"
        tf_py_test(
            name = test_name,
            size = size,
            srcs = srcs,
            args = args,
            data = data,
            flaky = flaky,
            grpc_enabled = grpc_enabled,
            kernels = kernels,
            main = main,
            shard_count = shard_count,
            tags = test_tags,
            xla_enabled = xla_enabled,
            xla_enable_strict_auto_jit = False,
            **kwargs
        )

def tpu_py_test(**kwargs):
    # Skip the tpu test for Keras oss.
    pass

# This is a trimmed down version of distribute_py_test since a lot of internal
# features are just not available to OSS build, and also not applicable to Keras.
# Especially the TPU tests branches are removed.
def distribute_py_test(
        name,
        srcs = [],
        size = "medium",
        deps = [],
        tags = [],
        data = [],
        main = None,
        args = [],
        tpu_args = [],
        tpu_tags = None,
        shard_count = 1,
        full_precision = False,
        xla_enable_strict_auto_jit = True,
        disable_mlir_bridge = True,
        disable_tpu_use_tfrt = None,
        **kwargs):
    # Default to PY3 since multi worker tests require PY3.
    kwargs.setdefault("python_version", "PY3")
    main = main if main else "%s.py" % name

    cuda_py_test(
        name = name,
        srcs = srcs,
        data = data,
        main = main,
        size = size,
        deps = deps,
        shard_count = shard_count,
        tags = tags,
        args = args,
        **kwargs
    )
