"""Targets for generating Keras API __init__.py files.

This bzl file is copied with slight modifications from
tensorflow/python/tools/api/generator/api_gen.bzl
so that we can avoid needing to depend on TF source code in Bazel build.

It should be noted that because this file is executed during the build,
and it imports TensorFlow code, that installing TensorFlow python package
is required to Bazel build Keras.
"""

def gen_api_init_files(
        name,
        output_files,
        root_init_template = None,
        srcs = [],
        api_name = "keras",
        api_version = 2,
        compat_api_versions = [],
        compat_init_templates = [],
        packages = ["keras"],
        package_deps = ["//keras:keras"],
        output_package = "keras.api",
        output_dir = "",
        root_file_name = "__init__.py"):
    """Creates API directory structure and __init__.py files.

    Creates a genrule that generates a directory structure with __init__.py
    files that import all exported modules (i.e. modules with tf_export
    decorators).

    Args:
      name: name of genrule to create.
      output_files: List of __init__.py files that should be generated.
        This list should include file name for every module exported using
        tf_export. For e.g. if an op is decorated with
        @tf_export('module1.module2', 'module3'). Then, output_files should
        include module1/module2/__init__.py and module3/__init__.py.
      root_init_template: Python init file that should be used as template for
        root __init__.py file. "# API IMPORTS PLACEHOLDER" comment inside this
        template will be replaced with root imports collected by this genrule.
      srcs: genrule sources. If passing root_init_template, the template file
        must be included in sources.
      api_name: Name of the project that you want to generate API files for
        (e.g. "tensorflow" or "estimator").
      api_version: TensorFlow API version to generate. Must be either 1 or 2.
      compat_api_versions: Older TensorFlow API versions to generate under
        compat/ directory.
      compat_init_templates: Python init file that should be used as template
        for top level __init__.py files under compat/vN directories.
        "# API IMPORTS PLACEHOLDER" comment inside this
        template will be replaced with root imports collected by this genrule.
      packages: Python packages containing the @tf_export decorators you want to
        process
      package_deps: Python library target containing your packages.
      output_package: Package where generated API will be added to.
      output_dir: Subdirectory to output API to.
        If non-empty, must end with '/'.
      root_file_name: Name of the root file with all the root imports.
    """
    root_init_template_flag = ""
    if root_init_template:
        root_init_template_flag = "--root_init_template=$(location " + root_init_template + ")"

    primary_package = packages[0]
    api_gen_binary_target = ("create_" + primary_package + "_api_%d_%s") % (api_version, name)
    native.py_binary(
        name = api_gen_binary_target,
        srcs = ["//keras/api:create_python_api_wrapper.py"],
        main = "//keras/api:create_python_api_wrapper.py",
        python_version = "PY3",
        srcs_version = "PY2AND3",
        visibility = ["//visibility:public"],
        deps = package_deps,
    )

    # Replace name of root file with root_file_name.
    output_files = [
        root_file_name if f == "__init__.py" else f
        for f in output_files
    ]
    all_output_files = ["%s%s" % (output_dir, f) for f in output_files]
    compat_api_version_flags = ""
    for compat_api_version in compat_api_versions:
        compat_api_version_flags += " --compat_apiversion=%d" % compat_api_version

    compat_init_template_flags = ""
    for compat_init_template in compat_init_templates:
        compat_init_template_flags += (
            " --compat_init_template=$(location %s)" % compat_init_template
        )

    # The Keras package within tf project is accessible via both paths below
    # Disable them for now so that we don't get SymbolExposedTwiceError
    # from create_python_api.py
    packages_to_ignore = ["tensorflow.python.keras", "tensorflow.keras"]
    native.genrule(
        name = name,
        outs = all_output_files,
        cmd = (
            "$(location :" + api_gen_binary_target + ") " +
            root_init_template_flag + " --apidir=$(@D)" + output_dir +
            " --apiname=" + api_name + " --apiversion=" + str(api_version) +
            compat_api_version_flags + " " + compat_init_template_flags +
            " --packages=" + ",".join(packages) +
            " --packages_to_ignore=" + ",".join(packages_to_ignore) +
            " --output_package=" + output_package + " $(OUTS)"
        ),
        srcs = srcs,
        exec_tools = [":" + api_gen_binary_target],
        visibility = ["//visibility:public"],
    )
