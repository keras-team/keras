workspace(name = "org_keras")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# The rules_python is used to install pip package for tensorflow, that will be
# used as python dependency.
http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
    sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

# TF package is used as dependency for protobuf and bzl files. It is NOT used
# for python dependency.
http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-2.3.0",
    sha256 = "1a6f24d9e3b1cf5cc55ecfe076d3a61516701bc045925915b26a9d39f4084c34",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v2.3.0.zip"
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(tf_repo_name="@org_tensorflow")

load("@rules_python//python:pip.bzl", "pip_install", "pip_repositories")
pip_repositories()
pip_install(
    name = "keras_deps",
    requirements = "//:requirements.txt",
)
