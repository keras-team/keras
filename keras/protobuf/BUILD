# Description:
#   Contains Keras protobufs
load("@org_tensorflow//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package(
    default_visibility = [
        "//keras:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tf_proto_library(
    name = "projector_config_proto",
    srcs = ["projector_config.proto"],
    cc_api_version = 2,
)

tf_proto_library(
    name = "saved_metadata_proto",
    srcs = ["saved_metadata.proto"],
    cc_api_version = 2,
    protodeps = [":versions_proto"],
)

tf_proto_library(
    name = "versions_proto",
    srcs = ["versions.proto"],
    cc_api_version = 2,
)
