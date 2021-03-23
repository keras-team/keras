# Description:
#   Contains Keras protobufs

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(
    default_visibility = [
        "//keras:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_proto_library(
    name = "saved_metadata_proto_py_pb2",
    srcs = ["saved_metadata.proto"],
    deps = [":versions_proto_py_pb2"],
)

py_proto_library(
    name = "projector_config_proto_py_pb2",
    srcs = ["projector_config.proto"],
)

py_proto_library(
    name = "versions_proto_py_pb2",
    srcs = ["versions.proto"],
)
