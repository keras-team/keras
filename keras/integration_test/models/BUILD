# Description:
#   Contains a collection of diverse Keras models to be used for integration tests.

package(
    default_visibility = [
        "//keras:friends",
    ],
    licenses = ["notice"],
)

py_library(
    name = "models",
    srcs = [
        "__init__.py",
        "bert.py",
        "ctc_speech_rnn.py",
        "dcgan.py",
        "edge_case_model.py",
        "efficientnet_v2.py",
        "input_spec.py",
        "low_level_model.py",
        "mini_unet.py",
        "mini_xception.py",
        "retinanet.py",
        "structured_data_classification.py",
        "text_classification.py",
        "timeseries_forecasting.py",
        "translation.py",
        "vae.py",
    ],
    srcs_version = "PY3",
    deps = ["//:expect_tensorflow_installed"],
)
