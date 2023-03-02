import tensorflow as tf
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec


def get_data_spec(batch_size):
    return (
        {
            "num_cat_feat": InputSpec(
                (batch_size,), dtype="int32", range=[0, 5]
            ),
            "string_cat_feat": InputSpec((batch_size,), dtype="string"),
            "num_feat": InputSpec((batch_size,)),
        },
        InputSpec((batch_size, 1), dtype="int32", range=[0, 2]),
    )


def get_input_preprocessor():
    dataset = tf.data.Dataset.from_tensor_slices(
        {
            "num_cat_feat": [0, 1, 2, 3, 4, 5],
            "string_cat_feat": ["zero", "one", "two", "three", "four", "five"],
            "num_feat": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
    ).batch(3)

    num_cat_feat = keras.Input(shape=(1,), name="num_cat_feat", dtype="int64")
    string_cat_feat = keras.Input(
        shape=(1,), name="string_cat_feat", dtype="string"
    )
    num_feat = keras.Input(shape=(1,), name="num_feat", dtype="float32")

    all_inputs = [
        num_cat_feat,
        string_cat_feat,
        num_feat,
    ]

    all_features = keras.layers.concatenate(
        [
            encode_categorical_feature(
                num_cat_feat, "num_cat_feat", dataset, False
            ),
            encode_categorical_feature(
                string_cat_feat, "string_cat_feat", dataset, True
            ),
            encode_numerical_feature(num_feat, "num_feat", dataset),
        ]
    )
    preprocessor = keras.Model(all_inputs, all_features)
    return preprocessor


def encode_numerical_feature(feature, name, dataset):
    normalizer = keras.layers.Normalization(mean=[1.0], variance=[2.0])
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = (
        keras.layers.StringLookup if is_string else keras.layers.IntegerLookup
    )
    lookup = lookup_class(output_mode="binary")
    feature_ds = dataset.map(lambda x: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    lookup.adapt(feature_ds)
    encoded_feature = lookup(feature)
    return encoded_feature


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    preprocessor = get_input_preprocessor()
    if include_preprocessing:
        all_inputs = preprocessor.inputs
        all_features = preprocessor.outputs[0]
    else:
        all_inputs = keras.Input(shape=preprocessor.outputs[0].shape)
        all_features = all_inputs
    x = keras.layers.Dense(32, activation="relu")(all_features)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(all_inputs, output)

    if compile:
        model.compile(
            "adam",
            "binary_crossentropy",
            metrics=["accuracy"],
            jit_compile=jit_compile,
        )
    return model


def get_custom_objects():
    return {}
