from keras.src.api_export import keras_export

# Unique source of truth for the version number.
__version__ = "3.7.0"


@keras_export("keras.version")
def version():
    return __version__
