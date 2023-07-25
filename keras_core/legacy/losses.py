from keras_core.api_export import keras_core_export


@keras_core_export("keras_core._legacy.losses.Reduction")
class Reduction:
    AUTO = "auto"
    NONE = "none"
    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"

    @classmethod
    def all(cls):
        return (cls.AUTO, cls.NONE, cls.SUM, cls.SUM_OVER_BATCH_SIZE)

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError(
                f'Invalid Reduction Key: {key}. Expected keys are "{cls.all()}"'
            )
