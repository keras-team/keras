from absl import logging

from keras.src.api_export import keras_export
from keras.src.quantizers.gptq_core import quantize_model


@keras_export(["keras.GPTQConfig", "keras.quantizers.GPTQConfig"])
class GPTQConfig:
    """Configuration class for the GPTQ algorithm.

    This class holds all the parameters needed to apply the GPTQ method
    to a model.

    Args:
        dataset: The calibration dataset. It can be an iterable that yields
            strings or pre-tokenized numerical tensors (e.g., a list of
            strings, a generator, or a NumPy array). This data is used to
            analyze the model's activations.
        tokenizer: A `keras_nlp.Tokenizer` instance (or a similar callable)
            that is used to process the `dataset` if it contains strings.
        wbits (int, optional): The number of bits to quantize weights to.
            Defaults to 4.
        nsamples (int, optional): The number of calibration data samples to
            use from the dataset. Defaults to 128.
        seqlen (int, optional): The sequence length to use for each calibration
            sample. Defaults to 512.
        percdamp (float, optional): The % of Hessian damping to use for
            stabilization during inverse calculation. Defaults to 0.01.
        group_size (int, optional): The size of weight groups to quantize
            together. A `group_size` of -1 indicates per-channel quantization.
            Defaults to 128.
        symmetric (bool, optional): If `True`, uses symmetric quantization.
            If `False`, uses asymmetric quantization. Defaults to `False`.
        act_order (bool, optional): If `True`, reorders weight columns based on
            activation magnitude, which can improve quantization accuracy.
            Defaults to `False`.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        wbits: int = 4,
        nsamples: int = 128,
        seqlen: int = 512,
        percdamp: float = 0.01,
        group_size: int = 128,
        symmetric: bool = False,
        act_order: bool = False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.nsamples = nsamples
        self.seqlen = seqlen
        self.percdamp = percdamp
        self.wbits = wbits
        self.group_size = group_size
        self.symmetric = symmetric
        self.act_order = act_order
        self.quantization_method = "gptq"

    def quantize(self, model):
        """
        Applies GPTQ quantization to the provided model using this
        configuration.
        """
        logging.info("Initiating quantization from GPTQConfig...")
        # The core logic is now delegated to gptqutils, which will handle
        # the dynamic imports and data loading.
        quantize_model(model=model, config=self)
