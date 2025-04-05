# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
import random
import tempfile
from pathlib import Path

import numpy
import onnx
import torch
from onnx_model import OnnxModel
from torch_onnx_export_helper import torch_onnx_export
from transformers import MT5Config, T5Config

from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)


class T5Encoder(torch.nn.Module):
    """T5 encoder outputs only the last hidden state"""

    def __init__(self, encoder, config: T5Config | MT5Config):
        super().__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask)[0]


class T5EncoderInputs:
    def __init__(self, input_ids, attention_mask):
        self.input_ids: torch.LongTensor = input_ids
        self.attention_mask: torch.LongTensor = attention_mask

    @staticmethod
    def create_dummy(
        batch_size: int, sequence_length: int, vocab_size: int, device: torch.device, use_int32_inputs: bool = False
    ):  # -> T5EncoderInputs
        """Create dummy inputs for T5 encoder.

        Args:
            batch_size (int): batch size
            sequence_length (int): sequence length
            vocab_size (int): vocabulary size
            device (torch.device): device of output tensors

        Returns:
            T5EncoderInputs: dummy inputs for encoder
        """
        dtype = torch.int32 if use_int32_inputs else torch.int64

        input_ids = torch.randint(
            low=0,
            high=vocab_size - 1,
            size=(batch_size, sequence_length),
            dtype=dtype,
            device=device,
        )

        attention_mask = torch.ones([batch_size, sequence_length], dtype=dtype, device=device)
        if sequence_length >= 2:
            for i in range(batch_size):
                padding_position = random.randint(0, sequence_length - 1)
                attention_mask[i, :padding_position] = 0
        return T5EncoderInputs(input_ids, attention_mask)

    def to_list(self) -> list:
        input_list = [v for v in [self.input_ids, self.attention_mask] if v is not None]
        return input_list


class T5EncoderHelper:
    @staticmethod
    def export_onnx(
        encoder: T5Encoder,
        device: torch.device,
        onnx_model_path: str,
        verbose: bool = True,
        use_external_data_format: bool = False,
        use_int32_inputs: bool = False,
    ):
        """Export encoder to ONNX

        Args:
            encoder (T5Encoder): encoder object
            device (torch.device): device of encoder object
            onnx_model_path (str): onnx path
            verbose (bool, optional): print verbose information. Defaults to True.
            use_external_data_format (bool, optional): use external data format or not. Defaults to False.
        """
        config = encoder.config
        encoder_inputs = T5EncoderInputs.create_dummy(
            batch_size=2,
            sequence_length=4,
            vocab_size=config.vocab_size,
            device=device,
            use_int32_inputs=use_int32_inputs,
        )

        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            temp_onnx_model_path = os.path.join(tmp_dir_name, "encoder.onnx")
            Path(temp_onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
            torch_onnx_export(
                encoder,
                args=tuple(encoder_inputs.to_list()),
                f=temp_onnx_model_path if use_external_data_format else onnx_model_path,
                export_params=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["hidden_states"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "hidden_states": {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=12,
                do_constant_folding=True,
                use_external_data_format=use_external_data_format,
                verbose=verbose,
            )

            if use_external_data_format:
                model = onnx.load_model(temp_onnx_model_path, load_external_data=True)
                OnnxModel.save(
                    model,
                    onnx_model_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                )

    @staticmethod
    def onnxruntime_inference(ort_session, inputs: T5EncoderInputs):
        """Run inference of ONNX model."""
        ort_inputs = {
            "input_ids": numpy.ascontiguousarray(inputs.input_ids.cpu().numpy()),
            "attention_mask": numpy.ascontiguousarray(inputs.attention_mask.cpu().numpy()),
        }

        return ort_session.run(None, ort_inputs)

    @staticmethod
    def verify_onnx(
        model: T5Encoder, ort_session: InferenceSession, device: torch.device, use_int32_inputs: bool = False
    ):
        """Compare the result from PyTorch and OnnxRuntime to verify the ONNX model is good."""
        inputs = T5EncoderInputs.create_dummy(
            batch_size=4,
            sequence_length=11,
            vocab_size=model.config.vocab_size,
            device=device,
            use_int32_inputs=use_int32_inputs,
        )
        input_list = inputs.to_list()
        torch_outputs = model(*input_list)

        ort_outputs = T5EncoderHelper.onnxruntime_inference(ort_session, inputs)

        max_diff = numpy.amax(numpy.abs(torch_outputs.cpu().numpy() - ort_outputs[0]))

        logger.info(f"max_diff={max_diff}")

        return max_diff
