# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
from pathlib import Path

import numpy as np
import torch
from float16 import float_to_float16_max_diff
from onnx_model import OnnxModel
from optimizer import optimize_model
from packaging import version
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor
from transformers import __version__ as transformers_version
from whisper_decoder import WhisperDecoder, WhisperDecoderHelper, WhisperDecoderInit
from whisper_encoder import WhisperEncoder, WhisperEncoderHelper
from whisper_encoder_decoder_init import WhisperEncoderDecoderInit, WhisperEncoderDecoderInitHelper

from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)

PRETRAINED_WHISPER_MODELS = [
    "whisper-tiny",
    "whisper-tiny.en",
    "whisper-base",
    "whisper-base.en",
    "whisper-small",
    "whisper-small.en",
    "whisper-medium",
    "whisper-medium.en",
    "whisper-large",
    "whisper-large-v2",
    "whisper-large-v3",
]


class WhisperHelper:
    @staticmethod
    def get_onnx_path(
        output_dir: str,
        model_name_or_path: str,
        suffix: str = "",
        new_folder: bool = False,
    ) -> str:
        """Build onnx path

        Args:
            output_dir (str): output directory
            model_name_or_path (str): pretrained model name, or path to the model checkpoint
            suffix (str, optional): suffix like "_encoder" or "_decoder_fp16" will be appended to file name. Defaults to None.
            new_folder (bool, optional): create a new directory for the model. Defaults to False.

        Returns:
            str: path of onnx model
        """
        model_name = model_name_or_path
        if os.path.isdir(model_name_or_path):
            model_name = Path(model_name_or_path).parts[-1]
        else:
            model_name = model_name.split("/")[-1]

        model_name += suffix

        directory = os.path.join(output_dir, model_name) if new_folder else output_dir
        return os.path.join(directory, model_name + ".onnx")

    @staticmethod
    def load_model_openai(
        model_name_or_path: str,
        cache_dir: str,
        device: torch.device,
    ) -> torch.nn.Module:
        """Load model given a pretrained name or path, then build models for ONNX conversion.

        Args:
            model_name_or_path (str): pretrained model name or path
            cache_dir (str): cache directory
            device (torch.device): device to run the model
            merge_encoder_and_decoder_init (bool, optional): Whether merge encoder and decoder initialization into one ONNX model. Defaults to True.
        Returns:
            Dict[str, torch.nn.Module]: mapping from name to modules for ONNX conversion.
        """
        from whisper import _ALIGNMENT_HEADS, _MODELS, _download
        from whisper.model import ModelDimensions, Whisper

        in_memory = False

        model_name = model_name_or_path.split("/")[-1][8:]
        checkpoint_file, alignment_heads = None, None
        if model_name in _MODELS:
            checkpoint_file = _download(_MODELS[model_name], cache_dir, in_memory)
            alignment_heads = _ALIGNMENT_HEADS[model_name]

        with open(checkpoint_file, "rb") as fp:
            checkpoint = torch.load(fp, map_location=device)
        del checkpoint_file

        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])

        if alignment_heads is not None:
            model.set_alignment_heads(alignment_heads)
        return model.to(device)

    @staticmethod
    def load_model(
        model_name_or_path: str,
        model_impl: str,
        cache_dir: str,
        device: torch.device,
        merge_encoder_and_decoder_init: bool = True,
    ) -> dict[str, torch.nn.Module]:
        """Load model given a pretrained name or path, then build models for ONNX conversion.

        Args:
            model_name_or_path (str): pretrained model name or path
            cache_dir (str): cache directory
            device (torch.device): device to run the model
            merge_encoder_and_decoder_init (bool, optional): Whether merge encoder and decoder initialization into one ONNX model. Defaults to True.
        Returns:
            Dict[str, torch.nn.Module]: mapping from name to modules for ONNX conversion.
        """
        extra_kwargs = {}
        if version.parse(transformers_version) >= version.parse("4.36.0"):
            extra_kwargs["attn_implementation"] = "eager"
        model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=cache_dir, **extra_kwargs)

        if model_impl == "openai":
            openai_model = WhisperHelper.load_model_openai(model_name_or_path, cache_dir, device)
            model_encoder, model_decoder = openai_model.encoder, openai_model.decoder
            passed_model = openai_model
        else:
            model_encoder, model_decoder = model, model
            passed_model = None

        decoder = WhisperDecoder(model_decoder, model.config, model_impl=model_impl, model=passed_model)
        decoder.eval().to(device)

        if merge_encoder_and_decoder_init:
            encoder_decoder_init = WhisperEncoderDecoderInit(
                model_encoder,
                model_decoder,
                model.config,
                decoder_start_token_id=None,
                model_impl=model_impl,
                model=passed_model,
            )
            return {"encoder_decoder_init": encoder_decoder_init, "decoder": decoder}
        else:
            encoder = WhisperEncoder(model.model.encoder, model.config)
            encoder.eval().to(device)
            decoder_init = WhisperDecoderInit(model.decoder, model.config)
            decoder_init.eval().to(device)
            return {
                "encoder": encoder,
                "decoder": decoder,
                "decoder_init": decoder_init,
            }

    @staticmethod
    def export_onnx(
        model: WhisperEncoder | WhisperDecoder | WhisperDecoderInit | WhisperEncoderDecoderInit,
        device: torch.device,
        onnx_model_path: str,
        verbose: bool = True,
        use_external_data_format: bool = False,
        use_decoder_input_ids: bool = True,
        use_int32_inputs: bool = False,
    ):
        if isinstance(model, WhisperEncoder):
            WhisperEncoderHelper.export_onnx(
                model,
                device,
                onnx_model_path,
                verbose,
                use_external_data_format,
            )
        elif isinstance(model, WhisperEncoderDecoderInit):
            WhisperEncoderDecoderInitHelper.export_onnx(
                model,
                device,
                onnx_model_path,
                use_decoder_input_ids,
                verbose,
                use_external_data_format,
                use_int32_inputs,
            )
        else:
            WhisperDecoderHelper.export_onnx(
                model,
                device,
                onnx_model_path,
                verbose,
                use_external_data_format,
                use_int32_inputs,
            )

    @staticmethod
    def auto_mixed_precision(
        onnx_model: OnnxModel,
        op_block_list: tuple[str] = (
            "SimplifiedLayerNormalization",
            "SkipSimplifiedLayerNormalization",
            "Relu",
            "Add",
        ),
    ):
        """Convert model to mixed precision.
           It detects whether original model has fp16 precision weights, and set parameters for float16 conversion automatically.
        Args:
            onnx_model (OnnxModel): optimized ONNX model
            op_block_list (List[str], optional): . Defaults to ["SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "Relu", "Add"]
        Returns:
            parameters(dict): a dictionary of parameters used in float16 conversion
        """
        op_full_set = {node.op_type for node in onnx_model.nodes()}
        fp32_op_set = set(op_block_list)
        fp16_op_set = op_full_set.difference(fp32_op_set)
        logger.info(f"fp32 op: {fp32_op_set} fp16 op: {fp16_op_set}")

        # logits is the first output
        logits_output_name = onnx_model.graph().output[0].name

        # We use the weight in last MatMul node to detect whether the model is stored with float16 weights from training.
        is_weight_fp16_precision = False
        output_name_to_node = onnx_model.output_name_to_node()
        assert logits_output_name in output_name_to_node
        node = output_name_to_node[logits_output_name]
        last_matmul_node = None
        if node.op_type == "MatMul":
            last_matmul_node = node
            logger.info(f"Found last MatMul node for logits: {node.name}")
            initializer = None
            for input in node.input:
                initializer = onnx_model.get_initializer(input)
                if initializer is not None:
                    break

            # when the max difference of value after converting float to float16 is lower than a threshold (1e-6),
            # we can deduce that the weights are stored in float16 precision.
            max_diff = float_to_float16_max_diff(initializer)
            logger.debug(f"max diff of converting weights in last MatMul node {node.name}: {max_diff}")
            is_weight_fp16_precision = max_diff < 1e-6
        else:
            logger.warning(f"Failed to find MatMul node for logits. Found {node.op_type} of node {node.name}")

        keep_io_types = []
        node_block_list = []
        if (not is_weight_fp16_precision) and (last_matmul_node is not None):
            # When original weight is float32 precision, keep logits and last MatMul in float32 could get better precision.
            keep_io_types = [logits_output_name]
            node_block_list = [last_matmul_node.name]

        parameters = {
            "keep_io_types": keep_io_types,
            "op_block_list": list(op_block_list),
            "node_block_list": node_block_list,
            "force_fp16_initializers": is_weight_fp16_precision,
        }

        logger.info(f"auto_mixed_precision parameters: {parameters}")
        onnx_model.convert_float_to_float16(use_symbolic_shape_infer=True, **parameters)

        return parameters

    @staticmethod
    def optimize_onnx(
        onnx_model_path: str,
        optimized_model_path: str,
        is_float16: bool,
        num_attention_heads: int,
        hidden_size: int,
        use_external_data_format: bool = False,
        auto_mixed_precision: bool = True,
        use_gpu: bool = False,
        provider: str = "cpu",
    ):
        """Optimize ONNX model with an option to convert it to use mixed precision."""

        from fusion_options import FusionOptions

        optimization_options = FusionOptions("bart")
        optimization_options.use_multi_head_attention = True
        optimization_options.disable_multi_head_attention_bias = provider == "rocm"

        m = optimize_model(
            onnx_model_path,
            model_type="bart",
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            opt_level=2 if not use_external_data_format else None,
            optimization_options=optimization_options,
            use_gpu=use_gpu,
            only_onnxruntime=False,
        )

        if is_float16:
            if auto_mixed_precision:
                WhisperHelper.auto_mixed_precision(m)
            else:
                m.convert_model_float32_to_float16(cast_input_output=False)

        m.save_model_to_file(optimized_model_path, use_external_data_format, all_tensors_to_one_file=True)

    @staticmethod
    def pt_transcription_for_verify_onnx(
        processor: WhisperProcessor,
        pt_model: torch.nn.Module,
        device: torch.device,
        batch_size: int = 1,
        prompt_mode: bool = False,
    ):
        # Try to import `datasets` pip package
        try:
            from datasets import load_dataset
        except Exception as e:
            logger.error(f"An error occurred while importing `datasets`: {e}", exc_info=True)  # noqa: G201
            install_cmd = "pip install datasets"
            logger.warning(f"Could not import `datasets`. Attempting to install `datasets` via `{install_cmd}`.")
            os.system(install_cmd)

        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        input_features_ = []
        if batch_size == 1:
            input_features = processor([ds[0]["audio"]["array"]], return_tensors="pt").input_features
        else:
            input_features_ = [
                processor([ds[3]["audio"]["array"]], return_tensors="pt").input_features,
                processor([ds[3]["audio"]["array"]], return_tensors="pt").input_features,
            ]
            assert len(input_features_) == batch_size
            input_features = torch.cat((input_features_[0], input_features_[1]))

        max_length, min_length, num_beams, num_return_sequences = 30, 0, 1, 1
        length_penalty, repetition_penalty = 1.0, 1.0
        inputs = {
            "input_features": input_features.to(device),
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "early_stopping": True,
            "use_cache": True,
        }

        if prompt_mode:
            prompts = ["John has doubts", "Maria has grave doubts"]
            prompt_ids = [processor.get_prompt_ids(p) for p in prompts]
            pt_transcription = []
            pt_outputs = []
            # The looping for model.generate is necessary here due to the limitation as per
            # https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.prompt_ids
            # prompt_ids input requires a tensor of rank 1
            for i in range(batch_size):
                inputs["prompt_ids"] = torch.from_numpy(prompt_ids[i])
                inputs["input_features"] = input_features_[i].to(device)
                pt_output = pt_model.generate(**inputs).detach().cpu().numpy()
                pt_outputs.append(pt_output)
                pt_transcription.append(processor.batch_decode(pt_output, skip_special_tokens=True)[0])
            inputs["input_features"] = input_features
            del inputs["prompt_ids"]
        else:
            prompt_ids = []
            pt_outputs = pt_model.generate(**inputs).detach().cpu().numpy()
            pt_transcription = [processor.batch_decode(pt_outputs, skip_special_tokens=True)[0]]
            pt_outputs = list(pt_outputs)
        del inputs["early_stopping"]
        del inputs["use_cache"]
        return inputs, pt_transcription, pt_outputs, prompt_ids

    @staticmethod
    def select_transcription_options(
        batch_size: int,
        prompt_mode: bool,
    ):
        if batch_size > 1 and prompt_mode:
            expected_transcription_no_comma_prompt1 = " John has doubts whether Sir Frederick Layton's work is really Greek after all and can discover in it but little of Rocky I"
            expected_transcription_misspelled_prompt1 = " John has doubts whether Sir Frederick Latins work is really Greek after all and can discover in it but little of Rocky I"
            expected_transcription_no_comma_prompt2 = " Maria has grave doubts whether Sir Frederick Layton's work is really Greek after all and can discover in it but little of Rocky"
            expected_transcription_misspelled_prompt2 = " Maria has grave doubts whether Sir Frederick Latins work is really Greek after all and can discover in it but little of Rocky I"
            expected_transcription_options = {
                expected_transcription_no_comma_prompt1,
                expected_transcription_no_comma_prompt2,
                expected_transcription_misspelled_prompt1,
                expected_transcription_misspelled_prompt2,
            }
        else:
            expected_transcription_no_comma = (
                " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."
            )
            expected_transcription_with_comma = (
                " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
            )
            expected_transcription_with_quote_and_comma = (
                ' "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
            )
            expected_transcription_options = {
                expected_transcription_no_comma,
                expected_transcription_with_comma,
                expected_transcription_with_quote_and_comma,
            }
        return expected_transcription_options

    @staticmethod
    def verify_onnx(
        model_name_or_path: str,
        cache_dir: str,
        ort_session: InferenceSession,
        device: torch.device,
        batch_size: int = 1,
        prompt_mode: bool = False,
    ):
        """Compare the result from PyTorch and ONNX Runtime to verify the ONNX model is good."""
        extra_kwargs = {}
        if version.parse(transformers_version) >= version.parse("4.36.0"):
            extra_kwargs["attn_implementation"] = "eager"
        pt_model = WhisperForConditionalGeneration.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, **extra_kwargs
        ).to(device)
        processor = WhisperProcessor.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        config = WhisperConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)

        inputs, pt_transcription, pt_outputs, decoder_prompt_ids = WhisperHelper.pt_transcription_for_verify_onnx(
            processor,
            pt_model,
            device,
            batch_size=batch_size,
            prompt_mode=prompt_mode,
        )

        start_id = [config.decoder_start_token_id]  # ex: [50258]
        prompt_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
        prompt_ids = [token[1] for token in prompt_ids]  # ex: [50259, 50358, 50363]
        forced_decoder_ids = start_id + prompt_ids  # ex: [50258, 50259, 50358, 50363]

        ort_names = [entry.name for entry in ort_session.get_inputs()]
        ort_dtypes = [entry.type for entry in ort_session.get_inputs()]
        ort_to_np = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
        }

        use_extra_decoding_ids = "extra_decoding_ids" in ort_names
        for name, dtype in zip(ort_names, ort_dtypes, strict=False):
            if name == "input_features":
                inputs[name] = inputs[name].detach().cpu().numpy()
            elif name == "vocab_mask":
                inputs[name] = np.ones(config.vocab_size, dtype=ort_to_np[dtype])
            elif name == "prefix_vocab_mask":
                inputs[name] = np.ones((batch_size, config.vocab_size), dtype=ort_to_np[dtype])
            elif name == "decoder_input_ids":
                if not prompt_mode:
                    raw_input_ids = [start_id] if use_extra_decoding_ids else [forced_decoder_ids]
                    inputs[name] = np.array(raw_input_ids, dtype=ort_to_np[dtype])
                else:
                    # This logic handles the scenario for when prompts are not of the same size
                    # For example if our prompt ids are [p1_id_1, p1_id_2] and [p2_id_1]
                    # The final decoder_input_ids will look as such after padding
                    # [prev_token, p1_id_1, p1_id_2, start_token, lang_token, transcribe_token]
                    # [prev_token, p2_id_1, PAD_TOKEN, start_token, lang_token, transcribe_token]
                    ort_prompts = []
                    for i in range(batch_size):
                        ort_prompts.append(decoder_prompt_ids[i].tolist())
                    max_len = max(len(p) for p in ort_prompts)
                    padded_prompts = []
                    for p in ort_prompts:
                        padded_prompt = [*p, *([config.pad_token_id] * (max_len - len(p)))]
                        padded_prompts.append(padded_prompt + forced_decoder_ids)
                    inputs[name] = np.array(padded_prompts, dtype=ort_to_np[dtype])
            elif name == "logits_processor":
                inputs[name] = np.array([1], dtype=ort_to_np[dtype])
            elif name == "cross_qk_layer_head":
                inputs[name] = np.array([[0, 0]], dtype=ort_to_np[dtype])
            elif name == "extra_decoding_ids":
                inputs[name] = np.repeat(np.array([prompt_ids], dtype=ort_to_np[dtype]), batch_size, 0)
            elif name == "temperature":
                inputs[name] = np.array([1.0], dtype=ort_to_np[dtype])
            else:
                inputs[name] = np.array([inputs[name]], dtype=ort_to_np[dtype])
        ort_outputs = ort_session.run(None, inputs)[0][:, 0, :]
        ort_transcription = processor.batch_decode(ort_outputs, skip_special_tokens=True)
        expected_transcription_options = WhisperHelper.select_transcription_options(batch_size, prompt_mode)

        parity = 1
        for i in range(batch_size):
            parity *= (
                pt_transcription[i] in expected_transcription_options
                and ort_transcription[i] in expected_transcription_options
            )
        max_diff = 0

        if not parity:
            for i in range(batch_size):
                if pt_outputs[i].shape != ort_outputs[i].shape:
                    diff = pt_outputs[i] - ort_outputs[i][:, : len(pt_outputs[i])]
                else:
                    diff = pt_outputs[i] - ort_outputs[i]
                max_diff_i = max(diff.min(), diff.max(), key=abs)
                max_diff = max(max_diff, max_diff_i)

        if max_diff != 0:
            logger.warning(f"PyTorch outputs: {pt_transcription}")
            logger.warning(f"ONNX Runtime outputs: {ort_transcription}")

        return max_diff
