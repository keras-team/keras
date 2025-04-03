# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import argparse
import copy
import logging
import os

import torch
from benchmark_helper import Precision, create_onnxruntime_session, prepare_environment, setup_logger
from t5_helper import PRETRAINED_MT5_MODELS, PRETRAINED_T5_MODELS, T5Helper

logger = logging.getLogger("")


def parse_arguments():
    parser = argparse.ArgumentParser()

    pretrained_models = PRETRAINED_T5_MODELS + PRETRAINED_MT5_MODELS
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        required=False,
        default=PRETRAINED_T5_MODELS[0],
        type=str,
        help="Model path, or pretrained model name in the list: " + ", ".join(pretrained_models),
    )

    parser.add_argument(
        "--model_type",
        required=False,
        type=str,
        default="t5",
        choices=["t5", "mt5"],
        help="Model type: either t5 (default) or mt5",
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    parser.add_argument(
        "--output",
        required=False,
        type=str,
        default=os.path.join(".", "onnx_models"),
        help="Output directory",
    )

    parser.add_argument(
        "-o",
        "--optimize_onnx",
        required=False,
        action="store_true",
        help="Use optimizer.py to optimize onnx model",
    )
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument("--use_gpu", required=False, action="store_true", help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=Precision,
        default=Precision.FLOAT32,
        choices=[Precision.FLOAT32, Precision.FLOAT16],
        help="Precision of model to run. fp32 for full precision, fp16 for half precision",
    )

    parser.add_argument("--verbose", required=False, action="store_true")
    parser.set_defaults(verbose=False)

    parser.add_argument("-e", "--use_external_data_format", required=False, action="store_true")
    parser.set_defaults(use_external_data_format=False)

    parser.add_argument(
        "-s",
        "--use_decoder_start_token",
        required=False,
        action="store_true",
        help="Use config.decoder_start_token_id. Otherwise, add an extra graph input for decoder_input_ids.",
    )
    parser.set_defaults(use_decoder_start_token=False)

    parser.add_argument(
        "-w",
        "--overwrite",
        required=False,
        action="store_true",
        help="overwrite existing ONNX model",
    )
    parser.set_defaults(overwrite=False)

    parser.add_argument(
        "--disable_auto_mixed_precision",
        required=False,
        action="store_true",
        help="use pure fp16 instead of mixed precision",
    )
    parser.set_defaults(disable_auto_mixed_precision=False)

    parser.add_argument(
        "--separate_encoder_and_decoder_init",
        required=False,
        action="store_true",
        help="Do not merge encode and decoder init. Output 3 instead of 2 onnx models.",
    )
    parser.set_defaults(separate_encoder_and_decoder_init=False)

    parser.add_argument(
        "--use_int64_inputs",
        required=False,
        action="store_true",
        help="Use int64 instead of int32 for input_ids, position_ids and attention_mask.",
    )
    parser.set_defaults(use_int64_inputs=False)

    parser.add_argument(
        "--state_dict_path",
        type=str,
        default="",
        help="filepath to load pre-trained model with custom state dictionary (e.g. pytorch_model.bin)",
    )

    args = parser.parse_args()

    return args


def export_onnx_models(
    model_name_or_path,
    cache_dir,
    output_dir,
    use_gpu,
    use_external_data_format,
    optimize_onnx,
    precision,
    verbose,
    use_decoder_start_token: bool = False,
    merge_encoder_and_decoder_init: bool = True,
    overwrite: bool = False,
    disable_auto_mixed_precision: bool = False,
    use_int32_inputs: bool = True,
    model_type: str = "t5",
    state_dict_path: str = "",
):
    device = torch.device("cuda:0" if use_gpu else "cpu")

    models = T5Helper.load_model(
        model_name_or_path, cache_dir, device, merge_encoder_and_decoder_init, model_type, state_dict_path
    )
    config = models["decoder"].config

    if (not use_external_data_format) and (config.num_layers > 24):
        logger.info("Try use_external_data_format when model size > 2GB")

    output_paths = []
    for name, model in models.items():
        model.to(device)
        filename_suffix = "_" + name

        onnx_path = T5Helper.get_onnx_path(
            output_dir,
            model_name_or_path,
            suffix=filename_suffix,
            new_folder=False,
        )

        if overwrite or not os.path.exists(onnx_path):
            logger.info(f"Exporting ONNX model to {onnx_path}")
            # We have to clone model before exporting onnx, otherwise verify_onnx will report large difference.
            cloned_model = copy.deepcopy(model).to(device)
            T5Helper.export_onnx(
                cloned_model,
                device,
                onnx_path,
                verbose,
                use_external_data_format,
                use_decoder_input_ids=not use_decoder_start_token,
                use_int32_inputs=use_int32_inputs,
            )
        else:
            logger.info(f"Skip exporting: existed ONNX model {onnx_path}")

        # Optimize ONNX graph. Note that we have not implemented graph optimization for T5 yet.
        if optimize_onnx or precision != Precision.FLOAT32:
            output_path = T5Helper.get_onnx_path(
                output_dir,
                model_name_or_path,
                suffix=filename_suffix + "_" + str(precision),
                new_folder=False,
            )

            if overwrite or not os.path.exists(output_path):
                logger.info(f"Optimizing model to {output_path}")
                T5Helper.optimize_onnx(
                    onnx_path,
                    output_path,
                    precision == Precision.FLOAT16,
                    config.num_heads,
                    config.hidden_size,
                    use_external_data_format,
                    auto_mixed_precision=not disable_auto_mixed_precision,
                    use_gpu=use_gpu,
                )
            else:
                logger.info(f"Skip optimizing: existed ONNX model {onnx_path}")
        else:
            output_path = onnx_path

        ort_session = create_onnxruntime_session(
            output_path,
            use_gpu=use_gpu,
            provider=["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"],
        )

        with torch.no_grad():
            max_diff = T5Helper.verify_onnx(model, ort_session, device, use_int32_inputs)
        logger.info(f"PyTorch and OnnxRuntime results max difference = {max_diff}")
        if max_diff > 1e-4:
            logger.warning("PyTorch and OnnxRuntime results are NOT close")

        output_paths.append(output_path)

    return output_paths


def main():
    args = parse_arguments()

    setup_logger(args.verbose)

    logger.info(f"Arguments:{args}")

    cache_dir = args.cache_dir
    output_dir = args.output if not args.output.endswith(".onnx") else os.path.dirname(args.output)
    prepare_environment(cache_dir, output_dir, args.use_gpu)

    if args.precision != Precision.FLOAT32:
        assert args.optimize_onnx, "fp16/int8 requires --optimize_onnx"

    if args.precision == Precision.FLOAT16:
        assert args.use_gpu, "fp16 requires --use_gpu"

    if args.optimize_onnx:
        logger.warning("Graph optimization for T5 is not implemented yet.")

    output_paths = export_onnx_models(
        args.model_name_or_path,
        cache_dir,
        output_dir,
        args.use_gpu,
        args.use_external_data_format,
        args.optimize_onnx,
        args.precision,
        args.verbose,
        args.use_decoder_start_token,
        not args.separate_encoder_and_decoder_init,
        args.overwrite,
        args.disable_auto_mixed_precision,
        not args.use_int64_inputs,
        args.model_type,
    )

    logger.info(f"Done! Outputs: {output_paths}")


if __name__ == "__main__":
    main()
