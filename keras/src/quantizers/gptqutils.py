import io
import logging
import random
import tarfile
import time

import numpy as np
import requests
from datasets import load_dataset

from keras.src import ops
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.layers import Embedding

from .gptq import GPTQ
from .quant import Quantizer


def get_dataloader(tokenizer, seqlen, dataset, nsamples=128, seed=0):
    """
    Prepares and chunks the calibration dataloader, repeating short datasets.
    """
    all_tokens = []

    # --- Step 1: Unify all input types into a single list of tokens ---
    if isinstance(dataset, str):
        logging.info(f"Loading '{dataset}' dataset from Hub...")
        if dataset == "wikitext2":
            d_name, d_config = "wikitext", "wikitext-2-raw-v1"
        elif dataset == "ptb":
            url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
            try:
                # 1. Download the archive into memory
                response = requests.get(url)
                response.raise_for_status()

                # 2. Extract only the test file from the in-memory archive
                with tarfile.open(
                    fileobj=io.BytesIO(response.content), mode="r:gz"
                ) as tar:
                    train_path = "./simple-examples/data/ptb.train.txt"
                    test_bytes = tar.extractfile(train_path).read()

                # 3. Decode the bytes and join into a single string
                test_lines = test_bytes.decode("utf-8").strip().split("\n")
                full_text = "\n\n".join(test_lines)
                all_tokens = tokenizer.tokenize(full_text)
                logging.info(
                    "✅ Successfully processed PTB training data for"
                    "calibration."
                )

                # 2. Perform sampling and chunking directly inside this block
                all_tokens = np.array(all_tokens, dtype=np.int32)
                required_tokens = nsamples * seqlen
                if len(all_tokens) < required_tokens:
                    logging.info(
                        f"Warning: PTB dataset is too short ({len(all_tokens)}"
                        "tokens). Repeating data."
                    )
                    repeats = -(-required_tokens // len(all_tokens))
                    all_tokens = np.tile(all_tokens, repeats)

                calibration_samples = []
                for _ in range(nsamples):
                    start_index = random.randint(
                        0, len(all_tokens) - seqlen - 1
                    )
                    end_index = start_index + seqlen
                    sample = all_tokens[start_index:end_index]
                    calibration_samples.append(ops.reshape(sample, (1, seqlen)))

                final_array = ops.stack(calibration_samples, axis=0)

                # 3. Return the correctly shaped array, isolating the logic
                return ops.convert_to_numpy(final_array)

            except Exception as e:
                logging.info(f"Failed to download or process PTB data: {e!r}")
                raise e
        elif dataset == "c4":
            logging.info(
                "   -> Using memory-efficient streaming strategy for C4."
            )
            streaming_dataset = load_dataset(
                "allenai/c4", "en", split="train", streaming=True
            )
            dataset_head = streaming_dataset.take(nsamples * 5)

            samples = []
            docs_for_sampling = list(dataset_head)

            for _ in range(nsamples):
                while True:
                    doc = random.choice(docs_for_sampling)
                    try:
                        # Call the tokenizer layer directly (the KerasNLP way)
                        # and squeeze the output to a 1D array.
                        tokenized_doc = np.squeeze(tokenizer(doc["text"]))
                        if len(tokenized_doc) >= seqlen:
                            break
                    except Exception:
                        docs_for_sampling.remove(doc)
                        if not docs_for_sampling:
                            raise ValueError(
                                "Could not find enough valid documents"
                                "in the C4 sample."
                            )
                        continue

                j = random.randint(0, len(tokenized_doc) - seqlen - 1)
                sample_slice = tokenized_doc[j : j + seqlen]
                samples.append(np.reshape(sample_slice, (1, seqlen)))

            return np.array(samples, dtype=np.int32)
        else:
            logging.info(f"Warning: No specific alias found for '{dataset}'.")
            logging.info(
                f"Attempting to load '{dataset}' directly with its "
                "default configuration."
            )
            d_name = dataset
            d_config = None  # Use the default configuration for the dataset

        # Default to "text" for wikitext2 and other datasets
        text_column = "text"

        raw_dataset = load_dataset(d_name, d_config, split="train")
        text_list = [d[text_column] for d in raw_dataset]
        full_text = "\n\n".join(text_list)
        all_tokens = tokenizer.tokenize(full_text)

    else:
        logging.info("\n==> Using pre-made dataset/generator...")
        dataset_list = list(dataset)

        if not dataset_list:
            raise ValueError("Provided dataset is empty.")

        if isinstance(dataset_list[0], str):
            logging.info("   (Dataset contains strings, tokenizing now...)")
            full_text = "\n\n".join(dataset_list)
            all_tokens = tokenizer.tokenize(full_text)
        else:
            logging.info("   (Dataset is pre-tokenized, concatenating...)")
            concatenated_tokens = ops.concatenate(
                [ops.reshape(s, [-1]) for s in dataset_list], axis=0
            )
            all_tokens = ops.convert_to_numpy(concatenated_tokens)

    all_tokens = np.array(all_tokens, dtype=np.int32)

    # --- Step 2: Repeat data if it's too short ---
    required_tokens = nsamples * seqlen
    if len(all_tokens) < required_tokens:
        logging.info(
            f"Warning: Dataset is too short ({len(all_tokens)} tokens)."
            " Repeating data to generate {nsamples} samples."
        )
        repeats = -(-required_tokens // len(all_tokens))  # Ceiling division
        all_tokens = np.tile(all_tokens, repeats)

    # --- Step 3: Chunk the token list into samples ---
    # utils.set_random_seed(seed)

    calibration_samples = []
    for _ in range(nsamples):
        # Generate a random starting index
        start_index = random.randint(0, len(all_tokens) - seqlen - 1)
        end_index = start_index + seqlen
        sample = all_tokens[start_index:end_index]
        calibration_samples.append(ops.reshape(sample, (1, seqlen)))

    final_array = ops.stack(calibration_samples, axis=0)
    return ops.convert_to_numpy(final_array)


def _find_layers_recursive(layer, prefix, found_layers):
    """
    Recursively search for Dense and EinsumDense layers and record them.
    """
    for sub_layer in layer._layers:
        # Construct a unique name for the layer based on its hierarchy
        layer_name = f"{prefix}.{sub_layer.name}"
        if isinstance(sub_layer, (Dense, EinsumDense)):
            found_layers[layer_name] = sub_layer

        # Recurse into nested layers that are not the target types
        elif hasattr(sub_layer, "_layers") and sub_layer._layers:
            _find_layers_recursive(sub_layer, layer_name, found_layers)


def find_layers_in_block(block):
    """
    A pluggable, generic function to find all Dense and EinsumDense layers
    within any transformer block by using a recursive search.
    """
    found_layers = {}
    # Start the recursive search from the block itself
    _find_layers_recursive(block, "block", found_layers)
    return found_layers


def apply_gptq_layerwise(
    model,
    dataloader,
    nsamples,
    percdamp,
    groupsize,
    symmetric,
    act_order,
    wbits,
):
    """
    Performs sequential, model-agnostic quantization by dynamically finding
    layers and capturing their inputs via hooks.
    """
    logging.info("Starting model quantization...")
    embedding_layer = None
    transformer_blocks = []
    if hasattr(model, "backbone"):
        logging.info("   -> Detected KerasNLP model structure.")
        backbone = model.backbone
        transformer_blocks = backbone.transformer_layers
        # Find the embedding layer by checking for common names or by type.
        if hasattr(backbone, "token_embedding"):
            embedding_layer = backbone.token_embedding
        elif hasattr(backbone, "embedding"):
            embedding_layer = backbone.embedding
        else:
            raise ValueError(
                "Could not automatically find an embedding layer in the model."
            )

    else:
        logging.info("   -> Detected custom model structure.")
        for layer in model.layers:
            # The first Embedding layer found is assumed to be the main one.
            if isinstance(layer, Embedding) and embedding_layer is None:
                embedding_layer = layer
            # A "block" is a container-like layer with its own sub-layers
            # that we can quantize. This is a heuristic that works for the
            # test.
            elif hasattr(layer, "_layers") and layer._layers:
                transformer_blocks.append(layer)

    if embedding_layer is None:
        raise ValueError(
            "Could not automatically find an embedding layer in the model."
        )
    if not transformer_blocks:
        raise ValueError(
            "Could not automatically find any transformer-like blocks to "
            "quantize."
        )

    # Initial inputs are the outputs of the token embedding layer
    inputs = [
        embedding_layer(ops.convert_to_tensor(batch, dtype="int32"))
        for batch in dataloader
    ]

    for i, block in enumerate(transformer_blocks):
        logging.info(f"\n--- Quantizing Block {i} ---")

        sub_layers_map = find_layers_in_block(block)
        if not sub_layers_map:
            logging.info(
                f"  No Dense or EinsumDense layers found in block {i}. "
                "Skipping."
            )
        else:
            logging.info(f"  Found layers: {list(sub_layers_map.keys())}")
            gptq_objects = {
                name: GPTQ(layer) for name, layer in sub_layers_map.items()
            }

            captured_inputs = {name: [] for name in sub_layers_map.keys()}
            original_calls = {}

            def create_hook(name, original_call_func):
                """A factory for creating a hook to capture layer inputs."""

                def hook(*args, **kwargs):
                    if args:
                        inp = args[0]
                    else:
                        inp = kwargs["inputs"]
                    captured_inputs[name].append(inp)
                    return original_call_func(*args, **kwargs)

                return hook

            try:
                for name, layer in sub_layers_map.items():
                    original_call = layer.call
                    original_calls[name] = original_call
                    layer.call = create_hook(name, original_call)

                logging.info(f"Capturing activations for block {i}...")
                for j in range(nsamples):
                    current_input = inputs[j]
                    if len(current_input.shape) == 2:
                        current_input = ops.expand_dims(current_input, axis=0)
                    _ = block(current_input)

            finally:
                for name, layer in sub_layers_map.items():
                    if name in original_calls:
                        layer.call = original_calls[name]

            logging.info(f"Building Hessians for block {i}...")
            for name, gptq_object in gptq_objects.items():
                layer_inputs = ops.concatenate(captured_inputs[name], axis=0)

                # Explicitly reshape the input tensor to be 2D, with the second
                # dimension matching the number of input features expected by
                # the layer's kernel.
                # This correctly handles inputs of any dimensionality
                # (e.g., 3D or 4D).
                num_features = gptq_object.rows
                inp_reshaped = ops.reshape(layer_inputs, (-1, num_features))
                gptq_object.update_hessian_with_batch(inp_reshaped)

            quantizer = Quantizer()
            quantizer.configure(
                wbits, perchannel=True, sym=symmetric, groupsize=groupsize
            )
            for name, gptq_object in gptq_objects.items():
                logging.info(f"  Quantizing {name}...")
                gptq_object.quantizer = quantizer
                gptq_object.quantize_and_correct_block(
                    percdamp=percdamp, groupsize=groupsize, actorder=act_order
                )
                gptq_object.free()

            del gptq_objects, captured_inputs, original_calls

        if i < len(transformer_blocks) - 1:
            logging.info(f"Generating inputs for block {i + 1}...")
            next_block_inputs = []
            for j in range(nsamples):
                current_input = inputs[j]
                if len(current_input.shape) == 2:
                    current_input = ops.expand_dims(current_input, axis=0)
                output = block(current_input)[0]
                next_block_inputs.append(output)
            inputs = next_block_inputs

    logging.info("\nQuantization process complete.")


def quantize_model(model, config):
    """
    Top-level function to quantize a Keras model using GPTQ.
    """
    logging.info("Starting GPTQ quantization process...")

    # 1. Load ALL data needed from the generator/source in a single call.
    total_samples_to_request = config.nsamples
    full_dataloader = get_dataloader(
        config.tokenizer,
        config.seqlen,
        config.dataset,
        nsamples=total_samples_to_request,
    )

    # 2. Split the materialized data. This works because full_dataloader
    # is now a NumPy array, which can be sliced and reused.
    calibration_dataloader = full_dataloader[: config.nsamples]

    tick = time.time()
    apply_gptq_layerwise(
        model,
        calibration_dataloader,  # Use the calibration slice
        len(calibration_dataloader),  # Use the actual number of samples
        config.percdamp,
        config.groupsize,
        config.symmetric,
        config.act_order,
        config.wbits,
    )
    logging.info(f"Total quantization time: {time.time() - tick:.2f} seconds")

    return
