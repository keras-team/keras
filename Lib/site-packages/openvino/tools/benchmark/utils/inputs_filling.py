# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import re
from typing import Dict, List
import numpy as np
from collections import defaultdict
from pathlib import Path
from importlib.util import find_spec

from openvino import Tensor, PartialShape, Type
from openvino.utils.types import get_dtype

from .constants import IMAGE_EXTENSIONS, NUMPY_EXTENSIONS, BINARY_EXTENSIONS
from .logging import logger
from .utils import AppInputInfo

if find_spec('cv2') is not None:
    try:
        import cv2
    except ImportError as ex:
        raise Exception("Failed to import opencv module. "
                        "Please try to uninstall opencv-python "
                        "and install opencv-python-headless instead.") from ex


class DataQueue:
    def __init__(self, input_data: dict, batch_sizes: list):
        self.input_data = input_data
        self.sizes_map = {}
        for name, tensors in input_data.items():
            self.sizes_map[name] = len(tensors)
        self.index_map = defaultdict.fromkeys(input_data.keys(), 0)
        self.batch_sizes = batch_sizes
        self.size = len(batch_sizes)
        self.current_group_id = 0

    def get_next_input(self):
        data = {}
        for input_name, input_tensors in self.input_data.items():
            data[input_name] = input_tensors[self.index_map[input_name]]
            self.index_map[input_name] = (self.index_map[input_name] + 1) % self.sizes_map[input_name]
        self.current_group_id = (self.current_group_id + 1) % self.size
        return data

    def get_next_batch_size(self):
        return self.batch_sizes[self.current_group_id]


def get_group_batch_sizes(app_input_info):
    batch_sizes = []
    niter = max(len(info.shapes) for info in app_input_info)
    for i in range(niter):
        batch_size = 0
        for info in app_input_info:
            batch_index = info.layout.get_index_by_name('N') if info.layout.has_name('N') else -1
            if batch_index != -1:
                shape = info.shapes[i % len(info.shapes)]
                if batch_size == 0:
                    batch_size = shape[batch_index]
                elif batch_size != shape[batch_index]:
                    raise Exception("Can't deterimine batch size: batch is different for different inputs!")
        if batch_size == 0:
            batch_size = 1
        batch_sizes.append(batch_size)
    return batch_sizes


def get_batch_sizes_per_input_map(app_input_info: List[AppInputInfo]):
    batch_sizes_map = {}
    for info in app_input_info:
        if info.layout.has_name('N'):
            if info.is_dynamic:
                batch_sizes_map[info.name] = info.getDimensionsByLayout('N')
            else:
                batch_sizes_map[info.name] = [len(info.getDimensionByLayout('N'))]
        else:
            batch_sizes_map[info.name] = [1] * len(info.shapes)
    return batch_sizes_map

def verify_objects_to_be_used(objects_to_be_used_map: Dict[str, List[str]], info: AppInputInfo, total_frames: int, input_type_name: str):
        if objects_to_be_used_map[info.name] > total_frames and objects_to_be_used_map[info.name] % total_frames != 0:
            objects_to_be_used_map[info.name] = objects_to_be_used_map[info.name] - objects_to_be_used_map[info.name] % total_frames
            logger.warning(f"Number of provided {input_type_name} for input '{info.name}' is not a multiple of the number of "
                            f"provided data shapes. Only {objects_to_be_used_map[info.name]} {input_type_name} will be processed for this input.")
        elif objects_to_be_used_map[info.name] < total_frames:
            logger.warning(f"Some {input_type_name} will be dublicated: {total_frames} is required, "
                            f"but only {objects_to_be_used_map[info.name]} were provided.")

def get_input_data(paths_to_input, app_input_info):
    image_mapping, numpy_mapping, binary_mapping = get_input_file_mappings(paths_to_input, app_input_info)

    image_sizes = get_image_sizes(app_input_info)
    batch_sizes_map = get_batch_sizes_per_input_map(app_input_info)

    images_to_be_used_map = {input_name: len(images)
                            for input_name, images in image_mapping.items()}
    numpys_to_be_used_map = {input_name: len(images)
                            for input_name, images in numpy_mapping.items()}
    binaries_to_be_used_map = {input_name: len(binaries)
                            for input_name, binaries in binary_mapping.items()}

    for info in app_input_info:
        if info.shapes:
            total_frames = np.sum(batch_sizes_map[info.name])
            if info.name in image_mapping:
                verify_objects_to_be_used(images_to_be_used_map, info, total_frames, "images")
            elif info.name in numpy_mapping:
                verify_objects_to_be_used(numpys_to_be_used_map, info, total_frames, "numpy arrays")
            elif info.name in binary_mapping:
                verify_objects_to_be_used(binaries_to_be_used_map, info, total_frames, "binaries")
            else:
                if not (info.is_image_info and len(image_sizes) == 1):
                    logger.warning(f"No input files were given for input '{info.name}'!. This input will be filled with random values!")
        else:
            if info.name in image_mapping:
                logger.info(f"Images given for input '{info.name}' will be processed with original shapes.")
            elif info.name in numpy_mapping:
                logger.info(f"Numpy arrays given for input '{info.name}' will be processed with original shapes.")
            else:
                raise Exception(f"Input {info.name} is dynamic. Provide data shapes!")

    data = {}
    for port, info in enumerate(app_input_info):
        if info.name in image_mapping:
            data[port] = get_image_tensors(image_mapping[info.name][:images_to_be_used_map[info.name]], info, batch_sizes_map[info.name])

        elif info.name in numpy_mapping:
            data[port] = get_numpy_tensors(numpy_mapping[info.name][:numpys_to_be_used_map[info.name]], info, batch_sizes_map[info.name])

        elif info.name in binary_mapping:
            data[port] = get_binary_tensors(binary_mapping[info.name][:binaries_to_be_used_map[info.name]], info, batch_sizes_map[info.name])

        elif info.is_image_info and len(image_sizes) == 1:
            image_size = image_sizes[0]
            logger.info(f"Create input tensors for input '{info.name}' with image sizes: {image_size}")
            data[port] = get_image_info_tensors(image_size, info)
        else:
            logger.info(f"Fill input '{info.name}' with random values ")
            data[port] = fill_tensors_with_random(info)

    return DataQueue(data, get_group_batch_sizes(app_input_info))


def get_image_tensors(image_paths: List[str], info: AppInputInfo, batch_sizes: List[int]) -> List[Tensor]:
    if 'cv2' not in sys.modules:
        logger.error("Loading images requires the opencv-python or opencv-python-headless package. "
                     "Please install it before continuing or run benchmark without "
                     "the -i flag to fill vectors with random data.")

    num_shapes = len(info.shapes)
    num_images = len(image_paths)

    processed_frames = 0
    widths = info.widths if info.is_dynamic else [info.width]
    heights = info.heights if info.is_dynamic else [info.height]
    process_with_original_shapes = num_shapes == 0
    tensors = []

    niter = max(num_shapes, num_images)
    for i in range(niter):
        shape = list(info.shapes[i % num_shapes]) if num_shapes else []
        dtype = get_dtype(info.element_type)
        images = np.ndarray(shape=shape, dtype=dtype)
        image_index = processed_frames
        current_batch_size = 1 if process_with_original_shapes else batch_sizes[i % num_shapes]
        for b in range(current_batch_size):
            image_index %= num_images
            image_filename = image_paths[image_index]
            logger.info(f'Prepare image {image_filename}')
            image = cv2.imread(image_filename)
            if process_with_original_shapes:
                logger.info(f'Image will be processed with original shape - {image.shape[:-1]}')
            elif info.layout.has_name('H') and info.layout.has_name('W'):
                new_im_size = (widths[i % num_shapes], heights[i % num_shapes])
                if image.shape[:-1] != new_im_size:
                    logger.warning(f"Image is resized from ({image.shape[:-1]}) to ({new_im_size})")
                    image = cv2.resize(image, new_im_size)

            model_channel = int(str(info.channels))
            image_channel = image.shape[-1]
            if model_channel == 1 and image_channel == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if model_channel == image_channel and str(info.layout) in ['[N,C,H,W]', '[C,H,W]']:
                image = image.transpose((2, 0, 1))

            if process_with_original_shapes:
                if len(info.partial_shape) == 4:
                    image = np.expand_dims(image, 0)
                p_shape = PartialShape(image.shape)
                if info.partial_shape.compatible(p_shape):
                    info.data_shapes.append(p_shape.to_shape())
                else:
                    raise Exception(f"Data shape '{str(p_shape)}' provided for input '{info.name}' "
                                    f"is not compatible with partial shape '{str(info.partial_shape)}' for this input.")
                tensors.append(Tensor(image.astype(dtype)))
            else:
                try:
                    if 3 == images[b].ndim and 1 == images[b].shape[2] and 2 == image.ndim:
                        # The model last dim has length 1, which means it takes greyscale images.
                        # Extend input image dims to match it
                        images[b] = image[:, :, None]
                    else:
                        images[b] = image
                except ValueError:
                    raise Exception(f"Image shape {image.shape} is not compatible with input shape {shape}! "
                                    f"Make sure -i parameter is valid.")
            image_index += 1
        processed_frames += current_batch_size
        if not process_with_original_shapes:
            tensors.append(Tensor(images))
    return tensors


def get_numpy_tensors(numpy_paths: List[str], info: AppInputInfo, batch_sizes: List[int]) -> List[Tensor]:

    num_shapes = len(info.shapes)
    num_arrays = len(numpy_paths)

    processed_frames = 0
    process_with_original_shapes = num_shapes == 0
    tensors = []

    niter = max(num_shapes, num_arrays)
    for i in range(niter):
        shape = list(info.shapes[i % num_shapes]) if num_shapes else []
        dtype = get_dtype(info.element_type)
        numpy_arrays = np.ndarray(shape=shape, dtype=dtype)
        numpy_index = processed_frames

        current_batch_size = 1 if process_with_original_shapes \
            else batch_sizes[i % num_shapes]

        for b in range(current_batch_size):
            numpy_index %= num_arrays
            numpy_filename: str = numpy_paths[numpy_index]
            extension = numpy_filename.lower().split('.')[-1]
            if extension == "npy":
                numpy_arr: np.ndarray = np.load(numpy_filename)

                if list(numpy_arr.shape) != shape and not process_with_original_shapes:
                    raise Exception(
                        f"Numpy array shape mismatch. File {numpy_filename} "
                        f"has shape: {numpy_arr.shape}, expected: {shape}")

                if numpy_arr.dtype != dtype:
                    raise Exception(
                        f"Numpy array in file {numpy_filename} is of "
                        f"{numpy_arr.dtype} format, which does not match "
                        f"input type {dtype}.")

                if process_with_original_shapes:
                    if len(info.partial_shape) - 1 == len(numpy_arr.shape):
                        numpy_arr = np.expand_dims(numpy_arr, 0)

                    p_shape = PartialShape(numpy_arr.shape)
                    if info.partial_shape.compatible(p_shape):
                        info.data_shapes.append(p_shape.to_shape())
                    else:
                        raise Exception(f"Data shape '{str(p_shape)}' provided for input '{info.name}' "
                                        f"is not compatible with partial shape '{str(info.partial_shape)}' for this input.")
                    tensors.append(Tensor(numpy_arr))
                else:
                    try:
                        if info.layout.has_name("N"):
                            numpy_arrays[[None] * info.layout.get_index_by_name("N") + [b]] = numpy_arr[b]
                        else:
                            numpy_arrays = numpy_arr
                    except ValueError:
                        raise Exception(f"Numpy array shape {numpy_arr.shape} is not compatible with input shape {shape}! "
                                        f"Make sure -i parameter is valid.")
            else:
                raise Exception(
                    f"Unsupported numpy file type: {extension}")
            numpy_index += 1

        processed_frames += current_batch_size
        if not process_with_original_shapes:
            tensors.append(Tensor(numpy_arrays))

    return tensors

def get_binary_tensors(binary_paths: List[str], info: AppInputInfo, batch_sizes: List[int]) -> List[Tensor]:
    num_shapes = len(info.shapes)
    num_binaries = len(binary_paths)
    niter = max(num_shapes, num_binaries)
    processed_frames = 0
    tensors = []
    for i in range(niter):
        shape_id = i % num_shapes
        dtype = np.uint8() if info.element_type.bitwidth < 8 else get_dtype(info.element_type)
        shape = list(info.shapes[shape_id])
        binaries = np.ndarray(shape=shape, dtype=dtype)
        binary_index = processed_frames
        current_batch_size = batch_sizes[shape_id]
        for b in range(current_batch_size):
            binary_index %= num_binaries
            binary_filename: str = binary_paths[binary_index]
            extension = binary_filename.lower().split('.')[-1]
            if extension == "bin":
                binary_file_bit_size = os.path.getsize(binary_filename) * 8
                blob_bit_size = info.element_type.bitwidth * int(np.prod(shape))
                if blob_bit_size != binary_file_bit_size:
                    raise Exception(
                        f"File {binary_filename} contains {binary_file_bit_size} bites but model expects {blob_bit_size}")
                from_file = np.fromfile(binary_filename, dtype)
                if info.layout.has_name("N"):
                    binaries[[None] * info.layout.get_index_by_name("N") + [b]] = from_file[b]
                else:
                    binaries = from_file
            else:
                raise Exception(
                    f"Unsupported binary file type: {extension}")

            binary_index += 1
        processed_frames += current_batch_size
        tensors.append(Tensor(binaries, shape, info.element_type))
    return tensors


def get_image_sizes(app_input_info):
    image_sizes = []
    for info in app_input_info:
        if info.is_image:
            if info.is_static:
                image_sizes.append((info.width, info.height))
            else:
                info_image_sizes = []
                for w, h in zip(info.widths, info.heights):
                    info_image_sizes.append((w, h))
                image_sizes.append(info_image_sizes)
    return image_sizes


def get_image_info_tensors(image_sizes, layer):
    im_infos = []
    for shape, image_size in zip(layer.shapes, image_sizes):
        im_info = np.ndarray(shape, dtype=get_dtype(layer.element_type))
        for b in range(shape[0]):
            for i in range(shape[1]):
                im_info[b][i] = image_size if i in [0, 1] else 1
        im_infos.append(Tensor(im_info))
    return im_infos

def get_random_4bit_tensor(shape, element_type, rs):
    pack_shape = [x for x in shape]
    pack_shape[-1] = pack_shape[-1]*element_type.bitwidth
    rand_data = (rs.uniform(0, 15, list(pack_shape)) >= 7).astype(int).flatten()
    rr = np.packbits(rand_data)
    return Tensor(rr, shape, element_type)

def fill_tensors_with_random(layer):
    is_4bit = layer.element_type.bitwidth == 4
    dtype = np.uint8 if is_4bit else get_dtype(layer.element_type)
    rand_min, rand_max = (0, 1) if dtype == bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    input_tensors = []
    for shape in layer.shapes:
        if shape:
            if is_4bit:
                ov_tensor = get_random_4bit_tensor(shape, layer.element_type, rs)
            else:
                ov_tensor = Tensor(rs.uniform(rand_min, rand_max, list(shape)).astype(dtype))
        else:
            if is_4bit:
                ov_tensor = get_random_4bit_tensor([1], layer.element_type, rs)
            else:
                ov_tensor = Tensor(np.ndarray([], dtype, np.array(rs.uniform(rand_min, rand_max)).astype(dtype)))
        input_tensors.append(ov_tensor)
    return input_tensors


def get_input_file_mappings(paths_to_inputs, app_input_info):
    image_dicts_list = []
    numpy_dicts_list = []
    binary_dicts_list = []

    for path in paths_to_inputs:
        image_dict, numpy_dict, binary_dict = parse_path(path, app_input_info)
        image_dicts_list.append(image_dict)
        numpy_dicts_list.append(numpy_dict)
        binary_dicts_list.append(binary_dict)

    def merge_dicts(dicts_list):
        merged = defaultdict(list)
        for dict in dicts_list:
            for k, v in dict.items():
                merged[k] += v
        return merged

    def remove_empty_items(dict):
        return {k: sorted(v) for k, v in dict.items() if v}

    return remove_empty_items(merge_dicts(image_dicts_list)), \
        remove_empty_items(merge_dicts(numpy_dicts_list)), \
        remove_empty_items(merge_dicts(binary_dicts_list))


def parse_path(path, app_input_info):
    """
    Parse "input_1:file1/dir1,file2/dir2,input_2:file3/dir3 or file1/dir1,file2/dir2" into three dicts,
    each containing input_name (str) as key and list of strings of binary/numpy/image filepaths as values.
    """
    input_names = list(info.name for info in app_input_info)
    input_node_names = list(info.node_name for info in app_input_info)
    parsed_names = re.findall(r"((?=[^,])(?![a-zA-Z]:\\)[\w\.]+):", path)
    wrong_names = list(name for name in parsed_names if name not in input_names + input_node_names)
    if wrong_names:
        raise Exception(
            f"Wrong input mapping! Cannot find inputs: {wrong_names}. "
            f"Available inputs: {input_names}. "
            "Please check `-i` input data"
        )
    tensor_names = [parsed_name if parsed_name in input_names else input_names[input_node_names.index(parsed_name)] for parsed_name in parsed_names]
    input_pathes = [path for path in re.split(r"(?=[^,])(?![a-zA-Z]:\\)[\w\.]+:", path) if path]
    input_path_mapping = defaultdict(list)
    # input mapping is used
    if tensor_names:
        input_path_mapping = {input_: files.strip(",").split(",") for input_, files in zip(tensor_names, input_pathes)}
    else:
        input_files = list()
        _input_pathes = input_pathes[0].strip(",").split(",")
        for _input_path in _input_pathes:
            input_path = Path(_input_path)
            if input_path.exists():
                if input_path.is_dir():
                    input_files += list(str(file_path) for file_path in input_path.iterdir())
                elif input_path.is_file():
                    input_files.append(str(input_path))
            else:
                raise Exception(f"Path '{str(input_path)}' doesn't exist \n {str(input_path)}")
        num_files, num_inputs = len(input_files), len(app_input_info)
        if num_inputs > 1:
            logger.warning(f"Model has {num_inputs} inputs. It's recommended to use name mapping to specify parameters for each input.")
        if num_files > num_inputs and num_files % num_inputs != 0:
            input_files = input_files[:num_files - num_files % num_inputs]
            logger.warning(f"Number of provided input files '{num_files}' is not a multiple of the number of "
                                   f"model inputs. Only {len(input_files)} files fill be used.")
        num_files = len(input_files)
        inputs_to_fill = list(info.name for info in app_input_info if not info.is_image_info)
        for i in range(num_files):
            input_path_mapping[inputs_to_fill[i % len(inputs_to_fill)]].append(input_files[i])

    images_mapping = defaultdict(list)
    numpy_mapping = defaultdict(list)
    binary_mapping = defaultdict(list)

    unsupported_files = list()
    for input_name, _input_pathes in input_path_mapping.items():
        for _input_path in _input_pathes:
            input_path = Path(_input_path)
            if input_path.exists():
                files = list()
                if input_path.is_dir():
                    files = input_path.iterdir()
                elif input_path.is_file():
                    files = [input_path]
                for file in files:
                    if file.suffix.lower() in IMAGE_EXTENSIONS:
                        images_mapping[input_name].append(str(file))
                    elif file.suffix.lower() in NUMPY_EXTENSIONS:
                        numpy_mapping[input_name].append(str(file))
                    elif file.suffix.lower() in BINARY_EXTENSIONS:
                        binary_mapping[input_name].append(str(file))
                    else:
                        unsupported_files.append(str(file))
            else:
                raise Exception(f"Path for input '{input_name}' doesn't exist \n {str(input_path)}")
    if unsupported_files:
        logger.warning(f"This files has unsupported extensions and will "
                        f"be ignored: {unsupported_files}.\n"
                        f"Supported extentions:\n"
                        f"Images: {IMAGE_EXTENSIONS}\n"
                        f"Binary: {BINARY_EXTENSIONS}\n"
                        f"Numpy: {NUMPY_EXTENSIONS}")
    return images_mapping, numpy_mapping, binary_mapping
