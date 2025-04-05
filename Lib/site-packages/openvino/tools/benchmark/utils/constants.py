# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

NPU_DEVICE_NAME = 'NPU'
CPU_DEVICE_NAME = 'CPU'
GPU_DEVICE_NAME = 'GPU'
HETERO_DEVICE_NAME = 'HETERO'
MULTI_DEVICE_NAME = 'MULTI'
AUTO_DEVICE_NAME = 'AUTO'
UNKNOWN_DEVICE_TYPE = 'UNKNOWN'

XML_EXTENSION = '.xml'
BIN_EXTENSION = '.bin'
BLOB_EXTENSION = '.blob'

IMAGE_EXTENSIONS = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png',
                    '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.tiff', '.tif']
NUMPY_EXTENSIONS = ['.npy']
BINARY_EXTENSIONS = ['.bin']

DEVICE_DURATION_IN_SECS = {
    CPU_DEVICE_NAME: 60,
    GPU_DEVICE_NAME: 60,
    NPU_DEVICE_NAME: 60,
    UNKNOWN_DEVICE_TYPE: 120
}
