# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_affine_grid import (
    apply_affine_transform,
    construct_original_grid,
)


def create_affine_matrix_3d(
    angle1,
    angle2,
    offset_x,
    offset_y,
    offset_z,
    shear_x,
    shear_y,
    shear_z,
    scale_x,
    scale_y,
    scale_z,
):
    rot_x = np.stack(
        [
            np.ones_like(angle1),
            np.zeros_like(angle1),
            np.zeros_like(angle1),
            np.zeros_like(angle1),
            np.cos(angle1),
            -np.sin(angle1),
            np.zeros_like(angle1),
            np.sin(angle1),
            np.cos(angle1),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    rot_y = np.stack(
        [
            np.cos(angle2),
            np.zeros_like(angle2),
            np.sin(angle2),
            np.zeros_like(angle2),
            np.ones_like(angle2),
            np.zeros_like(angle2),
            -np.sin(angle2),
            np.zeros_like(angle2),
            np.cos(angle2),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    shear = np.stack(
        [
            np.ones_like(shear_x),
            shear_x,
            shear_y,
            shear_z,
            np.ones_like(shear_x),
            shear_x,
            shear_y,
            shear_x,
            np.ones_like(shear_x),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    scale = np.stack(
        [
            scale_x,
            np.zeros_like(scale_x),
            np.zeros_like(scale_x),
            np.zeros_like(scale_x),
            scale_y,
            np.zeros_like(scale_x),
            np.zeros_like(scale_x),
            np.zeros_like(scale_x),
            scale_z,
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    translation = np.transpose(np.array([offset_x, offset_y, offset_z])).reshape(
        -1, 1, 3
    )
    rotation_matrix = rot_y @ rot_x @ shear @ scale  # (N, 3, 3)
    rotation_matrix = np.transpose(rotation_matrix, (0, 2, 1))
    affine_matrix = np.hstack((rotation_matrix, translation))
    affine_matrix = np.transpose(affine_matrix, (0, 2, 1))
    return affine_matrix.astype(np.float32)


def create_affine_matrix_2d(
    angle1, offset_x, offset_y, shear_x, shear_y, scale_x, scale_y
):
    rot = np.stack(
        [np.cos(angle1), -np.sin(angle1), np.sin(angle1), np.cos(angle1)], axis=-1
    ).reshape(-1, 2, 2)
    shear = np.stack(
        [np.ones_like(shear_x), shear_x, shear_y, np.ones_like(shear_x)], axis=-1
    ).reshape(-1, 2, 2)
    scale = np.stack(
        [scale_x, np.zeros_like(scale_x), np.zeros_like(scale_x), scale_y], axis=-1
    ).reshape(-1, 2, 2)
    translation = np.transpose(np.array([offset_x, offset_y])).reshape(-1, 1, 2)
    rotation_matrix = rot @ shear @ scale  # (N, 3, 3)
    rotation_matrix = np.transpose(rotation_matrix, (0, 2, 1))
    affine_matrix = np.hstack((rotation_matrix, translation))
    affine_matrix = np.transpose(affine_matrix, (0, 2, 1))
    return affine_matrix.astype(np.float32)


def create_theta_2d():
    angle = np.array([np.pi / 4, np.pi / 3])
    offset_x = np.array([5.0, 2.5])
    offset_y = np.array([-3.3, 1.1])
    shear_x = np.array([-0.5, 0.5])
    shear_y = np.array([0.3, -0.3])
    scale_x = np.array([2.2, 1.1])
    scale_y = np.array([3.1, 0.9])
    theta_2d = create_affine_matrix_2d(
        angle, offset_x, offset_y, shear_x, shear_y, scale_x, scale_y
    )
    return theta_2d


def create_theta_3d():
    angle1 = np.array([np.pi / 4, np.pi / 3])
    angle2 = np.array([np.pi / 6, np.pi / 2])
    offset_x = np.array([5.0, 2.5])
    offset_y = np.array([-3.3, 1.1])
    offset_z = np.array([-1.1, 2.2])
    shear_x = np.array([-0.5, 0.5])
    shear_y = np.array([0.3, -0.3])
    shear_z = np.array([0.7, -0.2])
    scale_x = np.array([2.2, 1.1])
    scale_y = np.array([3.1, 0.9])
    scale_z = np.array([0.5, 1.5])

    theta_3d = create_affine_matrix_3d(
        angle1,
        angle2,
        offset_x,
        offset_y,
        offset_z,
        shear_x,
        shear_y,
        shear_z,
        scale_x,
        scale_y,
        scale_z,
    )
    return theta_3d


class AffineGrid(Base):
    @staticmethod
    def export_2d_no_reference_evaluator() -> None:
        theta_2d = create_theta_2d()
        N, C, H, W = len(theta_2d), 3, 5, 6
        data_size = (H, W)
        for align_corners in (0, 1):
            node = onnx.helper.make_node(
                "AffineGrid",
                inputs=["theta", "size"],
                outputs=["grid"],
                align_corners=align_corners,
            )

            original_grid = construct_original_grid(data_size, align_corners)
            grid = apply_affine_transform(theta_2d, original_grid)

            test_name = "test_affine_grid_2d"
            if align_corners == 1:
                test_name += "_align_corners"
            expect(
                node,
                inputs=[theta_2d, np.array([N, C, H, W], dtype=np.int64)],
                outputs=[grid],
                name=test_name,
            )

    @staticmethod
    def export_3d_no_reference_evaluator() -> None:
        theta_3d = create_theta_3d()
        N, C, D, H, W = len(theta_3d), 3, 4, 5, 6
        data_size = (D, H, W)
        for align_corners in (0, 1):
            node = onnx.helper.make_node(
                "AffineGrid",
                inputs=["theta", "size"],
                outputs=["grid"],
                align_corners=align_corners,
            )

            original_grid = construct_original_grid(data_size, align_corners)
            grid = apply_affine_transform(theta_3d, original_grid)

            test_name = "test_affine_grid_3d"
            if align_corners == 1:
                test_name += "_align_corners"
            expect(
                node,
                inputs=[theta_3d, np.array([N, C, D, H, W], dtype=np.int64)],
                outputs=[grid],
                name=test_name,
            )
