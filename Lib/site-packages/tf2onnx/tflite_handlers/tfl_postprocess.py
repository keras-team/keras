# SPDX-License-Identifier: Apache-2.0


"""
tfl_postprocess
"""

import logging
import numpy as np
from onnx.onnx_pb import TensorProto

from tf2onnx.handler import tfl_op
from tf2onnx import utils
from tf2onnx.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement,invalid-name


@tfl_op(["TFL_TFLite_Detection_PostProcess"])
class TflDetectionPostProcess:
    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # This ops is basically NMS with a little post-processing.
        # TFLite implementation:
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/detection_postprocess.cc

        # box_encodings.shape = [batch_dim, box_num, 4]
        # class_predictions.shape = [batch_dim, box_num, num_classes(+1)]
        # anchors.shape = [box_num, 4]
        box_encodings, class_predictions, anchors = node.input

        classes_dtype = ctx.get_dtype(node.output[1])
        box_cnt_dtype = ctx.get_dtype(node.output[3])

        num_classes = node.get_attr_value('num_classes')
        max_detections = node.get_attr_value('max_detections')

        # Remove 'other' class if present.
        max_int64 = int(utils.get_max_value(np.int64))
        class_predictions = GraphBuilder(ctx).make_slice(
            {'data': class_predictions, 'starts': [-num_classes], 'ends': [max_int64], 'axes': [2]})

        scaling_vector = [node.get_attr_value(a) for a in ['y_scale', 'x_scale', 'h_scale', 'w_scale']]
        scale_const = ctx.make_const(utils.make_name('scale_const'), np.array(scaling_vector, np.float32)).output[0]

        scaled_boxes = ctx.make_node('Div', [box_encodings, scale_const]).output[0]
        anchors_yx = GraphBuilder(ctx).make_slice({'data': anchors, 'starts': [0], 'ends': [2], 'axes': [1]})
        anchors_hw = GraphBuilder(ctx).make_slice({'data': anchors, 'starts': [2], 'ends': [4], 'axes': [1]})
        boxes_yx = GraphBuilder(ctx).make_slice({'data': scaled_boxes, 'starts': [0], 'ends': [2], 'axes': [2]})
        boxes_hw = GraphBuilder(ctx).make_slice({'data': scaled_boxes, 'starts': [2], 'ends': [4], 'axes': [2]})

        scaled_boxes_yx = ctx.make_node('Mul', [boxes_yx, anchors_hw]).output[0]
        boxes_hw_exp = ctx.make_node('Exp', [boxes_hw]).output[0]
        scaled_boxes_hw = ctx.make_node('Mul', [boxes_hw_exp, anchors_hw]).output[0]
        const_half = ctx.make_const(utils.make_name('const_half'), np.array(0.5, np.float32)).output[0]
        boxes_half_hw = ctx.make_node('Mul', [scaled_boxes_hw, const_half]).output[0]
        boxes_center_yx = ctx.make_node('Add', [scaled_boxes_yx, anchors_yx]).output[0]

        boxes_lower_left = ctx.make_node('Sub', [boxes_center_yx, boxes_half_hw]).output[0]
        boxes_upper_right = ctx.make_node('Add', [boxes_center_yx, boxes_half_hw]).output[0]
        adjusted_boxes = ctx.make_node('Concat', [boxes_lower_left, boxes_upper_right], attr={'axis': 2}).output[0]

        iou_threshold = np.array(node.get_attr_value('nms_iou_threshold'), np.float32)
        iou_threshold_const = ctx.make_const(utils.make_name('iou_threshold'), iou_threshold).output[0]

        score_threshold = np.array(node.get_attr_value('nms_score_threshold'), np.float32)
        score_threshold_const = ctx.make_const(utils.make_name('score_threshold'), score_threshold).output[0]

        if node.get_attr_value('use_regular_nms', False):
            boxes_per_class = np.array(node.get_attr_value('detections_per_class', 100), np.int64)
        else:
            # When tflite uses FastNMS, detections_per_class is ignored.
            logging.warning("NMS node %s uses fast NMS. ONNX will approximate with standard NMS.", node.name)
            boxes_per_class = np.array(max_detections, np.int64)
        max_boxes_per_class_const = ctx.make_const(utils.make_name('max_boxes_per_class'), boxes_per_class).output[0]

        # scores.shape = [batch_dim, classes_num, box_num]
        scores = ctx.make_node('Transpose', [class_predictions], attr={'perm': [0, 2, 1]}).output[0]

        nms_inputs = [adjusted_boxes, scores, max_boxes_per_class_const, iou_threshold_const, score_threshold_const]
        # shape: [-1, 3], elts of format [batch_index, class_index, box_index]
        selected_indices = ctx.make_node('NonMaxSuppression', nms_inputs, attr={'center_point_box': 0},
                                         op_name_scope=node.name).output[0]

        selected_boxes_idx = GraphBuilder(ctx).make_slice(
            {'data': selected_indices, 'starts': [2], 'ends': [3], 'axes': [1]})
        selected_boxes_idx_sq = GraphBuilder(ctx).make_squeeze({'data': selected_boxes_idx, 'axes': [1]})

        selected_classes = GraphBuilder(ctx).make_slice(
            {'data': selected_indices, 'starts': [1], 'ends': [2], 'axes': [1]})
        selected_classes_sq = GraphBuilder(ctx).make_squeeze({'data': selected_classes, 'axes': [1]})

        box_and_class_idx = ctx.make_node('Concat', [selected_boxes_idx, selected_classes], attr={'axis': 1}).output[0]

        box_cnt = ctx.make_node('Shape', [selected_classes_sq]).output[0]

        adjusted_boxes_sq = GraphBuilder(ctx).make_squeeze({'data': adjusted_boxes, 'axes': [0]})
        detection_boxes = ctx.make_node('Gather', [adjusted_boxes_sq, selected_boxes_idx_sq]).output[0]
        class_predictions_sq = GraphBuilder(ctx).make_squeeze({'data': class_predictions, 'axes': [0]})
        detection_scores = ctx.make_node('GatherND', [class_predictions_sq, box_and_class_idx]).output[0]

        k_const = ctx.make_const(utils.make_name('const_k'), np.array([max_detections], np.int64)).output[0]
        if ctx.opset >= 12:
            min_k = ctx.make_node('Min', [k_const, box_cnt]).output[0]
        else:
            # Lower opsets only support Min between floats
            box_cnt_float = ctx.make_node('Cast', [box_cnt], attr={'to': TensorProto.FLOAT}).output[0]
            k_const_float = ctx.make_node('Cast', [k_const], attr={'to': TensorProto.FLOAT}).output[0]
            min_k_float = ctx.make_node('Min', [k_const_float, box_cnt_float]).output[0]
            min_k = ctx.make_node('Cast', [min_k_float], attr={'to': TensorProto.INT64}).output[0]
        min_k_cast = ctx.make_node('Cast', [min_k], attr={'to': box_cnt_dtype}).output[0]

        scores_top_k, scores_top_k_idx = ctx.make_node('TopK', [detection_scores, min_k], output_count=2).output

        scores_top_k_idx_unsq = GraphBuilder(ctx).make_unsqueeze({'data': scores_top_k_idx, 'axes': [0]})
        scores_top_k_unsq = GraphBuilder(ctx).make_unsqueeze({'data': scores_top_k, 'axes': [0]})

        selected_classes_sort = ctx.make_node('Gather', [selected_classes_sq, scores_top_k_idx_unsq]).output[0]
        classes_sort_cast = ctx.make_node('Cast', [selected_classes_sort], attr={'to': classes_dtype}).output[0]
        detection_boxes_sorted = ctx.make_node('Gather', [detection_boxes, scores_top_k_idx_unsq]).output[0]

        pad_amount = ctx.make_node('Sub', [k_const, min_k]).output[0]

        quad_zero_const = ctx.make_const(utils.make_name('quad_zero_const'), np.array([0, 0, 0, 0], np.int64)).output[0]
        duo_zero_const = ctx.make_const(utils.make_name('duo_zero_const'), np.array([0, 0], np.int64)).output[0]
        zero_const = ctx.make_const(utils.make_name('zero_const'), np.array([0], np.int64)).output[0]

        pads_3d = ctx.make_node('Concat', [quad_zero_const, pad_amount, zero_const], attr={'axis': 0}).output[0]
        pads_2d = ctx.make_node('Concat', [duo_zero_const, zero_const, pad_amount], attr={'axis': 0}).output[0]

        detection_boxes_padded = ctx.make_node('Pad', [detection_boxes_sorted, pads_3d]).output[0]
        detection_classes_padded = ctx.make_node('Pad', [classes_sort_cast, pads_2d]).output[0]
        detection_scores_padded = ctx.make_node('Pad', [scores_top_k_unsq, pads_2d]).output[0]

        ctx.replace_all_inputs(node.output[0], detection_boxes_padded)
        ctx.replace_all_inputs(node.output[1], detection_classes_padded)
        ctx.replace_all_inputs(node.output[2], detection_scores_padded)
        ctx.replace_all_inputs(node.output[3], min_k_cast)

        ctx.remove_node(node.name)
