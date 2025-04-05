# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def get_convert_model_help_specifics():
    from openvino.tools.ovc.cli_parser import CanonicalizePathCheckExistenceAction, readable_dirs_or_files_or_empty
    from openvino.tools.ovc.version import VersionChecker
    return {
        'input_model':
            {'description':
                 'Input model file(s) from PyTorch (ExportedProgram saved on a disk), '
                 'TensorFlow, ONNX, PaddlePaddle. '
                 'Use openvino.convert_model in Python to convert models from PyTorch.'
                 '',
             'action': CanonicalizePathCheckExistenceAction,
             'type': readable_dirs_or_files_or_empty,
             'aliases': {}},
        'input':
            {'description':
                 'Information of model input required for model conversion. '
                 'This is a comma separated list with optional '
                 'input names and shapes. The order of inputs '
                 'in converted model will match the order of '
                 'specified inputs. The shape is specified as comma-separated list. '
                 'Example, to set `input_1` input with shape [1,100] and `sequence_len` input '
                 'with shape [1,?]: \"input_1[1,100],sequence_len[1,?]\", where "?" is a dynamic dimension, '
                 'which means that such a dimension can be specified later in the runtime. '
                 'If the dimension is set as an integer (like 100 in [1,100]), such a dimension is not supposed '
                 'to be changed later, during a model conversion it is treated as a static value. '
                 'Example with unnamed inputs: \"[1,100],[1,?]\".'},
        'output':
            {'description':
                 'One or more comma-separated model outputs to be preserved in the converted model. '
                 'Other outputs are removed. If `output` parameter is not specified then all outputs from '
                 'the original model are preserved. '
                 'Do not add :0 to the names for TensorFlow. The order of outputs in the converted model is the '
                 'same as the order of specified names. '
                 'Example: ovc model.onnx output=out_1,out_2'},
        'extension':
            {'description':
                 'Paths or a comma-separated list of paths to libraries '
                 '(.so or .dll) with extensions.'},
        'version':
            {'action': 'version',
             # FIXME: Why the following is not accessible from arg parser?
             'version': 'OpenVINO Model Converter (ovc) {}'.format(VersionChecker().get_ie_version())},
        'verbose':
            {'description': 'Print detailed information about conversion.'}
    }
