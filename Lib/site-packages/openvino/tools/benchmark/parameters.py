# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys, argparse

from openvino.tools.benchmark.utils.utils import show_available_devices

INPUT_OUTPUT_PRECISION_CHOICES = [
    'bool', \
    'f16', 'f32', 'f64', \
    'i8', 'i16', 'i32', 'i64', \
    'u8', 'u16', 'u32', 'u64']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue

class print_help(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        show_available_devices()
        sys.exit()
class HelpFormatterWithLines(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        lines = super()._split_lines(text, width)
        lines += ['']
        if "simple JSON file" not in text:
            return lines
        lines = text.split('\n')
        return lines

def parse_args():
    parser = argparse.ArgumentParser(
        conflict_handler='resolve',  # Allow dla_benchmark for FPGA to override show_available_devices() for --help
        add_help=False, formatter_class=HelpFormatterWithLines)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action=print_help, nargs='?', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-i', '--paths_to_input', action='append', nargs='+', type=str, required=False,
                      help='Optional. '
                           'Path to a folder with images and/or binaries or to specific image or binary file.'
                           'It is also allowed to map files to model inputs: '
                           'input_1:file_1/dir1,file_2/dir2,input_4:file_4/dir4 input_2:file_3/dir3 '
                           'Currently supported data types: bin, npy. If OPENCV is enabled, this functionality'
                           'is extended with the following data types: bmp, dib, jpeg, jpg, jpe, jp2, png, pbm, '
                           'pgm, ppm, sr, ras, tiff, tif.')
    args.add_argument('-m', '--path_to_model', type=str, required=True,
                      help='Required. Path to an .xml/.onnx file with a trained model or '
                           'to a .blob file with a trained compiled model.')
    args.add_argument('-d', '--target_device', type=str, required=False, default='CPU',
                      help='Optional. Specify a target device to infer on (the list of available devices is shown below). '
                           'Default value is CPU. Use \'-d HETERO:<comma separated devices list>\' format to specify HETERO plugin. '
                           'Use \'-d MULTI:<comma separated devices list>\' format to specify MULTI plugin. '
                           'The application looks for a suitable plugin for the specified device.')
    args.add_argument('-hint', '--perf_hint', type=str, required=False, default='', choices=('throughput', 'tput', 'cumulative_throughput', 'ctput', 'latency', 'none'),
                      help='Optional. Performance hint (latency or throughput or cumulative_throughput or none). Performance hint allows the OpenVINO device to select the right model-specific settings.\n'
                            '\'throughput\': device performance mode will be set to THROUGHPUT. \n'
                            '\'cumulative_throughput\': device performance mode will be set to CUMULATIVE_THROUGHPUT. \n'
                            '\'latency\': device performance mode will be set to LATENCY. \n'
                            '\'none\': no device performance mode will be set. \n'
                            'Using explicit \'nstreams\' or other device-specific options, please set hint to \'none\'')
    args.add_argument('-niter', '--number_iterations', type=check_positive, required=False, default=None,
                      help='Optional. Number of iterations. '
                           'If not specified, the number of iterations is calculated depending on a device.')
    args.add_argument('-max_irate', '--maximum_inference_rate', type=float, required=False, default=0,
                      help='Optional. Maximum inference rate by frame per second. '
                           'If not specified, default value is 0, the inference will run at maximium rate depending on a device capabilities. '
                           'Tweaking this value allow better accuracy in power usage measurement by limiting the execution.')
    args.add_argument('-t', '--time', type=check_positive, required=False, default=None,
                      help='Optional. Time in seconds to execute topology.')

    shapes = parser.add_argument_group('Input shapes')
    shapes.add_argument('-b', '--batch_size', type=str, required=False, default='',
                        help='Optional. ' +
                             'Batch size value. ' +
                             'If not specified, the batch size value is determined from Intermediate Representation')
    shapes.add_argument('-shape', type=str, required=False, default='',
                        help='Optional. '
                             'Set shape for input. For example, "input1[1,3,224,224],input2[1,4]" or "[1,3,224,224]" in case of one input size. '
                             'This parameter affect model Parameter shape, can be dynamic. For dynamic dimesions use symbol `?`, `-1` or range `low.. up`.')
    shapes.add_argument('-data_shape', type=str, required=False, default='',
                        help='Optional. '
                             'Optional if model shapes are all static (original ones or set by -shape).'
                             'Required if at least one input shape is dynamic and input images are not provided.'
                             'Set shape for input tensors. For example, "input1[1,3,224,224][1,3,448,448],input2[1,4][1,8]" or "[1,3,224,224][1,3,448,448] in case of one input size.')
    shapes.add_argument('-layout', type=str, required=False, default='',
                        help='Optional. '
                             'Prompts how model layouts should be treated by application. '
                             'For example, "input1[NCHW],input2[NC]" or "[NCHW]" in case of one input size.')

    advs = parser.add_argument_group('Advanced options')
    advs.add_argument('-extensions', '--extensions', type=str, required=False, default=None,
                      help='Optional. Path or a comma-separated list of paths to libraries (.so or .dll) with extensions.')
    advs.add_argument('-c', '--path_to_cldnn_config', type=str, required=False,
                      help='Optional. Required for GPU custom kernels. Absolute path to an .xml file with the '
                           'kernels description.')
    advs.add_argument('-cdir', '--cache_dir', type=str, required=False, default='',
                      help="Optional. Enable model caching to specified directory")
    advs.add_argument('-lfile', '--load_from_file', required=False, nargs='?', default=argparse.SUPPRESS,
                      help="Optional. Loads model from file directly without read_model.")
    args.add_argument('-api', '--api_type', type=str, required=False, default='async', choices=['sync', 'async'],
                      help='Optional. Enable using sync/async API. Default value is async.')
    advs.add_argument('-nireq', '--number_infer_requests', type=check_positive, required=False, default=0,
                      help='Optional. Number of infer requests. Default value is determined automatically for device.')
    advs.add_argument('-nstreams', '--number_streams', type=str, required=False, default=None,
                      help='Optional. Number of streams to use for inference on the CPU/GPU '
                           '(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> '
                           'or just <nstreams>). '
                           'Default value is determined automatically for a device. Please note that although the automatic selection '
                           'usually provides a reasonable performance, it still may be non - optimal for some cases, especially for very small models. '
                           'Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency '
                           'estimations the number of streams should be set to 1. '
                           'See samples README for more details.')
    advs.add_argument('-inference_only', '--inference_only', type=str2bool, required=False, default=None, nargs='?', const=True,
                      help='Optional. If true inputs filling only once before measurements (default for static models), '
                                     'else inputs filling is included into loop measurement (default for dynamic models)', )
    advs.add_argument('-infer_precision', type=str, required=False,
                      help='Optional. Specifies the inference precision. Example #1: \'-infer_precision bf16\'. Example #2: \'-infer_precision CPU:bf16,GPU:f32\'')

    prpr = parser.add_argument_group('Preprocessing options')
    prpr.add_argument('-ip', '--input_precision', type=str, required=False, choices=INPUT_OUTPUT_PRECISION_CHOICES,
                      help='Optional. Specifies precision for all input layers of the model.')
    prpr.add_argument('-op', '--output_precision', type=str, required=False, choices=INPUT_OUTPUT_PRECISION_CHOICES,
                      help='Optional. Specifies precision for all output layers of the model.')
    prpr.add_argument('-iop', '--input_output_precision', type=str, required=False,
                      help='Optional. Specifies precision for input and output layers by name. Example: -iop "input:f16, output:f16". Notice that quotes are required. Overwrites precision from ip and op options for specified layers.')
    prpr.add_argument('--mean_values', type=str, required=False, default='', metavar='[R,G,B]',
                      help='Optional. Mean values to be used for the input image per channel. Values to be provided in the [R,G,B] format. Can be defined for '
                           'desired input of the model, for example: "--mean_values data[255,255,255],info[255,255,255]". The exact meaning and order of '
                           'channels depend on how the original model was trained. Applying the values affects performance and may cause type conversion')
    prpr.add_argument('--scale_values', type=str, required=False, default='', metavar='[R,G,B]',
                      help='Optional. Scale values to be used for the input image per channel. Values are provided in the [R,G,B] format. Can be defined for '
                           'desired input of the model, for example: "--scale_values data[255,255,255],info[255,255,255]". The exact meaning and order of '
                           'channels depend on how the original model was trained. If both --mean_values and --scale_values are specified, the mean is '
                           'subtracted first and then scale is applied regardless of the order of options in command line. Applying the values affects '
                           'performance and may cause type conversion')

    devp = parser.add_argument_group('Device-specific performance options')
    devp.add_argument('-nthreads', '--number_threads', type=int, required=False, default=None,
                      help='Number of threads to use for inference on the CPU '
                           '(including HETERO and MULTI cases).')
    devp.add_argument('-pin', '--infer_threads_pinning', type=str, required=False,  choices=['YES', 'NO'],
                      help='Optional. Enable threads->cores pinning for CPU-involved inference.')

    stat = parser.add_argument_group('Statistics dumping options')
    stat.add_argument('-latency_percentile', '--latency_percentile', type=int, required=False, default=50,
                      help='Optional. Defines the percentile to be reported in latency metric. The valid range is [1, 100]. The default value is 50 (median).')
    stat.add_argument('-report_type', '--report_type', type=str, required=False,
                      choices=['no_counters', 'average_counters', 'detailed_counters'],
                      help="Optional. Enable collecting statistics report. \"no_counters\" report contains "
                           "configuration options specified, resulting FPS and latency. \"average_counters\" "
                           "report extends \"no_counters\" report and additionally includes average PM "
                           "counters values for each layer from the model. \"detailed_counters\" report "
                           "extends \"average_counters\" report and additionally includes per-layer PM "
                           "counters and latency for each executed infer request.")
    stat.add_argument('-report_folder', '--report_folder', type=str, required=False, default='',
                      help="Optional. Path to a folder where statistics report is stored.")
    args.add_argument('-json_stats', '--json_stats', type=str2bool, required=False, default=False, nargs='?', const=True,
                      help="Optional. Enables JSON-based statistics output (by default reporting system will use CSV format). Should be used together with -report_folder option.")
    stat.add_argument('-pc', '--perf_counts', type=str2bool, required=False, default=False, nargs='?', const=True,
                      help='Optional. Report performance counters.', )
    stat.add_argument('-pcsort', '--perf_counts_sort', type=str, required=False, default="",
                      choices=['no_sort', 'sort', 'simple_sort'],
                      help='Optional. Report performance counters and analysis the sort hotpoint opts.'
                           '  sort: Analysis opts time cost, print by hotpoint order'
                           '  no_sort: Analysis opts time cost, print by normal order'
                           '  simple_sort: Analysis opts time cost, only print EXECUTED opts by normal order', )
    stat.add_argument('-pcseq', '--pcseq', type=str2bool, required=False, default=False, nargs='?', const=True,
                      help='Optional. Report latencies for each shape in -data_shape sequence.', )
    advs.add_argument('-exec_graph_path', '--exec_graph_path', type=str, required=False,
                      help='Optional. Path to a file where to store executable graph information serialized.')
    stat.add_argument('-dump_config', type=str, required=False, default='',
                      help="Optional. Path to JSON file to dump OpenVINO parameters, which were set by application.")
    stat.add_argument('-load_config', type=str, required=False, default='',
                      help="Optional. Path to JSON file to load custom OpenVINO parameters.\n"
                           "Please note, command line parameters have higher priority then parameters from configuration file.\n"
                           "Example 1: a simple JSON file for HW device with primary properties.\n"
                           "             {\n"
                           "                \"CPU\": {\"NUM_STREAMS\": \"3\", \"PERF_COUNT\": \"NO\"}\n"
                           "             }\n"
                           "Example 2: a simple JSON file for meta device(AUTO/MULTI) with HW device properties.\n"
                           "             {\n"
                           "                \"AUTO\": {\n"
                           "                     \"PERFORMANCE_HINT\": \"THROUGHPUT\",\n"
                           "                     \"PERF_COUNT\": \"NO\",\n"
                           "                     \"DEVICE_PROPERTIES\": \"{CPU:{INFERENCE_PRECISION_HINT:f32,NUM_STREAMS:3},GPU:{INFERENCE_PRECISION_HINT:f32,NUM_STREAMS:5}}\"\n"
                           "                }\n"
                           "             }")
    return parser
