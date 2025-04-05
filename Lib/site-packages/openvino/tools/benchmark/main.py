# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from datetime import datetime

from openvino import Dimension, properties

from openvino.tools.benchmark.benchmark import Benchmark
from openvino.tools.benchmark.parameters import parse_args
from openvino.tools.benchmark.utils.constants import MULTI_DEVICE_NAME, \
    CPU_DEVICE_NAME, GPU_DEVICE_NAME, \
    BLOB_EXTENSION, AUTO_DEVICE_NAME
from openvino.tools.benchmark.utils.inputs_filling import get_input_data
from openvino.tools.benchmark.utils.logging import logger
from openvino.tools.benchmark.utils.utils import next_step, get_number_iterations, pre_post_processing, \
    process_help_inference_string, print_perf_counters, print_perf_counters_sort, dump_exec_graph, get_duration_in_milliseconds, \
    get_command_line_arguments, parse_value_per_device, parse_devices, get_inputs_info, \
    print_inputs_and_outputs_info, get_network_batch_size, load_config, dump_config, get_latency_groups, \
    check_for_static, can_measure_as_static, parse_value_for_virtual_device, is_virtual_device, is_virtual_device_found
from openvino.tools.benchmark.utils.statistics_report import StatisticsReport, JsonStatisticsReport, CsvStatisticsReport, \
    averageCntReport, detailedCntReport

def parse_and_check_command_line():
    def arg_not_empty(arg_value,empty_value):
        return not arg_value is None and not arg_value == empty_value

    parser = parse_args()
    args = parser.parse_args()

    if args.latency_percentile < 1 or args.latency_percentile > 100:
        parser.print_help()
        raise RuntimeError("The percentile value is incorrect. The applicable values range is [1, 100].")

    if not args.perf_hint == "none" and (arg_not_empty(args.number_streams, "") or arg_not_empty(args.number_threads, 0) or arg_not_empty(args.infer_threads_pinning, "")):
        raise Exception("-nstreams, -nthreads and -pin options are fine tune options. To use them you " \
                        "should explicitely set -hint option to none. This is not OpenVINO limitation " \
                        "(those options can be used in OpenVINO together), but a benchmark_app UI rule.")

    if args.report_type == "average_counters" and MULTI_DEVICE_NAME in args.target_device:
        raise Exception("only detailed_counters report type is supported for MULTI device")

    _, ext = os.path.splitext(args.path_to_model)
    is_network_compiled = True if ext == BLOB_EXTENSION else False
    is_precisiton_set = not (args.input_precision == "" and args.output_precision == "" and args.input_output_precision == "")

    if is_network_compiled and is_precisiton_set:
        raise Exception("Cannot set precision for a compiled model. " \
                        "Please re-compile your model with required precision.")

    if args.api_type == "sync":
        if args.time == 0 and (args.number_infer_requests > args.number_iterations):
            raise Exception("Number of infer requests should be less than or equal to number of iterations in sync mode.")

    return args, is_network_compiled

def main():
    statistics = None
    try:
        # ------------------------------ 1. Parsing and validating input arguments ------------------------------
        next_step()
        logger.info("Parsing input parameters")
        args, is_network_compiled = parse_and_check_command_line()

        command_line_arguments = get_command_line_arguments(sys.argv)
        if args.report_type:
            _statistics_class = JsonStatisticsReport if args.json_stats else CsvStatisticsReport
            statistics = _statistics_class(StatisticsReport.Config(args.report_type, args.report_folder))
            statistics.add_parameters(StatisticsReport.Category.COMMAND_LINE_PARAMETERS, command_line_arguments)

        def is_flag_set_in_command_line(flag):
            return any(x.strip('-') == flag for x, y in command_line_arguments)

        device_name = args.target_device

        devices = parse_devices(device_name)
        device_number_streams = parse_value_per_device(devices, args.number_streams, "nstreams")
        device_infer_precision = parse_value_per_device(devices, args.infer_precision, "infer_precision")

        config = {}
        if args.load_config:
            load_config(args.load_config, config)

        if is_network_compiled:
            logger.info("Model is compiled")

        # ------------------------------ 2. Loading OpenVINO Runtime -------------------------------------------
        next_step(step_id=2)

        benchmark = Benchmark(args.target_device, args.number_infer_requests,
                              args.number_iterations, args.time, args.api_type,
                              args.inference_only, args.maximum_inference_rate)

        if args.extensions:
            benchmark.add_extension(path_to_extensions=args.extensions)

        ## GPU (clDNN) Extensions
        if GPU_DEVICE_NAME in device_name and args.path_to_cldnn_config:
            if GPU_DEVICE_NAME not in config.keys():
                config[GPU_DEVICE_NAME] = {}
            config[GPU_DEVICE_NAME]['CONFIG_FILE'] = args.path_to_cldnn_config

        if GPU_DEVICE_NAME in config.keys() and 'CONFIG_FILE' in config[GPU_DEVICE_NAME].keys():
            cldnn_config = config[GPU_DEVICE_NAME]['CONFIG_FILE']
            benchmark.add_extension(path_to_cldnn_config=cldnn_config)

        benchmark.print_version_info()

        # --------------------- 3. Setting device configuration --------------------------------------------------------
        next_step()

        def set_performance_hint(device):
            perf_hint = properties.hint.PerformanceMode.THROUGHPUT
            supported_properties = benchmark.core.get_property(device, properties.supported_properties())
            if properties.hint.performance_mode() in supported_properties:
                if is_flag_set_in_command_line('hint'):
                    if args.perf_hint == "throughput" or args.perf_hint == "tput":
                        perf_hint = properties.hint.PerformanceMode.THROUGHPUT
                    elif args.perf_hint == "latency":
                        perf_hint = properties.hint.PerformanceMode.LATENCY
                    elif args.perf_hint == "cumulative_throughput" or args.perf_hint == "ctput":
                        perf_hint = properties.hint.PerformanceMode.CUMULATIVE_THROUGHPUT
                    elif args.perf_hint=='none':
                        # Not set PerformanceMode, and plugin will apply its internal default PerformanceMode
                        return
                    else:
                        raise RuntimeError("Incorrect performance hint. Please set -hint option to"
                            "`throughput`(tput), `latency', 'cumulative_throughput'(ctput) value or 'none'.")
                else:
                    perf_hint = properties.hint.PerformanceMode.LATENCY if benchmark.api_type == "sync" else properties.hint.PerformanceMode.THROUGHPUT
                    logger.warning(f"Performance hint was not explicitly specified in command line. " +
                    f"Device({device}) performance hint will be set to {perf_hint}.")
                config[device][properties.hint.performance_mode()] = perf_hint
            else:
                logger.warning(f"Device {device} does not support performance hint property(-hint).")


        def get_device_type_from_name(name) :
            new_name = str(name)
            new_name = new_name.split(".", 1)[0]
            new_name = new_name.split("(", 1)[0]
            return new_name

        ## Set default values from dumped config
        default_devices = set()
        for device in devices:
            device_type = get_device_type_from_name(device)
            if device_type in config and device not in config:
                config[device] = config[device_type].copy()
                default_devices.add(device_type)

        for def_device in default_devices:
            config.pop(def_device)

        perf_counts = False
        # check if using the virtual device
        hw_devices_list = devices.copy()
        # Remove the hardware devices if AUTO/MULTI/HETERO appears in the devices list.
        is_virtual = is_virtual_device_found(devices)
        if is_virtual:
            devices.clear()
            # Parse out the currect virtual device as the target device.
            virtual_device = device_name.partition(":")[0]
            hw_devices_list.remove(virtual_device)
            devices.append(virtual_device)
            parse_value_for_virtual_device(virtual_device, device_number_streams)
            parse_value_for_virtual_device(virtual_device, device_infer_precision)

        for device in devices:
            supported_properties = benchmark.core.get_property(device, properties.supported_properties())
            if device not in config.keys():
                config[device] = {}

            ## high-level performance modes
            set_performance_hint(device)

            if is_flag_set_in_command_line('nireq'):
                config[device][properties.hint.num_requests()] = str(args.number_infer_requests)

            ## Set performance counter
            if is_flag_set_in_command_line('pc'):
                ## set to user defined value
                config[device][properties.enable_profiling()] = True if args.perf_counts else False
            elif properties.enable_profiling() in config[device].keys() and config[device][properties.enable_profiling()] == True:
                logger.warning(f"Performance counters for {device} device is turned on. " +
                               "To print results use -pc option.")
            elif args.report_type in [ averageCntReport, detailedCntReport ]:
                logger.warning(f"Turn on performance counters for {device} device " +
                               f"since report type is {args.report_type}.")
                config[device][properties.enable_profiling()] = True
            elif args.exec_graph_path is not None:
                logger.warning(f"Turn on performance counters for {device} device " +
                               "due to execution graph dumping.")
                config[device][properties.enable_profiling()] = True
            elif is_flag_set_in_command_line('pcsort'):
                ## set to default value
                logger.warning(f"Turn on performance counters for {device} device " +
                               f"since pcsort value is {args.perf_counts_sort}.")
                config[device][properties.enable_profiling()] = True if args.perf_counts_sort else False
            else:
                ## set to default value
                config[device][properties.enable_profiling()] = args.perf_counts
            perf_counts = True if config[device][properties.enable_profiling()] == True else perf_counts

            ## insert or append property into hw device properties list
            def update_configs(hw_device, property_name, property_value):
                (key, value) = properties.device.properties({hw_device:{property_name:property_value}})
                # add property into hw device properties list.
                if key not in config[device].keys():
                    config[device][key] = value
                else:
                    current_config = config[device][key].get()
                    if hw_device not in current_config.keys():
                        current_config.update(value.get())
                    else:
                        current_device_config = current_config[hw_device]
                        for prop in value.get().items():
                            current_device_config.update(prop[1])
                        current_config[hw_device].update(current_device_config)
                    config[device][key].set(current_config)

            def update_device_config_for_virtual_device(value, config, key):
                # check if the element contains the hardware device property
                if len(value.split(':')) == 1:
                    config[device][key] = device_infer_precision[device]
                else:
                    # set device nstreams properties in the AUTO/MULTI plugin
                    value_vec = value[value.find('{') + 1:value.rfind('}')].split(',')
                    device_properties  = {value_vec[i].split(':')[0] : value_vec[i].split(':')[1] for i in range(0, len(value_vec))}
                    for hw_device in device_properties.keys():
                        update_configs(hw_device, key, device_properties[hw_device])

            ## infer precision
            def set_infer_precision():
                key = properties.hint.inference_precision()
                if device in device_infer_precision.keys():
                    ## set to user defined value
                    if key in supported_properties:
                        config[device][key] = device_infer_precision[device]
                    elif is_virtual_device(device):
                        update_device_config_for_virtual_device(device_infer_precision[device], config, key)
                    else:
                        raise Exception(f"Device {device} doesn't support config key INFERENCE_PRECISION_HINT!" \
                                        " Please specify -infer_precision for correct devices in format" \
                                        " <dev1>:<infer_precision1>,<dev2>:<infer_precision2> or via configuration file.")
                return

            ## the rest are individual per-device settings (overriding the values the device will deduce from perf hint)
            def set_throughput_streams():
                key = get_device_type_from_name(device) + "_THROUGHPUT_STREAMS"
                if device in device_number_streams.keys():
                    ## set to user defined value
                    if key in supported_properties:
                        config[device][key] = device_number_streams[device]
                    elif properties.streams.num() in supported_properties:
                        key = properties.streams.num()
                        config[device][key] = device_number_streams[device]
                    elif is_virtual_device(device):
                        key = properties.streams.num()
                        update_device_config_for_virtual_device(device_number_streams[device], config, key)
                    else:
                        raise Exception(f"Device {device} doesn't support config key '{key}'! " +
                                        "Please specify -nstreams for correct devices in format  <dev1>:<nstreams1>,<dev2>:<nstreams2>")
                elif key not in config[device].keys() and args.api_type == "async" and key not in config[device].keys() \
                    and 'PERFORMANCE_HINT' in config[device].keys() and config[device]['PERFORMANCE_HINT'] == '':
                    ## set the _AUTO value for the #streams
                    logger.warning(f"-nstreams default value is determined automatically for {device} device. " +
                                   "Although the automatic selection usually provides a reasonable performance, "
                                   "but it still may be non-optimal for some cases, for more information look at README.")
                    if key in supported_properties:
                        config[device][key] = get_device_type_from_name(device) + "_THROUGHPUT_AUTO"
                    elif properties.streams.Num() in supported_properties:
                        key = properties.streams.Num()
                        config[device][key] = "-1"  # Set AUTO mode for streams number
                    elif is_virtual_device(device):
                        # Set nstreams to default value auto if no nstreams specified from cmd line.
                        for hw_device in hw_devices_list:
                            hw_supported_properties = benchmark.core.get_property(hw_device, properties.supported_properties())
                            key = get_device_type_from_name(hw_device) + "_THROUGHPUT_STREAMS"
                            value = get_device_type_from_name(hw_device) + "_THROUGHPUT_AUTO"
                            if key not in hw_supported_properties:
                                key = properties.streams.Num()
                                value = properties.streams.Num.AUTO
                            if key in hw_supported_properties:
                                update_configs(hw_device, key, value)
                if key in config[device].keys():
                    device_number_streams[device] = config[device][key]
                return

            def set_nthreads_pin(property_name, property_value):
                if property_name in supported_properties or device_name == AUTO_DEVICE_NAME:
                    # create nthreads/pin primary property for HW device or AUTO if -d is AUTO directly.
                    config[device][property_name] = property_value
                elif is_virtual:
                    # Create secondary property of -nthreads/-pin only for CPU if CPU device appears in the devices
                    # list specified by -d.
                    if CPU_DEVICE_NAME in hw_devices_list:
                        update_configs(CPU_DEVICE_NAME, property_name, property_value)
                return

            if args.number_threads and is_flag_set_in_command_line("nthreads"):
                # limit threading for CPU portion of inference
                set_nthreads_pin(properties.inference_num_threads(), str(args.number_threads))

            if is_flag_set_in_command_line('pin'):
                ## set for CPU to user defined value
                set_nthreads_pin(properties.hint.enable_cpu_pinning(), args.infer_threads_pinning)

            set_throughput_streams()
            set_infer_precision()

            if is_virtual_device(device):
                if device in device_number_streams.keys():
                    del device_number_streams[device]

        device_config = {}
        # In case of multiple devices found prefer the one given in CLI argument
        if benchmark.device.find(device_name) == 0 and device_name in config.keys():
            device_config = config[device_name]
        else:
            for device in config:
                if benchmark.device.find(device) == 0:
                    device_config = config[device]
        if args.cache_dir:
            benchmark.set_cache_dir(args.cache_dir)

        ## If set batch size, disable the auto batching
        if args.batch_size:
            logger.warning("Batch size is set. Auto batching will be disabled")
            device_config["ALLOW_AUTO_BATCHING"] = False

        topology_name = ""
        load_from_file_enabled = is_flag_set_in_command_line('load_from_file') or is_flag_set_in_command_line('lfile')
        if load_from_file_enabled and not is_network_compiled:
            if args.mean_values or args.scale_values:
                raise RuntimeError("--mean_values and --scale_values aren't supported with --load_from_file. "
                    "The values can be set via model_optimizer while generating xml")
            next_step()
            print("Skipping the step for loading model from file")
            next_step()
            print("Skipping the step for loading model from file")
            next_step()
            print("Skipping the step for loading model from file")

            # --------------------- 7. Loading the model to the device -------------------------------------------------
            next_step()

            start_time = datetime.utcnow()
            compiled_model = benchmark.core.compile_model(args.path_to_model, benchmark.device, device_config)
            duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
            logger.info(f"Compile model took {duration_ms} ms")
            if statistics:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('compile model time (ms)', duration_ms)
                                          ])
            app_inputs_info, _ = get_inputs_info(args.shape, args.data_shape, args.layout, args.batch_size, args.scale_values, args.mean_values, compiled_model.inputs)
            batch_size = get_network_batch_size(app_inputs_info)
        elif not is_network_compiled:
            # --------------------- 4. Read the Intermediate Representation of the network -----------------------------
            next_step()

            logger.info("Loading model files")

            start_time = datetime.utcnow()
            model = benchmark.read_model(args.path_to_model)
            topology_name = model.get_name()
            duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
            logger.info(f"Read model took {duration_ms} ms")
            logger.info("Original model I/O parameters:")
            print_inputs_and_outputs_info(model)

            if statistics:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('read model time (ms)', duration_ms)
                                          ])

            # --------------------- 5. Resizing network to match image sizes and given batch ---------------------------
            next_step()

            app_inputs_info, reshape = get_inputs_info(args.shape, args.data_shape, args.layout, args.batch_size, args.scale_values, args.mean_values, model.inputs)

            # use batch size according to provided layout and shapes
            batch_size = get_network_batch_size(app_inputs_info)
            logger.info(f'Model batch size: {batch_size}')

            if reshape:
                start_time = datetime.utcnow()
                shapes = { info.name : info.partial_shape for info in app_inputs_info }
                logger.info(
                    'Reshaping model: {}'.format(', '.join("'{}': {}".format(k, str(v)) for k, v in shapes.items())))
                model.reshape(shapes)
                duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
                logger.info(f"Reshape model took {duration_ms} ms")
                if statistics:
                    statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                              [
                                                  ('reshape model time (ms)', duration_ms)
                                              ])

            # --------------------- 6. Configuring inputs and outputs of the model --------------------------------------------------
            next_step()

            pre_post_processing(model, app_inputs_info, args.input_precision, args.output_precision, args.input_output_precision)
            print_inputs_and_outputs_info(model)

            # --------------------- 7. Loading the model to the device -------------------------------------------------
            next_step()
            start_time = datetime.utcnow()
            compiled_model = benchmark.core.compile_model(model, benchmark.device, device_config)

            duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
            logger.info(f"Compile model took {duration_ms} ms")
            if statistics:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('compile model time (ms)', duration_ms)
                                          ])
        else:
            if args.mean_values or args.scale_values:
                raise RuntimeError("--mean_values and --scale_values aren't supported for compiled model. "
                    "The values can be set via model_optimizer while generating xml")
            next_step()
            print("Skipping the step for compiled model")
            next_step()
            print("Skipping the step for compiled model")
            next_step()
            print("Skipping the step for compiled model")

            # --------------------- 7. Loading the model to the device -------------------------------------------------
            next_step()

            start_time = datetime.utcnow()
            compiled_model = benchmark.core.import_model(args.path_to_model, benchmark.device, device_config)
            duration_ms = f"{(datetime.utcnow() - start_time).total_seconds() * 1000:.2f}"
            logger.info(f"Import model took {duration_ms} ms")
            if statistics:
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ('import model time (ms)', duration_ms)
                                          ])
            app_inputs_info, _ = get_inputs_info(args.shape, args.data_shape, args.layout, args.batch_size, args.scale_values, args.mean_values, compiled_model.inputs)
            batch_size = get_network_batch_size(app_inputs_info)

        # --------------------- 8. Querying optimal runtime parameters --------------------------------------------------
        next_step()

        ## actual device-deduced settings
        keys = compiled_model.get_property(properties.supported_properties())
        logger.info("Model:")
        for k in keys:
            skip_keys = (properties.supported_properties())
            if k not in skip_keys:
                value = compiled_model.get_property(k)
                if k == properties.device.properties():
                    for device_key in value.keys():
                        logger.info(f'  {device_key}:')
                        for k2, value2 in value.get(device_key).items():
                            if k2 not in skip_keys:
                                logger.info(f'    {k2}: {value2}')
                else:
                    logger.info(f'  {k}: {value}')

        # Update number of streams
        for device in device_number_streams.keys():
            try:
                key = get_device_type_from_name(device) + '_THROUGHPUT_STREAMS'
                device_number_streams[device] = compiled_model.get_property(key)
            except:
                key = 'NUM_STREAMS'
                device_number_streams[device] = compiled_model.get_property(key)

        # ------------------------------------ 9. Creating infer requests and preparing input data ----------------------
        next_step()

        # Create infer requests
        requests = benchmark.create_infer_requests(compiled_model)

        # Prepare input data
        paths_to_input = list()
        if args.paths_to_input:
            for path in args.paths_to_input:
                if ":" in next(iter(path), ""):
                    paths_to_input.extend(path)
                else:
                    paths_to_input.append(os.path.abspath(*path))

        data_queue = get_input_data(paths_to_input, app_inputs_info)

        static_mode = check_for_static(app_inputs_info)
        allow_inference_only_or_sync = can_measure_as_static(app_inputs_info)
        if not allow_inference_only_or_sync and benchmark.api_type == 'sync':
            raise Exception("Benchmarking of the model with dynamic shapes is available for async API only. "
                            "Please use -api async -hint latency -nireq 1 to emulate sync behavior.")

        if benchmark.inference_only == None:
            if static_mode:
                benchmark.inference_only = True
            else:
                benchmark.inference_only = False
        elif benchmark.inference_only and not allow_inference_only_or_sync:
            raise Exception("Benchmarking dynamic model available with input filling in measurement loop only!")

        # update batch size in case dynamic network with one data_shape
        if allow_inference_only_or_sync and batch_size.is_dynamic:
            batch_size = Dimension(data_queue.batch_sizes[data_queue.current_group_id])

        benchmark.latency_groups = get_latency_groups(app_inputs_info)

        if len(benchmark.latency_groups) > 1:
            logger.info(f"Defined {len(benchmark.latency_groups)} tensor groups:")
            for group in benchmark.latency_groups:
                logger.info(f"\t{str(group)}")

        # Iteration limit
        benchmark.niter = get_number_iterations(benchmark.niter, benchmark.nireq, max(len(info.shapes) for info in app_inputs_info), benchmark.api_type)

        # Set input tensors before first inference
        for request in requests:
            data_tensors = data_queue.get_next_input()
            for port, data_tensor in data_tensors.items():
                input_tensor = request.get_input_tensor(port)
                if not static_mode:
                    input_tensor.shape = data_tensor.shape
                if not len(input_tensor.shape):
                    input_tensor.data.flat[:] = data_tensor.data
                else:
                    input_tensor.data[:] = data_tensor.data

        if statistics:
            statistics.add_parameters(StatisticsReport.Category.RUNTIME_CONFIG,
                                      [
                                          ('topology', topology_name),
                                          ('target device', device_name),
                                          ('API', args.api_type),
                                          ('inference_only', benchmark.inference_only),
                                          ('precision', "UNSPECIFIED"),
                                          ('batch size', str(batch_size)),
                                          ('number of iterations', str(benchmark.niter)),
                                          ('number of parallel infer requests', str(benchmark.nireq)),
                                          ('duration (ms)', str(get_duration_in_milliseconds(benchmark.duration_seconds))),
                                       ])

            for nstreams in device_number_streams.items():
                statistics.add_parameters(StatisticsReport.Category.RUNTIME_CONFIG,
                                         [
                                            (f"number of {nstreams[0]} streams", str(nstreams[1])),
                                         ])

        # ------------------------------------ 10. Measuring performance -----------------------------------------------

        output_string = process_help_inference_string(benchmark, device_number_streams)

        next_step(additional_info=output_string)

        if benchmark.inference_only:
            logger.info("Benchmarking in inference only mode (inputs filling are not included in measurement loop).")
        else:
            logger.info("Benchmarking in full mode (inputs filling are included in measurement loop).")
        duration_ms = f"{benchmark.first_infer(requests):.2f}"
        logger.info(f"First inference took {duration_ms} ms")
        if statistics:
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                    [
                                        ('first inference time (ms)', duration_ms)
                                    ])

        pcseq = args.pcseq
        if static_mode or len(benchmark.latency_groups) == 1:
            pcseq = False

        fps, median_latency_ms, avg_latency_ms, min_latency_ms, max_latency_ms, total_duration_sec, iteration = benchmark.main_loop(requests, data_queue, batch_size, args.latency_percentile, pcseq)

        # ------------------------------------ 11. Dumping statistics report -------------------------------------------
        next_step()

        if args.dump_config:
            dump_config(args.dump_config, config)
            logger.info(f"OpenVINO configuration settings were dumped to {args.dump_config}")

        if args.exec_graph_path:
            dump_exec_graph(compiled_model, args.exec_graph_path)

        if perf_counts:
            perfs_count_list = []
            for request in requests:
                perfs_count_list.append(request.profiling_info)

            if args.perf_counts_sort:
                total_sorted_list = print_perf_counters_sort(perfs_count_list,sort_flag=args.perf_counts_sort)
                if statistics:
                    statistics.dump_performance_counters_sorted(total_sorted_list)

            elif args.perf_counts:
                print_perf_counters(perfs_count_list)

            if statistics:
                # if not args.perf_counts_sort:
                statistics.dump_performance_counters(perfs_count_list)

        if statistics:
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('total execution time (ms)', f'{get_duration_in_milliseconds(total_duration_sec):.2f}'),
                                          ('total number of iterations', str(iteration)),
                                      ])
            if MULTI_DEVICE_NAME not in device_name:
                latency_prefix = None
                if args.latency_percentile == 50:
                    latency_prefix = 'latency (ms)'
                elif args.latency_percentile != 50:
                    latency_prefix = 'latency (' + str(args.latency_percentile) + ' percentile) (ms)'
                if latency_prefix:
                    statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                            [
                                                (latency_prefix, f'{median_latency_ms:.2f}'),
                                            ])
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("avg latency", f'{avg_latency_ms:.2f}'),
                                          ])
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("min latency", f'{min_latency_ms:.2f}'),
                                          ])
                statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("max latency", f'{max_latency_ms:.2f}'),
                                          ])
                if pcseq:
                    for group in benchmark.latency_groups:
                        statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("group", str(group)),
                                          ])
                        statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("avg latency", f'{group.avg:.2f}'),
                                          ])
                        statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("min latency", f'{group.min:.2f}'),
                                          ])
                        statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                          [
                                              ("max latency", f'{group.max:.2f}'),
                                          ])
            statistics.add_parameters(StatisticsReport.Category.EXECUTION_RESULTS,
                                      [
                                          ('throughput', f'{fps:.2f}'),
                                      ])
            statistics.dump()

        try:
            exeDevice = compiled_model.get_property("EXECUTION_DEVICES")
            logger.info(f'Execution Devices:{exeDevice}')
        except:
            pass
        logger.info(f'Count:            {iteration} iterations')
        logger.info(f'Duration:         {get_duration_in_milliseconds(total_duration_sec):.2f} ms')
        if MULTI_DEVICE_NAME not in device_name:
            logger.info('Latency:')
            if args.latency_percentile == 50:
                logger.info(f'   Median:        {median_latency_ms:.2f} ms')
            elif args.latency_percentile != 50:
                logger.info(f'   {args.latency_percentile} percentile:     {median_latency_ms:.2f} ms')
            logger.info(f'   Average:       {avg_latency_ms:.2f} ms')
            logger.info(f'   Min:           {min_latency_ms:.2f} ms')
            logger.info(f'   Max:           {max_latency_ms:.2f} ms')

            if pcseq:
                logger.info("Latency for each data shape group:")
                for idx,group in enumerate(benchmark.latency_groups):
                    logger.info(f"{idx+1}.{str(group)}")
                    if args.latency_percentile == 50:
                        logger.info(f'   Median:     {group.median:.2f} ms')
                    elif args.latency_percentile != 50:
                        logger.info(f'   {args.latency_percentile} percentile:     {group.median:.2f} ms')
                    logger.info(f'   Average:    {group.avg:.2f} ms')
                    logger.info(f'   Min:        {group.min:.2f} ms')
                    logger.info(f'   Max:        {group.max:.2f} ms')

        logger.info(f'Throughput:   {fps:.2f} FPS')

        del compiled_model

        next_step.step_id = 0
    except Exception as e:
        logger.exception(e)

        if statistics:
            statistics.add_parameters(
                StatisticsReport.Category.EXECUTION_RESULTS,
                [('error', str(e))]
            )
            statistics.dump()
        sys.exit(1)
