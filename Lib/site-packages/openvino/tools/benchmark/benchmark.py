# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from datetime import datetime
from math import ceil
from openvino import Core, get_version, AsyncInferQueue

from .utils.constants import GPU_DEVICE_NAME, XML_EXTENSION, BIN_EXTENSION
from .utils.logging import logger
from .utils.utils import get_duration_seconds

def percentile(values, percent):
    return values[ceil(len(values) * percent / 100) - 1]

class Benchmark:
    def __init__(self, device: str, number_infer_requests: int = 0, number_iterations: int = None,
                 duration_seconds: int = None, api_type: str = 'async', inference_only = None,
                 maximum_inference_rate: float = 0):
        self.device = device
        self.core = Core()
        self.nireq = number_infer_requests if api_type == 'async' else 1
        self.niter = number_iterations
        self.duration_seconds = get_duration_seconds(duration_seconds, self.niter, self.device)
        self.api_type = api_type
        self.inference_only = inference_only
        self.latency_groups = []
        self.max_irate = maximum_inference_rate

    def __del__(self):
        del self.core

    def add_extension(self, path_to_extensions: str=None, path_to_cldnn_config: str=None):
        if path_to_cldnn_config:
            self.core.set_property(GPU_DEVICE_NAME, {'CONFIG_FILE': path_to_cldnn_config})
            logger.info(f'GPU extensions is loaded {path_to_cldnn_config}')

        if path_to_extensions:
            for extension in path_to_extensions.split(","):
                logger.info(f"Loading extension {extension}")
                self.core.add_extension(extension)

    def print_version_info(self) -> None:
        version = get_version()
        logger.info('OpenVINO:')
        logger.info(f"{'Build ':.<39} {version}")
        logger.info("")

        logger.info("Device info:")
        for device, version in self.core.get_versions(self.device).items():
            logger.info(f"{device}")
            logger.info(f"{'Build ':.<39} {version.build_number}")

        logger.info("")
        logger.info("")

    def set_config(self, config = {}):
        for device in config.keys():
            self.core.set_property(device, config[device])

    def set_cache_dir(self, cache_dir: str):
        self.core.set_property({'CACHE_DIR': cache_dir})

    def read_model(self, path_to_model: str):
        model_filename = os.path.abspath(path_to_model)
        head, ext = os.path.splitext(model_filename)
        weights_filename = os.path.abspath(head + BIN_EXTENSION) if ext == XML_EXTENSION else ""
        return self.core.read_model(model_filename, weights_filename)

    def create_infer_requests(self, compiled_model):
        if self.api_type == 'sync':
            requests = [compiled_model.create_infer_request()]
        else:
            requests = AsyncInferQueue(compiled_model, self.nireq)
            self.nireq = len(requests)
        return requests

    def first_infer(self, requests):
        if self.api_type == 'sync':
            requests[0].infer()
            return requests[0].latency
        else:
            id = requests.get_idle_request_id()
            requests.start_async()
            requests.wait_all()
            return requests[id].latency

    def inference_rate_delay(self, processed_frames, exec_time):
        if self.max_irate > 0:
            nextRunFinishTime = 1 / self.max_irate * processed_frames
            delay = nextRunFinishTime - exec_time
            time.sleep(delay if delay > 0 else 0)

    def sync_inference(self, request, data_queue):
        processed_frames = 0
        exec_time = 0
        iteration = 0
        times = []
        start_time = datetime.utcnow()
        while (self.niter and iteration < self.niter) or \
              (self.duration_seconds and exec_time < self.duration_seconds):
            processed_frames += data_queue.get_next_batch_size()
            if self.inference_only == False:
                request.set_input_tensors(data_queue.get_next_input())
            request.infer()
            times.append(request.latency)
            iteration += 1

            exec_time = (datetime.utcnow() - start_time).total_seconds()
            self.inference_rate_delay(processed_frames, exec_time)
        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        return sorted(times), total_duration_sec, iteration

    def async_inference_only(self, infer_queue, data_queue):
        processed_frames = 0
        exec_time = 0
        iteration = 0
        times = []
        in_fly = set()
        start_time = datetime.utcnow()
        while (self.niter and iteration < self.niter) or \
              (self.duration_seconds and exec_time < self.duration_seconds) or \
              (iteration % self.nireq):
            processed_frames += data_queue.get_next_batch_size()
            idle_id = infer_queue.get_idle_request_id()
            if idle_id in in_fly:
                times.append(infer_queue[idle_id].latency)
            else:
                in_fly.add(idle_id)
            infer_queue.start_async()
            iteration += 1

            exec_time = (datetime.utcnow() - start_time).total_seconds()
            self.inference_rate_delay(processed_frames, exec_time)

        infer_queue.wait_all()
        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        for infer_request_id in in_fly:
            times.append(infer_queue[infer_request_id].latency)
        return sorted(times), total_duration_sec, iteration

    def async_inference_full_mode(self, infer_queue, data_queue, pcseq):
        processed_frames = 0
        exec_time = 0
        iteration = 0
        times = []
        num_groups = len(self.latency_groups)
        start_time = datetime.utcnow()
        in_fly = set()
        while (self.niter and iteration < self.niter) or \
              (self.duration_seconds and exec_time < self.duration_seconds) or \
              (iteration % num_groups):
            processed_frames += data_queue.get_next_batch_size()
            idle_id = infer_queue.get_idle_request_id()
            if idle_id in in_fly:
                times.append(infer_queue[idle_id].latency)
                if pcseq:
                    self.latency_groups[infer_queue.userdata[idle_id]].times.append(infer_queue[idle_id].latency)
            else:
                in_fly.add(idle_id)
            group_id = data_queue.current_group_id
            infer_queue[idle_id].set_input_tensors(data_queue.get_next_input())
            infer_queue.start_async(userdata=group_id)
            iteration += 1

            exec_time = (datetime.utcnow() - start_time).total_seconds()
            self.inference_rate_delay(processed_frames, exec_time)
        infer_queue.wait_all()
        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        
        for infer_request_id in in_fly:
            times.append(infer_queue[infer_request_id].latency)
            if pcseq:
                self.latency_groups[infer_queue.userdata[infer_request_id]].times.append(infer_queue[infer_request_id].latency)
        
        return sorted(times), total_duration_sec, processed_frames, iteration

    def main_loop(self, requests, data_queue, batch_size, latency_percentile, pcseq):
        if self.api_type == 'sync':
            times, total_duration_sec, iteration = self.sync_inference(requests[0], data_queue)
            fps = len(batch_size) * iteration / total_duration_sec
        elif self.inference_only:
            times, total_duration_sec, iteration = self.async_inference_only(requests, data_queue)
            fps = len(batch_size) * iteration / total_duration_sec
        else:
            times, total_duration_sec, processed_frames, iteration = self.async_inference_full_mode(requests, data_queue, pcseq)
            fps = processed_frames / total_duration_sec

        median_latency_ms = percentile(times, latency_percentile)
        avg_latency_ms = sum(times) / len(times)
        min_latency_ms = times[0]
        max_latency_ms = times[-1]

        if pcseq:
            for group in self.latency_groups:
                if group.times:
                    group.times.sort()
                    group.median = percentile(group.times, latency_percentile)
                    group.avg = sum(group.times) / len(group.times)
                    group.min = group.times[0]
                    group.max = group.times[-1]
        return fps, median_latency_ms, avg_latency_ms, min_latency_ms, max_latency_ms, total_duration_sec, iteration
