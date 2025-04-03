# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import threading
from concurrent import futures
from time import sleep

from ..backend.backend import TelemetryBackend
from ..utils.message import Message

MAX_QUEUE_SIZE = 1000


class TelemetrySender:
    def __init__(self, max_workers=None):
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self.queue_size = 0
        self._lock = threading.Lock()

    def send(self, backend: TelemetryBackend, message: Message):
        def _future_callback(future):
            with self._lock:
                self.queue_size -= 1

        free_space = False
        with self._lock:
            if self.queue_size < MAX_QUEUE_SIZE:
                free_space = True
                self.queue_size += 1
            else:
                pass  # dropping a message because the queue is full
        # to avoid dead lock we should not add callback inside the "with self._lock" block because it will be executed
        # immediately if the fut is available
        if free_space:
            try:
                fut = self.executor.submit(backend.send, message)
                fut.add_done_callback(_future_callback)
            except Exception as err:
                pass  # nosec

    def force_shutdown(self, timeout: float):
        """
        Forces all threads to be stopped after timeout. The "shutdown" method of the ThreadPoolExecutor removes only not
        yet scheduled threads and keep running the existing one. In order to stop the already running use some low-level
        attribute. The operation with low-level attributes is wrapped with the try/except to avoid potential crash if
        these attributes will removed or renamed.

        :param timeout: timeout to wait before the shutdown
        :return: None
        """
        need_sleep = False
        with self._lock:
            if self.queue_size > 0:
                need_sleep = True
        if need_sleep:
            sleep(timeout)
        self.executor.shutdown(wait=False)

        try:
            self.executor._threads.clear()
            futures.thread._threads_queues.clear()
        except Exception as err:
            pass  # nosec
