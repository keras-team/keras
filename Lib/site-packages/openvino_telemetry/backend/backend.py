# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import abc

from ..utils.message import Message


class BackendRegistry:
    """
    The class that stores information about all registered telemetry backends
    """
    r = {}

    @classmethod
    def register_backend(cls, id: str, backend):
        cls.r[id] = backend

    @classmethod
    def get_backend(cls, id: str):
        if id not in cls.r:
            raise RuntimeError('The backend with id "{}" is not registered'.format(id))
        return cls.r.get(id)


class TelemetryBackendMetaClass(abc.ABCMeta):
    def __init__(cls, clsname, bases, methods):
        super().__init__(clsname, bases, methods)
        if cls.id is not None:
            BackendRegistry.register_backend(cls.id, cls)


class TelemetryBackend(metaclass=TelemetryBackendMetaClass):
    id = None

    @abc.abstractmethod
    def __init__(self, tid: str, app_name: str, app_version: str):
        """
        Initializer of the class
        :param tid: database id
        :param app_name: name of the application
        :param app_version: version of the application
        """

    @abc.abstractmethod
    def send(self, message: Message):
        """
        Sends the message to the backend.
        :param message: The Message object to send
        :return: None
        """

    @abc.abstractmethod
    def build_event_message(self, event_category: str, event_action: str, event_label: str, event_value: int = 1,
                            **kwargs):
        """
        Should return the Message object build from the event message.
        """

    @abc.abstractmethod
    def build_error_message(self, category: str, error_msg: str, **kwargs):
        """
        Should return the Message object build from the error message.
        """

    @abc.abstractmethod
    def build_stack_trace_message(self, category: str, error_msg: str, **kwargs):
        """
        Should return the Message object build from the stack trace message.
        """

    @abc.abstractmethod
    def build_session_start_message(self, category: str, **kwargs):
        """
        Should return the Message object corresponding to the session start.
        """

    @abc.abstractmethod
    def build_session_end_message(self, category: str, **kwargs):
        """
        Should return the Message object corresponding to the session end.
        """

    @abc.abstractmethod
    def remove_cid_file(self):
        """
        Should remove client ID file.
        """

    @abc.abstractmethod
    def generate_new_cid_file(self):
        """
        Should generate new Client ID file.
        """

    @abc.abstractmethod
    def cid_file_initialized(self):
        """
        Should check if client ID file is initialized.

        :return: True if client ID file is initialized, otherwise False.
        """

    @abc.abstractmethod
    def set_stats(self, data: dict):
        """
        Pass additional statistics, which will be added to telemetry messages
        """