# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os
import sys
from enum import Enum

from .backend.backend import BackendRegistry
from .utils.sender import TelemetrySender
from .utils.opt_in_checker import OptInChecker, ConsentCheckResult, DialogResult
from .utils.stats_processor import StatsProcessor


class OptInStatus(Enum):
    ACCEPTED = "accepted"
    DECLINED = "declined"
    UNDEFINED = "undefined"


class SingletonMetaClass(type):
    def __init__(self, cls_name, super_classes, dic):
        self.__single_instance = None
        super().__init__(cls_name, super_classes, dic)

    def __call__(cls, *args, **kwargs):
        if cls.__single_instance is None:
            cls.__single_instance = super(SingletonMetaClass, cls).__call__(*args, **kwargs)
        return cls.__single_instance


class Telemetry(metaclass=SingletonMetaClass):
    """
    The main class to send telemetry data. It uses singleton pattern. The instance should be initialized with the
    application name, version and tracking id just once. Later the instance can be created without parameters.
    Args:
        :param app_name: The name of the application.
        :param app_version: The version of the application.
        :param tid: The ID of telemetry base.
        :param backend: Telemetry backend name.
        :param enable_opt_in_dialog: boolean flag to turn on or turn off opt-in dialog.
        If enable_opt_in_dialog=True opt-in dialog is shown during first usage of openvino tools,
        no telemetry is sent until user accepts telemetry with dialog.
        If enable_opt_in_dialog=False, telemetry is sent without opt-in dialog, unless user explicitly turned it off
        with opt_in_out script.
        :param disable_in_ci: Turn off telemetry for CI jobs.
    """

    def __init__(self, app_name: str = None, app_version: str = None, tid: str = None,
                 backend: [str, None] = 'ga', enable_opt_in_dialog=True, disable_in_ci=False):
        # The case when instance is already configured
        if app_name is None:
            if not hasattr(self, 'sender') or self.sender is None:
                raise RuntimeError('The first instantiation of the Telemetry should be done with the '
                                   'application name, version and TID.')
            return

        self.init(app_name, app_version, tid, backend, enable_opt_in_dialog, disable_in_ci)

    def init(self, app_name: str = None, app_version: str = None, tid: str = None,
             backend: [str, None] = 'ga', enable_opt_in_dialog=True, disable_in_ci=False):
        opt_in_checker = OptInChecker()
        opt_in_check_result = opt_in_checker.check(enable_opt_in_dialog, disable_in_ci)
        if enable_opt_in_dialog:
            self.consent = opt_in_check_result == ConsentCheckResult.ACCEPTED
        else:
            self.consent = opt_in_check_result == ConsentCheckResult.ACCEPTED or \
                           opt_in_check_result == ConsentCheckResult.NO_FILE

        if tid is None:
            log.warning("Telemetry will not be sent as TID is not specified.")

        self.tid = tid
        self.backend = BackendRegistry.get_backend(backend)(self.tid, app_name, app_version)
        self.sender = TelemetrySender()

        if self.consent and not self.backend.cid_file_initialized():
            self.backend.generate_new_cid_file()

        if self.consent:
            data = self.get_stats()
            if data is not None and isinstance(data, dict):
                self.backend.set_stats(data)

        if not enable_opt_in_dialog and self.consent:
            # Try to create directory for client ID if it does not exist
            if not opt_in_checker.create_or_check_consent_dir():
                log.warning("Could not create directory for storing client ID. No data will be sent.")
                return

            # Generate client ID if it does not exist
            if not self.backend.cid_file_initialized():
                self.backend.generate_new_cid_file()
            return

        # Consent file may be absent, for example, during the first run of Openvino tool.
        # In this case we trigger opt-in dialog that asks user permission for sending telemetry.
        if opt_in_check_result == ConsentCheckResult.NO_FILE:
            if opt_in_checker.create_or_check_consent_dir():
                answer = ConsentCheckResult.DECLINED

                # check if it is openvino tool
                if not self.check_by_cmd_line_if_dialog_needed():
                    return

                # create openvino_telemetry file if possible with "0" value
                if not opt_in_checker.update_result(answer):
                    return
                try:
                    # run opt-in dialog
                    answer = opt_in_checker.opt_in_dialog()
                    if answer == DialogResult.ACCEPTED:
                        # If the dialog result is "accepted" we generate new client ID file and update openvino_telemetry
                        # file with "1" value. Telemetry data will be collected in this case.
                        self.consent = True
                        self.backend.generate_new_cid_file()
                        self.send_opt_in_event(OptInStatus.ACCEPTED)

                        # Here we send telemetry with "accepted" dialog result
                        opt_in_checker.update_result(ConsentCheckResult.ACCEPTED)
                    elif answer == DialogResult.TIMEOUT_REACHED:
                        # If timer was reached and we have no response from user, we should not send
                        # any data except for dialog result.
                        # At this point we have already created openvino_telemetry file with "0" value,
                        # which means that no telemetry data will be collected in further functions.
                        # As openvino_telemetry file exists on the system the dialog won't be shown again.

                        # Here we send telemetry with "timer reached" dialog result
                        self.send_opt_in_event(OptInStatus.UNDEFINED, force_send=True)
                    else:
                        # If the dialog result is "declined" we should not send any data except for dialog result.
                        # At this point we have already created openvino_telemetry file with "0" value,
                        # which means that no telemetry data will be collected in further functions.
                        # As openvino_telemetry file exists on the system the dialog won't be shown again.

                        # Here we send telemetry with "declined" dialog result
                        self.send_opt_in_event(OptInStatus.DECLINED, force_send=True)
                except KeyboardInterrupt:
                    pass

    def check_by_cmd_line_if_dialog_needed(self):
        scripts_to_run_dialog = [
            os.path.join("openvino", "tools", "mo", "main"),
            os.path.join("openvino", "tools", "ovc", "main"),
            "ovc",
            "mo",
            "pot",
            "omz_downloader",
            "omz_converter",
            "omz_data_downloader",
            "omz_info_dumper",
            "omz_quantizer",
            "accuracy_check"
        ]
        extensions = [".py", ".exe", ""]
        args = sys.argv
        if len(args) == 0:
            return False
        script_name = args[0]

        for script_to_run_dialog in scripts_to_run_dialog:
            for ext in extensions:
                script_to_check = script_to_run_dialog + ext
                if script_name == script_to_check or script_name.endswith(os.sep + script_to_check):
                    return True
        return False

    def force_shutdown(self, timeout: float = 1.0):
        """
        Stops currently running threads which may be hanging because of no Internet connection.

        :param timeout: maximum timeout time
        :return: None
        """
        self.sender.force_shutdown(timeout)

    def send_event(self, event_category: str, event_action: str, event_label: str, event_value: int = 1,
                   app_name=None, app_version=None, force_send=False, **kwargs):
        """
        Send single event.

        :param event_category: category of the event
        :param event_action: action of the event
        :param event_label: the label associated with the action
        :param event_value: the integer value corresponding to this label
        :param app_name: application name
        :param app_version: application version
        :param force_send: forces to send event ignoring the consent value
        :param kwargs: additional parameters
        :return: None
        """
        if self.consent or force_send:
            self.sender.send(self.backend, self.backend.build_event_message(event_category, event_action, event_label,
                                                                            event_value, app_name, app_version,
                                                                            **kwargs))

    def start_session(self, category: str, **kwargs):
        """
        Sends a message about starting of a new session.

        :param kwargs: additional parameters
        :param category: the application code
        :return: None
        """
        if self.consent:
            self.sender.send(self.backend, self.backend.build_session_start_message(category, **kwargs))

    def end_session(self, category: str, **kwargs):
        """
        Sends a message about ending of the current session.

        :param kwargs: additional parameters
        :param category: the application code
        :return: None
        """
        if self.consent:
            self.sender.send(self.backend, self.backend.build_session_end_message(category, **kwargs))

    def send_error(self, category: str, error_msg: str, **kwargs):
        if self.consent:
            self.sender.send(self.backend, self.backend.build_error_message(category, error_msg, **kwargs))

    def send_stack_trace(self, category: str, stack_trace: str, **kwargs):
        if self.consent:
            self.sender.send(self.backend, self.backend.build_stack_trace_message(category, stack_trace, **kwargs))

    @staticmethod
    def _update_opt_in_status(tid: str, new_opt_in_status: bool):
        """
        Updates opt-in status.

        :param tid: ID of telemetry base.
        :param new_opt_in_status: new opt-in status.
        :return: None
        """
        app_name = 'opt_in_out'
        app_version = Telemetry.get_version()
        opt_in_checker = OptInChecker()
        # If enable_opt_in_dialog=True is set opt_in_checker.check() makes extra checks,
        # if it is the main process or a notebook opt_in_checker.check() returns ConsentCheckResult.DECLINED.
        # It is needed only in init() method to prevent opt-in dialog for these cases.
        # Here these checks are not needed, because _update_opt_in_status() is executed only from opt_in_out
        # which does not trigger opt-in dialog in any case.
        opt_in_check = opt_in_checker.check(enable_opt_in_dialog=False)

        prev_status = OptInStatus.UNDEFINED
        if opt_in_check == ConsentCheckResult.DECLINED:
            prev_status = OptInStatus.DECLINED
        elif opt_in_check == ConsentCheckResult.ACCEPTED:
            prev_status = OptInStatus.ACCEPTED

        if new_opt_in_status:
            updated = opt_in_checker.update_result(ConsentCheckResult.ACCEPTED)
        else:
            updated = opt_in_checker.update_result(ConsentCheckResult.DECLINED)
        if not updated:
            print("Could not update the consent file. No telemetry will be sent.")
            return

        telemetry = Telemetry(tid=tid, app_name=app_name, app_version=app_version, backend='ga4')

        # In order to prevent sending of duplicate events, after multiple run of opt_in_out --opt_in/--opt_out
        # we send opt_in event only if consent value is changed
        if new_opt_in_status:
            telemetry.backend.generate_new_cid_file()
            if prev_status != OptInStatus.ACCEPTED:
                telemetry.send_opt_in_event(OptInStatus.ACCEPTED, prev_status)
            print("You have successfully opted in to send the telemetry data.")
        else:
            if prev_status != OptInStatus.DECLINED:
                telemetry.send_opt_in_event(OptInStatus.DECLINED, prev_status, force_send=True)
            telemetry.backend.remove_cid_file()
            from .utils.stats_processor import StatsProcessor
            StatsProcessor().remove_stats_file()
            print("You have successfully opted out to send the telemetry data.")

    def send_opt_in_event(self, new_state: OptInStatus, prev_state: OptInStatus = OptInStatus.UNDEFINED,
                          label: str = "", force_send=False):
        """
        Sends opt-in event.

        :param new_state: new opt-in status.
        :param prev_state: previous opt-in status.
        :param label: the label with the information of opt-in status change.
        :param force_send: forces to send event ignoring the consent value
        :return: None
        """
        if new_state == OptInStatus.UNDEFINED:
            self.send_event("opt_in", "timer_reached", label, force_send=force_send)
        else:
            label = "{{prev_state:{}, new_state: {}}}".format(prev_state.value, new_state.value)
            self.send_event("opt_in", new_state.value, label, force_send=force_send)

    def get_stats(self):
        stats = StatsProcessor()
        file_exists, data = stats.get_stats()
        if not file_exists:
            created = stats.create_new_stats_file()
            data = {}
            if not created:
                return None
        if "usage_count" in data:
            usage_count = data["usage_count"]
            if usage_count < sys.maxsize:
                usage_count += 1
        else:
            usage_count = 1
        data["usage_count"] = usage_count
        stats.update_stats(data)
        return data

    @staticmethod
    def opt_in(tid: str):
        """
        Enables sending anonymous telemetry data.

        :param tid: ID of telemetry base.
        :return: None
        """
        Telemetry._update_opt_in_status(tid, True)

    @staticmethod
    def opt_out(tid: str):
        """
        Disables sending anonymous telemetry data.

        :param tid: ID of telemetry base.
        :return: None
        """
        Telemetry._update_opt_in_status(tid, False)

    @staticmethod
    def get_version():
        """
        Returns version of telemetry library.
        """
        return '2025.1.0'
