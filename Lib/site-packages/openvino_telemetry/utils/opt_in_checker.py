# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os
import time
from enum import Enum
from pathlib import Path
from platform import system
from sys import stdin

from .colored_print import colored_print
from .input_with_timeout import input_with_timeout


class ConsentCheckResult(Enum):
    DECLINED = 0
    ACCEPTED = 1
    NO_FILE = 2


class DialogResult(Enum):
    DECLINED = 0
    ACCEPTED = 1
    TIMEOUT_REACHED = 2


class OptInChecker:
    dialog_timeout = 50  # seconds
    path_to_opt_in_out_script = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    opt_in_out_script_name = "opt_in_out"
    doc_link = "docs.openvino.ai"
    opt_in_out_script_run_command = "\'{} --opt_out\'".format(opt_in_out_script_name)
    opt_in_question = "Intel would like your permission to collect software performance and usage data for the " \
                      "purpose of improving Intel products and services. This data will be collected directly " \
                      "by Intel or through the use of Google Analytics. This data will be stored in countries " \
                      "where Intel or Google operate. Intel operates around the world and your usage data will " \
                      "be sent to Intel in the United States or other countries.\nYou can opt-out at any time " \
                      "in the future by running {}.\n" \
                      "More Information is available at {}.\n" \
                      "Please type ‘Y’ to give your consent or ‘N’ to decline.".format(opt_in_out_script_run_command,
                                                                                       doc_link)
    opt_in_question_incorrect_input = "Please type ‘Y’ to give your consent or ‘N’ to decline."
    response_confirmation_accept = "The selected option was to collect telemetry data."
    response_confirmation_decline = "The selected option was NOT to collect telemetry data."
    response_confirmation_timer_reached = "The timer has expired and no data will be collected."

    @staticmethod
    def _ask_opt_in(question: str, timeout: int):
        """
        Runs input with timeout and checks user input.
        :param question: question that will be printed on the screen.
        :param timeout: timeout to wait.
        :return: opt-in dialog result.
        """
        colored_print(question)
        answer = input_with_timeout(prompt='>>', timeout=timeout)
        answer = answer.lower().strip()
        if answer == "n" or answer == "no":
            colored_print(OptInChecker.response_confirmation_decline)
            return DialogResult.DECLINED
        if answer == "y" or answer == "yes":
            colored_print(OptInChecker.response_confirmation_accept)
            return DialogResult.ACCEPTED
        return DialogResult.TIMEOUT_REACHED

    def opt_in_dialog(self):
        """
        Runs opt-in dialog until the timeout is expired.
        :return: opt-in dialog result.
        """
        start_time = time.time()
        answer = self._ask_opt_in(self.opt_in_question, self.dialog_timeout)
        time_passed = time.time() - start_time
        while time_passed < self.dialog_timeout and answer == DialogResult.TIMEOUT_REACHED:
            answer = self._ask_opt_in(self.opt_in_question_incorrect_input, self.dialog_timeout - time_passed)
            time_passed = time.time() - start_time

        if answer == DialogResult.TIMEOUT_REACHED:
            colored_print(OptInChecker.response_confirmation_timer_reached)

        return answer

    @staticmethod
    def consent_file_base_dir():
        """
        Returns the base directory of the consent file.
        :return: base directory of the consent file.
        """
        platform = system()

        dir_to_check = None

        if platform == 'Windows':
            dir_to_check = '$LOCALAPPDATA'
        elif platform in ['Linux', 'Darwin']:
            dir_to_check = Path.home()

        if dir_to_check is None:
            log.info('Failed to find location of the openvino_telemetry file.')
            return None

        consent_base_dir = os.path.expandvars(dir_to_check)
        if not os.path.isdir(consent_base_dir):
            log.info('Failed to find location of the openvino_telemetry file.')
            return None

        return consent_base_dir

    @staticmethod
    def consent_file_subdirectory():
        """
        Returns consent file subdirectory.
        :return: consent file subdirectory.
        """
        platform = system()
        if platform == 'Windows':
            return 'Intel Corporation'
        elif platform in ['Linux', 'Darwin']:
            return 'intel'
        log.info('Failed to find location of the openvino_telemetry file.')
        return None

    def consent_file(self):
        """
        Returns the consent file path.
        :return: consent file path.
        """
        return os.path.join(self.consent_file_base_dir(), self.consent_file_subdirectory(), "openvino_telemetry")

    def create_new_consent_file(self):
        """
        Creates a new consent file.
        :return: True if the file is created successfully, otherwise False
        """
        if not self.create_or_check_consent_dir():
            return False
        try:
            open(self.consent_file(), 'w').close()
        except Exception:
            return False
        return True

    def create_or_check_consent_dir(self):
        """
        Creates consent file directory and checks if the directory is writable.
        :return: True if the directory is created and writable, otherwise False
        """
        base_dir = self.consent_file_base_dir()
        consent_file_subdirectory = self.consent_file_subdirectory()
        if base_dir is None or consent_file_subdirectory is None:
            return False
        base_is_dir = os.path.isdir(base_dir)
        base_dir_exists = os.path.exists(base_dir)
        base_w_access = os.access(base_dir, os.W_OK)

        if not base_dir_exists or not base_is_dir:
            return False
        if not base_w_access:
            log.warning("Failed to create openvino_telemetry file. "
                        "Please allow write access to the following directory: {}".format(base_dir))
            return False

        consent_file_dir = os.path.join(base_dir, consent_file_subdirectory)
        consent_file_is_dir = os.path.isdir(consent_file_dir)
        consent_file_dir_exists = os.path.exists(consent_file_dir)

        # If consent path exists and it is not directory, we try to remove it
        if consent_file_dir_exists and not consent_file_is_dir:
            try:
                os.remove(consent_file_dir)
            except:
                log.warning("Unable to create directory for openvino_telemetry file, as {} is invalid directory.".format(consent_file_dir))
                return False

        if not os.path.exists(consent_file_dir):
            try:
                os.mkdir(consent_file_dir)

                # check that directory is created
                if not os.path.exists(consent_file_dir):
                    return False
            except Exception as e:
                log.warning("Failed to create directory for openvino_telemetry file: {}".format(str(e)))
                return False

        consent_file_w_access = os.access(consent_file_dir, os.W_OK)
        if not consent_file_w_access:
            log.warning("Failed to create openvino_telemetry file. "
                        "Please allow write access to the following directory: {}".format(consent_file_dir))
            return False
        return True

    def update_result(self, result: ConsentCheckResult):
        """
        Updates the 'opt_in' value in the consent file.
        :param result: opt-in dialog result.
        :return: False if the consent file is not writable, otherwise True
        """
        if self.consent_file_base_dir() is None or self.consent_file_subdirectory() is None:
            return False
        if not os.path.exists(self.consent_file()):
            if not self.create_new_consent_file():
                return False
        if not os.access(self.consent_file(), os.W_OK):
            log.warning("Failed to update opt-in status. "
                        "Please allow write access to the following file: {}".format(self.consent_file()))
            return False
        try:
            with open(self.consent_file(), 'w') as file:
                if result == ConsentCheckResult.ACCEPTED:
                    file.write("1")
                else:
                    file.write("0")
        except Exception:
            return False
        return True

    def consent_file_is_empty(self):
        """
        Checks if the consent file is empty.
        :return: True if consent file is empty, otherwise False.
        """
        if os.stat(self.consent_file()).st_size == 0:
            return True
        return False

    def get_info_from_consent_file(self):
        """
        Gets information from consent file.
        :return: the tuple, where the first element is True if the file is read successfully, otherwise False
        and the second element is the content of the consent file.
        """
        if not os.access(self.consent_file(), os.R_OK):
            return False, {}
        try:
            with open(self.consent_file(), 'r') as file:
                content = file.readline().strip()
        except Exception:
            return False, {}
        return True, content

    @staticmethod
    def _check_input_is_terminal():
        """
        Checks if stdin is terminal.
        :return: True if stdin is terminal, otherwise False
        """
        return stdin.isatty()

    @staticmethod
    def _check_main_process():
        platform = system()
        if platform == 'Windows':
            # In Windows 'os' module does not have getpid() and getsid(),
            # so the following checks are not applicable.
            # Subprocess check in Windows is handled by self._check_input_is_terminal(),
            # which does not work for Unix subprocesses.
            return True

        try:
            # Check that current process is the leader of process group
            if os.getpid() != os.getpgid(0):
                return False

            # Check that parent process is in same session as current process
            if os.getsid(os.getppid()) != os.getsid(0):
                return False
        except:
            # If we couldn't check main process, disable opt-in dialog
            return False
        return True

    @staticmethod
    def _check_run_in_notebook():
        """
        Checks that script is executed in Jupyter Notebook.
        :return: True script is executed in Jupyter Notebook, otherwise False
        """
        try:
            return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        except NameError:
            pass
        return False

    @staticmethod
    def _run_in_ci():
        """
        Checks that script is executed in CI job.
        :return: True if script is executed in CI job, otherwise False
        """
        if "CI" in os.environ and os.environ["CI"].lower() == "true":
            return True

        if "TF_BUILD" in os.environ and len(os.environ["TF_BUILD"]):
            return True

        if "JENKINS_URL" in os.environ and len(os.environ["JENKINS_URL"]):
            return True

        return False

    def check(self, enable_opt_in_dialog, disable_in_ci=False):
        """
        Checks if user has accepted the collection of the information by checking the consent file.
        :return: consent check result
        """
        if self.consent_file_base_dir() is None or self.consent_file_subdirectory() is None:
            return ConsentCheckResult.DECLINED

        if disable_in_ci and self._run_in_ci():
            return ConsentCheckResult.DECLINED

        if not os.path.exists(self.consent_file()):
            if enable_opt_in_dialog:
                if not self._check_main_process():
                    return ConsentCheckResult.DECLINED

                if not self._check_input_is_terminal() or self._check_run_in_notebook():
                    return ConsentCheckResult.DECLINED
            return ConsentCheckResult.NO_FILE

        if not self.consent_file_is_empty():
            _, content = self.get_info_from_consent_file()
            if content == "1":
                return ConsentCheckResult.ACCEPTED
            elif content == "0":
                return ConsentCheckResult.DECLINED
        log.warning("Incorrect format of the file with opt-in status.")
        return ConsentCheckResult.DECLINED
