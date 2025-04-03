# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os

from .opt_in_checker import OptInChecker


def save_cid_to_file(file_name: str, cid: str):
    """
    Save the client ID to the specified file
    """
    try:
        # create directories recursively first
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w') as file:
            file.write(cid)
    except Exception as e:
        return False
    return True


def get_cid(file_name: str, validator: [callable, None]):
    """
    Get existing Client ID.
    :param file_name: name of the file with the client ID
    :param validator: the function to validate the client ID
    :return: existing client ID file
    """
    full_path = os.path.join(get_cid_path(), file_name)
    cid = None
    if os.path.exists(full_path):
        with open(full_path, 'r') as file:
            cid = file.readline().strip()
        if cid is not None and (validator is not None and not validator(cid)):
            cid = None
    return cid


def get_or_generate_cid(file_name: str, generator: callable, validator: [callable, None], old_name=None):
    """
    Get existing Client ID or generate a new one.
    :param file_name: name of the file with the client ID
    :param generator: the function to generate the client ID
    :param validator: the function to validate the client ID
    :param old_name: legacy name of the file with the client ID
    :return: existing or a new client ID file
    """
    cid = get_cid(file_name, validator)
    if cid is not None:
        return cid
    if old_name is not None:
        cid = get_cid(old_name, validator)
    full_path = os.path.join(get_cid_path(), file_name)
    if cid is None:
        cid = generator()
    save_cid_to_file(full_path, cid)
    return cid


def get_cid_path():
    """
    Returns a directory with the OpenVINO randomly generated client ID file.

    :return: the directory with the client ID file
    """
    return os.path.join(OptInChecker.consent_file_base_dir(), OptInChecker.consent_file_subdirectory())


def remove_cid_file(file_name: str):
    """
    Removes client ID file.
    :param file_name: name of the file with the client ID
    :return: None
    """
    cid_file = os.path.join(get_cid_path(), file_name)
    if os.path.exists(cid_file):
        if not os.access(cid_file, os.W_OK):
            log.warning("Failed to remove client ID file {}.".format(cid_file))
            return
        os.remove(cid_file)
