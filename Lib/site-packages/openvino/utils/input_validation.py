# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for validating user input."""

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np

from openvino.exceptions import UserInputError

log = logging.getLogger(__name__)


def assert_list_of_ints(value_list: Iterable[int], message: str) -> None:
    """Verify that the provided value is an iterable of integers."""
    try:
        for value in value_list:
            if not isinstance(value, int):
                raise TypeError
    except TypeError:
        log.warning(message)
        raise UserInputError(message, value_list)


def _check_value(op_name, attr_key, value, val_type, cond=None):
    # type: (str, str, Any, Type, Optional[Callable[[Any], bool]]) -> bool
    """Check whether provided value satisfies specified criteria.

    :param      op_name:        The operator name which attributes are checked.
    :param      attr_key:       The attribute name.
    :param      value:          The value to check.
    :param      val_type:       Required value type.
    :param      cond:           The optional function running additional checks.

    :raises     UserInputError:

    returns:    True if attribute satisfies all criterias. Otherwise False.
    """
    if not np.issubdtype(type(value), val_type):
        raise UserInputError(
            f'{op_name} operator attribute "{attr_key}" value must by of type {val_type}.',
        )
    if cond is not None and not cond(value):
        raise UserInputError(
            f'{op_name} operator attribute "{attr_key}" value does not satisfy provided condition.',
        )
    return True


def check_valid_attribute(op_name, attr_dict, attr_key, val_type, cond=None, required=False):
    # type: (str, dict, str, Type, Optional[Callable[[Any], bool]], Optional[bool]) -> bool
    """Check whether specified attribute satisfies given criteria.

    :param  op_name:    The operator name which attributes are checked.
    :param attr_dict:   Dictionary containing key-value attributes to check.
    :param attr_key:    Key value for validated attribute.
    :param val_type:    Value type for validated attribute.
    :param cond:        Any callable wich accept attribute value and returns True or False.
    :param required:    Whether provided attribute key is not required. This mean it may be missing
                        from provided dictionary.

    :raises     UserInputError:

    returns True if attribute satisfies all criterias. Otherwise False.
    """
    result = True

    if required and attr_key not in attr_dict:
        raise UserInputError(
            f'Provided dictionary is missing {op_name} operator required attribute "{attr_key}"',
        )

    if attr_key not in attr_dict:
        return result

    attr_value = attr_dict[attr_key]

    if np.isscalar(attr_value):
        result = result and _check_value(op_name, attr_key, attr_value, val_type, cond)
    else:
        for value in attr_value:
            result = result and _check_value(op_name, attr_key, value, val_type, cond)

    return result


def check_valid_attributes(
    op_name,  # type: str
    attributes,  # type: Dict[str, Any]
    requirements,  # type: List[Tuple[str, bool, Type, Optional[Callable]]]
):
    # type: (...) -> bool
    """Perform attributes validation according to specified type, value criteria.

    :param  op_name:        The operator name which attributes are checked.
    :param  attributes:     The dictionary with user provided attributes to check.
    :param  requirements:   The list of tuples describing attributes' requirements. The tuple should
                            contain following values:
                            (attr_name: str,
                            is_required: bool,
                            value_type: Type,
                            value_condition: Callable)

    :raises     UserInputError:

    :returns True if all attributes satisfies criterias. Otherwise False.
    """
    for attr, required, val_type, cond in requirements:
        check_valid_attribute(op_name, attributes, attr, val_type, cond, required)
    return True


def is_positive_value(value):  # type: (Any) -> bool
    """Determine whether the specified x is positive value.

    :param      value:    The value to check.

    returns   True if the specified x is positive value, False otherwise.
    """
    return value > 0


def is_non_negative_value(value):  # type: (Any) -> bool
    """Determine whether the specified x is non-negative value.

    :param      value:    The value to check.

    returns   True if the specified x is non-negative value, False otherwise.
    """
    return value >= 0
