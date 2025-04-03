# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino exceptions hierarchy. All exceptions are descendants of OVError."""


class OVError(Exception):
    """Base class for OV exceptions."""


class UserInputError(OVError):
    """User provided unexpected input."""


class OVTypeError(OVError, TypeError):
    """Type mismatch error."""
