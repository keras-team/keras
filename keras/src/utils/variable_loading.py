"""
Utility functions for loading variables with sharded support.

This module provides common utilities for loading variables that may be sharded
across multiple devices, which is useful for distributed training scenarios.
"""

from absl import logging


def load_variable_with_sharded_support(variable, weight_data):
    """
    Load a variable safely, handling both sharded and non-sharded cases.

    This function automatically detects if a variable is sharded
    (has a '_layout' attribute) and uses the appropriate loading method
    to avoid OOM issues.

    Args:
        variable: The variable to load data into
        weight_data: The weight data to load (typically a numpy array)

    Returns:
        None
    """
    # Check if variable has a layout (is sharded)
    if hasattr(variable, "_layout") and variable._layout is not None:
        # Use _direct_assign for sharded variables to avoid OOM
        logging.info(
            f"Loading sharded variable ({variable.name}) with _direct_assign"
        )
        variable._direct_assign(weight_data)
        logging.info(f"Variable ({variable.name}) loaded successfully")
    else:
        # Use normal assign for non-sharded variables
        logging.info(
            f"Loading non-sharded variable ({variable.name}) with assign"
        )
        variable.assign(weight_data)
        logging.info(f"Variable ({variable.name}) loaded successfully")
