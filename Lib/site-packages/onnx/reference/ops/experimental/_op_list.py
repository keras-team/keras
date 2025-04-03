# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import textwrap
from typing import Any

from onnx.reference.op_run import OpFunction
from onnx.reference.ops._helpers import build_registered_operators_any_domain
from onnx.reference.ops.experimental._op_run_experimental import OpRunExperimental
from onnx.reference.ops.experimental.op_im2col import Im2Col  # noqa: F401


def _build_registered_operators() -> dict[str, dict[int | None, OpRunExperimental]]:
    return build_registered_operators_any_domain(globals().copy())  # type: ignore[return-value]


def load_op(
    domain: str,
    op_type: str,
    version: None | int,
    custom: Any = None,
    evaluator_cls: type | None = None,  # noqa: ARG001
) -> Any:
    """Loads the implemented for a specified operator.

    Args:
        domain: domain
        op_type: oprator type
        version: requested version
        custom: custom implementation (like a function)
        evaluator_cls: unused

    Returns:
        class
    """
    global _registered_operators  # noqa: PLW0603
    if _registered_operators is None:
        _registered_operators = _build_registered_operators()  # type: ignore[assignment]
    if custom is not None:
        return lambda *args: OpFunction(*args, impl=custom)  # type: ignore
    if domain != "experimental":
        raise ValueError(f"Domain must be '' not {domain!r}.")
    if op_type not in _registered_operators:  # type: ignore
        available = "\n".join(textwrap.wrap(", ".join(sorted(_registered_operators))))  # type: ignore
        raise NotImplementedError(
            f"No registered implementation for operator {op_type!r} "
            f"and domain {domain!r} in\n{available}"
        )
    impl = _registered_operators[op_type]  # type: ignore
    if None not in impl:
        raise RuntimeError(
            f"No default implementation for operator {op_type!r} "
            f"and domain {domain!r}, found "
            f"{', '.join(map(str, impl))}."
        )
    if version is None or len(impl) == 1:
        cl = impl[None]
    else:
        best = -1
        for v in impl:
            if v is None:
                continue
            if best < v <= version:
                best = v
        if best == -1:
            raise RuntimeError(
                f"No implementation for operator {op_type!r} "
                f"domain {domain!r} and version {version!r}, found "
                f"{', '.join(map(str, impl))}."
            )
        cl = impl[best]
    if cl is None:
        available = "\n".join(textwrap.wrap(", ".join(sorted(_registered_operators))))  # type: ignore
        raise ValueError(
            f"Not registered implementation for operator {op_type!r}, "
            f"domain {domain!r}, and {version!r} in\n{available}"
        )
    return cl


_registered_operators: dict[str, dict[int | None, OpRunExperimental]] | None = None
