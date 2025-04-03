"""Determines if a contraction can use BLAS or not."""

from typing import List, Sequence, Tuple, Union

from opt_einsum.typing import ArrayIndexType

__all__ = ["can_blas"]


def can_blas(
    inputs: List[str],
    result: str,
    idx_removed: ArrayIndexType,
    shapes: Union[Sequence[Tuple[int]], None] = None,
) -> Union[str, bool]:
    """Checks if we can use a BLAS call.

    Parameters
    ----------
    inputs : list of str
        Specifies the subscripts for summation.
    result : str
        Resulting summation.
    idx_removed : set
        Indices that are removed in the summation
    shapes : sequence of tuple[int], optional
        If given, check also that none of the indices are broadcast dimensions.

    Returns:
    -------
    type : str or bool
        The type of BLAS call to be used or False if none.

    Notes:
    -----
    We assume several operations are not efficient such as a transposed
    DDOT, therefore 'ijk,jki->' should prefer einsum. These return the blas
    type appended with "/EINSUM" to differentiate when they can still be done
    with tensordot if required, e.g. when a backend has no einsum.

    Examples:
    --------
    >>> can_blas(['ij', 'jk'], 'ik', set('j'))
    'GEMM'

    >>> can_blas(['ijj', 'jk'], 'ik', set('j'))
    False

    >>> can_blas(['ab', 'cd'], 'abcd', set())
    'OUTER/EINSUM'

    >>> # looks like GEMM but actually 'j' is broadcast:
    >>> can_blas(['ij', 'jk'], 'ik', set('j'), shapes=[(4, 1), (5, 6)])
    False
    """
    # Can only do two
    if len(inputs) != 2:
        return False

    input_left, input_right = inputs

    for c in set(input_left + input_right):
        # can't deal with repeated indices on same input or more than 2 total
        nl, nr = input_left.count(c), input_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False

        # can't do implicit summation or dimension collapse e.g.
        #     "ab,bc->c" (implicitly sum over 'a')
        #     "ab,ca->ca" (take diagonal of 'a')
        if nl + nr - 1 == int(c in result):
            return False

    # check for broadcast indices e.g:
    #     "ij,jk->ik" (but one of the 'j' dimensions is broadcast up)
    if shapes is not None:
        for c in idx_removed:
            if shapes[0][input_left.find(c)] != shapes[1][input_right.find(c)]:
                return False

    # Prefer einsum if not removing indices
    #     (N.B. tensordot outer faster for large arrays?)
    if len(idx_removed) == 0:
        return "OUTER/EINSUM"

    # Build a few temporaries
    sets = [set(x) for x in inputs]
    keep_left = sets[0] - idx_removed
    keep_right = sets[1] - idx_removed
    rs = len(idx_removed)

    # DDOT
    if inputs[0] == inputs[1]:
        return "DOT"

    # DDOT does not make sense if you have to transpose - prefer einsum
    elif sets[0] == sets[1]:
        return "DOT/EINSUM"

    # GEMM no transpose
    if input_left[-rs:] == input_right[:rs]:
        return "GEMM"

    # GEMM transpose both
    elif input_left[:rs] == input_right[-rs:]:
        return "GEMM"

    # GEMM transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        return "GEMM"

    # GEMM transpose left
    elif input_left[:rs] == input_right[:rs]:
        return "GEMM"

    # Einsum is faster than vectordot if we have to copy
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        return "GEMV/EINSUM"

    # Conventional tensordot
    else:
        return "TDOT"
