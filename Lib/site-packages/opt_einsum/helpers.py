"""Contains helper functions for opt_einsum testing scripts."""

from typing import Any, Collection, Dict, FrozenSet, Iterable, List, Tuple, overload

from opt_einsum.typing import ArrayIndexType, ArrayType

__all__ = ["compute_size_by_dict", "find_contraction", "flop_count"]

_valid_chars = "abcdefghijklmopqABC"
_sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4]
_default_dim_dict = dict(zip(_valid_chars, _sizes))


@overload
def compute_size_by_dict(indices: Iterable[int], idx_dict: List[int]) -> int: ...


@overload
def compute_size_by_dict(indices: Collection[str], idx_dict: Dict[str, int]) -> int: ...


def compute_size_by_dict(indices: Any, idx_dict: Any) -> int:
    """Computes the product of the elements in indices based on the dictionary
    idx_dict.

    Parameters
    ----------
    indices : iterable
        Indices to base the product on.
    idx_dict : dictionary
        Dictionary of index _sizes

    Returns:
    -------
    ret : int
        The resulting product.

    Examples:
    --------
    >>> compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
    90

    """
    ret = 1
    for i in indices:  # lgtm [py/iteration-string-and-sequence]
        ret *= idx_dict[i]
    return ret


def find_contraction(
    positions: Collection[int],
    input_sets: List[ArrayIndexType],
    output_set: ArrayIndexType,
) -> Tuple[FrozenSet[str], List[ArrayIndexType], ArrayIndexType, ArrayIndexType]:
    """Finds the contraction for a given set of input and output sets.

    Parameters
    ----------
    positions : iterable
        Integer positions of terms used in the contraction.
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript

    Returns:
    -------
    new_result : set
        The indices of the resulting contraction
    remaining : list
        List of sets that have not been contracted, the new set is appended to
        the end of this list
    idx_removed : set
        Indices removed from the entire contraction
    idx_contraction : set
        The indices used in the current contraction

    Examples:
    --------
    # A simple dot product test case
    >>> pos = (0, 1)
    >>> isets = [set('ab'), set('bc')]
    >>> oset = set('ac')
    >>> find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})

    # A more complex case with additional terms in the contraction
    >>> pos = (0, 2)
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('ac')
    >>> find_contraction(pos, isets, oset)
    ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})
    """
    remaining = list(input_sets)
    inputs = (remaining.pop(i) for i in sorted(positions, reverse=True))
    idx_contract = frozenset.union(*inputs)
    idx_remain = output_set.union(*remaining)

    new_result = idx_remain & idx_contract
    idx_removed = idx_contract - new_result
    remaining.append(new_result)

    return new_result, remaining, idx_removed, idx_contract


def flop_count(
    idx_contraction: Collection[str],
    inner: bool,
    num_terms: int,
    size_dictionary: Dict[str, int],
) -> int:
    """Computes the number of FLOPS in the contraction.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction

    Returns:
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples:
    --------
    >>> flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
    30

    >>> flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
    60

    """
    overall_size = compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1

    return overall_size * op_factor


def has_array_interface(array: ArrayType) -> ArrayType:
    if hasattr(array, "__array_interface__"):
        return True
    else:
        return False
