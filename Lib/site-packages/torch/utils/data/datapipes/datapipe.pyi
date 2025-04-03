# mypy: allow-untyped-defs
# This base template ("datapipe.pyi.in") is generated from mypy stubgen with minimal editing for code injection
# The output file will be "datapipe.pyi". This is executed as part of torch/CMakeLists.txt
# Note that, for mypy, .pyi file takes precedent over .py file, such that we must define the interface for other
# classes/objects here, even though we are not injecting extra code into them at the moment.

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)

from torch.utils.data import Dataset, default_collate, IterableDataset
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
UNTRACABLE_DATAFRAME_PIPES: Any

class DataChunk(List[_T]):
    items: List[_T]
    def __init__(self, items: Iterable[_T]) -> None: ...
    def as_str(self, indent: str = "") -> str: ...
    def __iter__(self) -> Iterator[_T]: ...
    def raw_iterator(self) -> Iterator[_T]: ...

class MapDataPipe(Dataset[_T_co], metaclass=_DataPipeMeta):
    functions: Dict[str, Callable] = ...
    reduce_ex_hook: Optional[Callable] = ...
    getstate_hook: Optional[Callable] = ...
    str_hook: Optional[Callable] = ...
    repr_hook: Optional[Callable] = ...
    def __getattr__(self, attribute_name: Any): ...
    @classmethod
    def register_function(cls, function_name: Any, function: Any) -> None: ...
    @classmethod
    def register_datapipe_as_function(
        cls,
        function_name: Any,
        cls_to_register: Any,
    ): ...
    def __getstate__(self): ...
    def __reduce_ex__(self, *args: Any, **kwargs: Any): ...
    @classmethod
    def set_getstate_hook(cls, hook_fn: Any) -> None: ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn: Any) -> None: ...
    # Functional form of 'BatcherMapDataPipe'
    def batch(self, batch_size: int, drop_last: bool = False, wrapper_class: Type[DataChunk] = DataChunk) -> MapDataPipe:
        r"""
        Create mini-batches of data (functional name: ``batch``).
    
        An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``,
        or ``length % batch_size`` for the last batch if ``drop_last`` is set to ``False``.
    
        Args:
            datapipe: Iterable DataPipe being batched
            batch_size: The size of each batch
            drop_last: Option to drop the last batch if it's not full
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.map import SequenceWrapper
            >>> dp = SequenceWrapper(range(10))
            >>> batch_dp = dp.batch(batch_size=2)
            >>> list(batch_dp)
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        """
    
    # Functional form of 'ConcaterMapDataPipe'
    def concat(self, *datapipes: MapDataPipe) -> MapDataPipe:
        r"""
        Concatenate multiple Map DataPipes (functional name: ``concat``).
    
        The new index of is the cumulative sum of source DataPipes.
        For example, if there are 2 source DataPipes both with length 5,
        index 0 to 4 of the resulting `ConcatMapDataPipe` would refer to
        elements of the first DataPipe, and 5 to 9 would refer to elements
        of the second DataPipe.
    
        Args:
            datapipes: Map DataPipes being concatenated
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.map import SequenceWrapper
            >>> dp1 = SequenceWrapper(range(3))
            >>> dp2 = SequenceWrapper(range(3))
            >>> concat_dp = dp1.concat(dp2)
            >>> list(concat_dp)
            [0, 1, 2, 0, 1, 2]
        """
    
    # Functional form of 'MapperMapDataPipe'
    def map(self, fn: Callable= ...) -> MapDataPipe:
        r"""
        Apply the input function over each item from the source DataPipe (functional name: ``map``).
    
        The function can be any regular Python function or partial object. Lambda
        function is not recommended as it is not supported by pickle.
    
        Args:
            datapipe: Source MapDataPipe
            fn: Function being applied to each item
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
            >>> def add_one(x):
            ...     return x + 1
            >>> dp = SequenceWrapper(range(10))
            >>> map_dp_1 = dp.map(add_one)
            >>> list(map_dp_1)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
            >>> list(map_dp_2)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        """
    
    # Functional form of 'ShufflerIterDataPipe'
    def shuffle(self, *, indices: Optional[List] = None) -> IterDataPipe:
        r"""
        Shuffle the input MapDataPipe via its indices (functional name: ``shuffle``).
    
        When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
        set up random seed are different based on :attr:`num_workers`.
    
        For single-process mode (:attr:`num_workers == 0`), the random seed is set before
        the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
        mode (:attr:`num_worker > 0`), ``worker_init_fn`` is used to set up a random seed
        for each worker process.
    
        Args:
            datapipe: MapDataPipe being shuffled
            indices: a list of indices of the MapDataPipe. If not provided, we assume it uses 0-based indexing
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.map import SequenceWrapper
            >>> dp = SequenceWrapper(range(10))
            >>> shuffle_dp = dp.shuffle().set_seed(0)
            >>> list(shuffle_dp)
            [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
            >>> list(shuffle_dp)
            [6, 1, 9, 5, 2, 4, 7, 3, 8, 0]
            >>> # Reset seed for Shuffler
            >>> shuffle_dp = shuffle_dp.set_seed(0)
            >>> list(shuffle_dp)
            [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
    
        Note:
            Even thought this ``shuffle`` operation takes a ``MapDataPipe`` as the input, it would return an
            ``IterDataPipe`` rather than a ``MapDataPipe``, because ``MapDataPipe`` should be non-sensitive to
            the order of data order for the sake of random reads, but ``IterDataPipe`` depends on the order
            of data during data-processing.
        """
    
    # Functional form of 'ZipperMapDataPipe'
    def zip(self, *datapipes: MapDataPipe[_T_co]) -> MapDataPipe:
        r"""
        Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).
    
        This MataPipe is out of bound as soon as the shortest input DataPipe is exhausted.
    
        Args:
            *datapipes: Map DataPipes being aggregated
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.map import SequenceWrapper
            >>> dp1 = SequenceWrapper(range(3))
            >>> dp2 = SequenceWrapper(range(10, 13))
            >>> zip_dp = dp1.zip(dp2)
            >>> list(zip_dp)
            [(0, 10), (1, 11), (2, 12)]
        """
    

class IterDataPipe(IterableDataset[_T_co], metaclass=_IterDataPipeMeta):
    functions: Dict[str, Callable] = ...
    reduce_ex_hook: Optional[Callable] = ...
    getstate_hook: Optional[Callable] = ...
    str_hook: Optional[Callable] = ...
    repr_hook: Optional[Callable] = ...
    _number_of_samples_yielded: int = ...
    _snapshot_state: _SnapshotState = _SnapshotState.Iterating  # noqa: PYI015
    _fast_forward_iterator: Optional[Iterator] = ...
    def __getattr__(self, attribute_name: Any): ...
    @classmethod
    def register_function(cls, function_name: Any, function: Any) -> None: ...
    @classmethod
    def register_datapipe_as_function(
        cls,
        function_name: Any,
        cls_to_register: Any,
        enable_df_api_tracing: bool = ...,
    ): ...
    def __getstate__(self): ...
    def __reduce_ex__(self, *args: Any, **kwargs: Any): ...
    @classmethod
    def set_getstate_hook(cls, hook_fn: Any) -> None: ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn: Any) -> None: ...
    # Functional form of 'BatcherIterDataPipe'
    def batch(self, batch_size: int, drop_last: bool = False, wrapper_class: Type[DataChunk] = DataChunk) -> IterDataPipe:
        r"""
        Creates mini-batches of data (functional name: ``batch``).
    
        An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``, or ``length % batch_size`` for the
        last batch if ``drop_last`` is set to ``False``.
    
        Args:
            datapipe: Iterable DataPipe being batched
            batch_size: The size of each batch
            drop_last: Option to drop the last batch if it's not full
            wrapper_class: wrapper to apply onto each batch (type ``List``) before yielding,
                defaults to ``DataChunk``
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(range(10))
            >>> dp = dp.batch(batch_size=3, drop_last=True)
            >>> list(dp)
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        """
    
    # Functional form of 'CollatorIterDataPipe'
    def collate(self, conversion: Union[Callable[..., Any], Dict[Union[str, Any], Union[Callable, Any]], None] = default_collate, collate_fn: Optional[Callable] = None) -> IterDataPipe:
        r"""
        Collates samples from DataPipe to Tensor(s) by a custom collate function (functional name: ``collate``).
    
        By default, it uses :func:`torch.utils.data.default_collate`.
    
        .. note::
            While writing a custom collate function, you can import :func:`torch.utils.data.default_collate` for the
            default behavior and `functools.partial` to specify any additional arguments.
    
        Args:
            datapipe: Iterable DataPipe being collated
            collate_fn: Customized collate function to collect and combine data or a batch of data.
                Default function collates to Tensor(s) based on data type.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> # Convert integer data to float Tensor
            >>> class MyIterDataPipe(torch.utils.data.IterDataPipe):
            ...     def __init__(self, start, end):
            ...         super(MyIterDataPipe).__init__()
            ...         assert end > start, "this example code only works with end >= start"
            ...         self.start = start
            ...         self.end = end
            ...
            ...     def __iter__(self):
            ...         return iter(range(self.start, self.end))
            ...
            ...     def __len__(self):
            ...         return self.end - self.start
            ...
            >>> ds = MyIterDataPipe(start=3, end=7)
            >>> print(list(ds))
            [3, 4, 5, 6]
            >>> def collate_fn(batch):
            ...     return torch.tensor(batch, dtype=torch.float)
            ...
            >>> collated_ds = CollateIterDataPipe(ds, collate_fn=collate_fn)
            >>> print(list(collated_ds))
            [tensor(3.), tensor(4.), tensor(5.), tensor(6.)]
        """
    
    # Functional form of 'ConcaterIterDataPipe'
    def concat(self, *datapipes: IterDataPipe) -> IterDataPipe:
        r"""
        Concatenates multiple Iterable DataPipes (functional name: ``concat``).
    
        The resulting DataPipe will yield all the elements from the first input DataPipe, before yielding from the subsequent ones.
    
        Args:
            datapipes: Iterable DataPipes being concatenated
    
        Example:
            >>> # xdoctest: +REQUIRES(module:torchdata)
            >>> import random
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp1 = IterableWrapper(range(3))
            >>> dp2 = IterableWrapper(range(5))
            >>> list(dp1.concat(dp2))
            [0, 1, 2, 0, 1, 2, 3, 4]
        """
    
    # Functional form of 'DemultiplexerIterDataPipe'
    def demux(self, num_instances: int, classifier_fn: Callable[[_T_co], Optional[int]], drop_none: bool = False, buffer_size: int = 1000) -> List[IterDataPipe]:
        r"""
        Splits the input DataPipe into multiple child DataPipes, using the given classification function (functional name: ``demux``).
    
        A list of the child DataPipes is returned from this operation.
    
        Args:
            datapipe: Iterable DataPipe being filtered
            num_instances: number of instances of the DataPipe to create
            classifier_fn: a function that maps values to an integer within the range ``[0, num_instances - 1]`` or ``None``
            drop_none: defaults to ``False``, if ``True``, the function will skip over elements classified as ``None``
            buffer_size: this defines the maximum number of inputs that the buffer can hold across all child
                DataPipes while waiting for their values to be yielded.
                Defaults to ``1000``. Use ``-1`` for the unlimited buffer.
    
        Examples:
            >>> # xdoctest: +REQUIRES(module:torchdata)
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> def odd_or_even(n):
            ...     return n % 2
            >>> source_dp = IterableWrapper(range(5))
            >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even)
            >>> list(dp1)
            [0, 2, 4]
            >>> list(dp2)
            [1, 3]
            >>> # It can also filter out any element that gets `None` from the `classifier_fn`
            >>> def odd_or_even_no_zero(n):
            ...     return n % 2 if n != 0 else None
            >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even_no_zero, drop_none=True)
            >>> list(dp1)
            [2, 4]
            >>> list(dp2)
            [1, 3]
        """
    
    # Functional form of 'FilterIterDataPipe'
    def filter(self, filter_fn: Callable, input_col=None) -> IterDataPipe:
        r"""
        Filters out elements from the source datapipe according to input ``filter_fn`` (functional name: ``filter``).
    
        Args:
            datapipe: Iterable DataPipe being filtered
            filter_fn: Customized function mapping an element to a boolean.
            input_col: Index or indices of data which ``filter_fn`` is applied, such as:
    
                - ``None`` as default to apply ``filter_fn`` to the data directly.
                - Integer(s) is used for list/tuple.
                - Key(s) is used for dict.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> def is_even(n):
            ...     return n % 2 == 0
            >>> dp = IterableWrapper(range(5))
            >>> filter_dp = dp.filter(filter_fn=is_even)
            >>> list(filter_dp)
            [0, 2, 4]
        """
    
    # Functional form of 'ForkerIterDataPipe'
    def fork(self, num_instances: int, buffer_size: int = 1000, copy: Optional[Literal["shallow", "deep"]] = None) -> List[IterDataPipe]:
        r"""
        Creates multiple instances of the same Iterable DataPipe (functional name: ``fork``).
    
        Args:
            datapipe: Iterable DataPipe being copied
            num_instances: number of instances of the datapipe to create
            buffer_size: this restricts how far ahead the leading child DataPipe
               can read relative to the slowest child DataPipe.
               Defaults to ``1000``. Use ``-1`` for the unlimited buffer.
            copy: copy strategy to use for items yielded by each branch. Supported
                options are ``None`` for no copying, ``"shallow"`` for shallow object
                copies, and ``"deep"`` for deep object copies. Defaults to ``None``.
    
        Note:
            All branches of the forked pipeline return the identical object unless
            the copy parameter is supplied. If the object is mutable or contains
            mutable objects, changing them in one branch will affect all others.
    
        Example:
            >>> # xdoctest: +REQUIRES(module:torchdata)
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper(range(5))
            >>> dp1, dp2 = source_dp.fork(num_instances=2)
            >>> list(dp1)
            [0, 1, 2, 3, 4]
            >>> list(dp2)
            [0, 1, 2, 3, 4]
        """
    
    # Functional form of 'GrouperIterDataPipe'
    def groupby(self, group_key_fn: Callable[[_T_co], Any], *, keep_key: bool = False, buffer_size: int = 10000, group_size: Optional[int] = None, guaranteed_group_size: Optional[int] = None, drop_remaining: bool = False) -> IterDataPipe:
        r"""
        Groups data from IterDataPipe by keys from ``group_key_fn``, yielding a ``DataChunk`` with batch size up to ``group_size``.
    
        (functional name: ``groupby``).
    
        The samples are read sequentially from the source ``datapipe``, and a batch of samples belonging to the same group
        will be yielded as soon as the size of the batch reaches ``group_size``. When the buffer is full,
        the DataPipe will yield the largest batch with the same key, provided that its size is larger
        than ``guaranteed_group_size``. If its size is smaller, it will be dropped if ``drop_remaining=True``.
    
        After iterating through the entirety of source ``datapipe``, everything not dropped due to the buffer capacity
        will be yielded from the buffer, even if the group sizes are smaller than ``guaranteed_group_size``.
    
        Args:
            datapipe: Iterable datapipe to be grouped
            group_key_fn: Function used to generate group key from the data of the source datapipe
            keep_key: Option to yield the matching key along with the items in a tuple,
                resulting in `(key, [items])` otherwise returning [items]
            buffer_size: The size of buffer for ungrouped data
            group_size: The max size of each group, a batch is yielded as soon as it reaches this size
            guaranteed_group_size: The guaranteed minimum group size to be yielded in case the buffer is full
            drop_remaining: Specifies if the group smaller than ``guaranteed_group_size`` will be dropped from buffer
                when the buffer is full
    
        Example:
            >>> import os
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> def group_fn(file):
            ...     return os.path.basename(file).split(".")[0]
            >>> source_dp = IterableWrapper(["a.png", "b.png", "a.json", "b.json", "a.jpg", "c.json"])
            >>> dp0 = source_dp.groupby(group_key_fn=group_fn)
            >>> list(dp0)
            [['a.png', 'a.json', 'a.jpg'], ['b.png', 'b.json'], ['c.json']]
            >>> # A group is yielded as soon as its size equals to `group_size`
            >>> dp1 = source_dp.groupby(group_key_fn=group_fn, group_size=2)
            >>> list(dp1)
            [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
            >>> # Scenario where `buffer` is full, and group 'a' needs to be yielded since its size > `guaranteed_group_size`
            >>> dp2 = source_dp.groupby(group_key_fn=group_fn, buffer_size=3, group_size=3, guaranteed_group_size=2)
            >>> list(dp2)
            [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
        """
    
    # Functional form of 'FileListerIterDataPipe'
    def list_files(self, masks: Union[str, List[str]] = "", *, recursive: bool = False, abspath: bool = False, non_deterministic: bool = False, length: int = -1) -> IterDataPipe:
        r"""
        Given path(s) to the root directory, yields file pathname(s) (path + filename) of files within the root directory.
    
        Multiple root directories can be provided (functional name: ``list_files``).
    
        Args:
            root: Root directory or a sequence of root directories
            masks: Unix style filter string or string list for filtering file name(s)
            recursive: Whether to return pathname from nested directories or not
            abspath: Whether to return relative pathname or absolute pathname
            non_deterministic: Whether to return pathname in sorted order or not.
                If ``False``, the results yielded from each root directory will be sorted
            length: Nominal length of the datapipe
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import FileLister
            >>> dp = FileLister(root=".", recursive=True)
            >>> list(dp)
            ['example.py', './data/data.tar']
        """
    
    # Functional form of 'MapperIterDataPipe'
    def map(self, fn: Callable, input_col=None, output_col=None) -> IterDataPipe:
        r"""
        Applies a function over each item from the source DataPipe (functional name: ``map``).
    
        The function can be any regular Python function or partial object. Lambda
        function is not recommended as it is not supported by pickle.
    
        Args:
            datapipe: Source Iterable DataPipe
            fn: Function being applied over each item
            input_col: Index or indices of data which ``fn`` is applied, such as:
    
                - ``None`` as default to apply ``fn`` to the data directly.
                - Integer(s) is used for list/tuple.
                - Key(s) is used for dict.
    
            output_col: Index of data where result of ``fn`` is placed. ``output_col`` can be specified
                only when ``input_col`` is not ``None``
    
                - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
                  multiple indices, the left-most one is used, and other indices will be removed.
                - Integer is used for list/tuple. ``-1`` represents to append result at the end.
                - Key is used for dict. New key is acceptable.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
            >>> def add_one(x):
            ...     return x + 1
            >>> dp = IterableWrapper(range(10))
            >>> map_dp_1 = dp.map(add_one)  # Invocation via functional form is preferred
            >>> list(map_dp_1)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> # We discourage the usage of `lambda` functions as they are not serializable with `pickle`
            >>> # Use `functools.partial` or explicitly define the function instead
            >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
            >>> list(map_dp_2)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        """
    
    # Functional form of 'MultiplexerIterDataPipe'
    def mux(self, *datapipes) -> IterDataPipe:
        r"""
        Yields one element at a time from each of the input Iterable DataPipes (functional name: ``mux``).
    
        As in, one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
        and so on. It ends when the shortest input DataPipe is exhausted.
    
        Args:
            datapipes: Iterable DataPipes that will take turn to yield their elements, until the shortest DataPipe is exhausted
    
        Example:
            >>> # xdoctest: +REQUIRES(module:torchdata)
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp1, dp2, dp3 = IterableWrapper(range(3)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
            >>> list(dp1.mux(dp2, dp3))
            [0, 10, 20, 1, 11, 21, 2, 12, 22]
        """
    
    # Functional form of 'FileOpenerIterDataPipe'
    def open_files(self, mode: str = "r", encoding: Optional[str] = None, length: int = -1) -> IterDataPipe:
        r"""
        Given pathnames, opens files and yield pathname and file stream in a tuple (functional name: ``open_files``).
    
        Args:
            datapipe: Iterable datapipe that provides pathnames
            mode: An optional string that specifies the mode in which
                the file is opened by ``open()``. It defaults to ``r``, other options are
                ``b`` for reading in binary mode and ``t`` for text mode.
            encoding: An optional string that specifies the encoding of the
                underlying file. It defaults to ``None`` to match the default encoding of ``open``.
            length: Nominal length of the datapipe
    
        Note:
            The opened file handles will be closed by Python's GC periodically. Users can choose
            to close them explicitly.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader
            >>> dp = FileLister(root=".").filter(lambda fname: fname.endswith('.txt'))
            >>> dp = FileOpener(dp)
            >>> dp = StreamReader(dp)
            >>> list(dp)
            [('./abc.txt', 'abc')]
        """
    
    # Functional form of 'StreamReaderIterDataPipe'
    def read_from_stream(self, chunk=None) -> IterDataPipe:
        r"""
        Given IO streams and their label names, yield bytes with label name as tuple.
    
        (functional name: ``read_from_stream``).
    
        Args:
            datapipe: Iterable DataPipe provides label/URL and byte stream
            chunk: Number of bytes to be read from stream per iteration.
                If ``None``, all bytes will be read until the EOF.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper, StreamReader
            >>> from io import StringIO
            >>> dp = IterableWrapper([("alphabet", StringIO("abcde"))])
            >>> list(StreamReader(dp, chunk=1))
            [('alphabet', 'a'), ('alphabet', 'b'), ('alphabet', 'c'), ('alphabet', 'd'), ('alphabet', 'e')]
        """
    
    # Functional form of 'RoutedDecoderIterDataPipe'
    def routed_decode(self, *handlers: Callable, key_fn: Callable= ...) -> IterDataPipe:
        r"""
        Decodes binary streams from input DataPipe, yields pathname and decoded data in a tuple.
    
        (functional name: ``routed_decode``)
    
        Args:
            datapipe: Iterable datapipe that provides pathname and binary stream in tuples
            handlers: Optional user defined decoder handlers. If ``None``, basic and image decoder
                handlers will be set as default. If multiple handles are provided, the priority
                order follows the order of handlers (the first handler has the top priority)
            key_fn: Function for decoder to extract key from pathname to dispatch handlers.
                Default is set to extract file extension from pathname
    
        Note:
            When ``key_fn`` is specified returning anything other than extension, the default
            handler will not work and users need to specify custom handler. Custom handler
            could use regex to determine the eligibility to handle data.
        """
    
    # Functional form of 'ShardingFilterIterDataPipe'
    def sharding_filter(self, sharding_group_filter=None) -> IterDataPipe:
        r"""
        Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``).
    
        After ``apply_sharding`` is called, each instance of the DataPipe (on different workers) will have every `n`-th element of the
        original DataPipe, where `n` equals to the number of instances.
    
        Args:
            source_datapipe: Iterable DataPipe that will be sharded
        """
    
    # Functional form of 'ShufflerIterDataPipe'
    def shuffle(self, *, buffer_size: int = 10000, unbatch_level: int = 0) -> IterDataPipe:
        r"""
        Shuffle the input DataPipe with a buffer (functional name: ``shuffle``).
    
        The buffer with ``buffer_size`` is filled with elements from the datapipe first. Then,
        each item will be yielded from the buffer by reservoir sampling via iterator.
    
        ``buffer_size`` is required to be larger than ``0``. For ``buffer_size == 1``, the
        datapipe is not shuffled. In order to fully shuffle all elements from datapipe,
        ``buffer_size`` is required to be greater than or equal to the size of datapipe.
    
        When it is used with :class:`torch.utils.data.DataLoader`, the methods to
        set up random seed are different based on :attr:`num_workers`.
    
        For single-process mode (:attr:`num_workers == 0`), the random seed is set before
        the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
        mode (:attr:`num_worker > 0`), `worker_init_fn` is used to set up a random seed
        for each worker process.
    
        Args:
            datapipe: The IterDataPipe being shuffled
            buffer_size: The buffer size for shuffling (default to ``10000``)
            unbatch_level: Specifies if it is necessary to unbatch source data before
                applying the shuffle
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(range(10))
            >>> shuffle_dp = dp.shuffle()
            >>> list(shuffle_dp)
            [0, 4, 1, 6, 3, 2, 9, 5, 7, 8]
        """
    
    # Functional form of 'UnBatcherIterDataPipe'
    def unbatch(self, unbatch_level: int = 1) -> IterDataPipe:
        r"""
        Undos batching of data (functional name: ``unbatch``).
    
        In other words, it flattens the data up to the specified level within a batched DataPipe.
    
        Args:
            datapipe: Iterable DataPipe being un-batched
            unbatch_level: Defaults to ``1`` (only flattening the top level). If set to ``2``,
                it will flatten the top two levels, and ``-1`` will flatten the entire DataPipe.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper([[[0, 1], [2]], [[3, 4], [5]], [[6]]])
            >>> dp1 = source_dp.unbatch()
            >>> list(dp1)
            [[0, 1], [2], [3, 4], [5], [6]]
            >>> dp2 = source_dp.unbatch(unbatch_level=2)
            >>> list(dp2)
            [0, 1, 2, 3, 4, 5, 6]
        """
    
    # Functional form of 'ZipperIterDataPipe'
    def zip(self, *datapipes: IterDataPipe) -> IterDataPipe:
        r"""
        Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).
    
        The output is stopped as soon as the shortest input DataPipe is exhausted.
    
        Args:
            *datapipes: Iterable DataPipes being aggregated
    
        Example:
            >>> # xdoctest: +REQUIRES(module:torchdata)
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
            >>> list(dp1.zip(dp2, dp3))
            [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
        """
    

class DFIterDataPipe(IterDataPipe):
    def _is_dfpipe(self): ...
    def __iter__(self): ...

class _DataPipeSerializationWrapper:
    def __init__(self, datapipe): ...
    def __getstate__(self): ...
    def __setstate__(self, state): ...
    def __len__(self): ...

class _IterDataPipeSerializationWrapper(_DataPipeSerializationWrapper, IterDataPipe):
    def __iter__(self): ...

class _MapDataPipeSerializationWrapper(_DataPipeSerializationWrapper, MapDataPipe):
    def __getitem__(self, idx): ...
