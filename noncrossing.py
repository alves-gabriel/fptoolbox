import more_itertools
from copy import copy

def _is_block_noncrossing(block : list[int]) -> bool:
    """
    >>> _is_block_noncrossing([3, 4, 5])
    True

    >>> _is_block_noncrossing([3, 4, 6])
    False
    """
    k, p = block[0], block[-1]

    return block == list(range(k, p + 1))

def _relabel(element : list[int], subtract : int, smallest : int) -> list[int]:
    """Relabels elements (lager than smallest) in a block after removing a partition."""
    return element - subtract if element > smallest else element

def remove_block(partition : list[list[int]], block : list[int]) -> list[list[int]]:
    """
    Only works if the block being remove is non-crossing.
    
    >>> remove_block([[1, 10], [2, 5, 9], [3, 4], [6], [7, 8]], [6])
    [[1, 9], [2, 5, 8], [3, 4], [6, 7]]

    >>> remove_block([[1, 9], [2, 5, 8], [3, 4], [6, 7]], [3, 4])
    [[1, 7], [2, 3, 6], [4, 5]]

    Notes
    ------

    For instance, trying to remove [1, 6] from [[1, 6], [2, 3], [4, 5]] raises an error.
    """

    # Removes the block from the partition
    if block not in partition:
        raise ValueError("Invalid block")
        
    if not _is_block_noncrossing(block):
        raise ValueError("Block must be non-crossing!")
        
    partition.remove(block)

    # Relabel elements according to the length of the removed block
    smallest_element = block[0]
    subtract = len(block)
    
    return [[_relabel(element, subtract, smallest_element) for element in block] for block in partition]

def is_partition_noncrossing(partition : list[list[int]]) -> bool:
    """
    >>> is_partition_noncrossing([[1, 10], [2, 5, 9], [3, 4], [6], [7, 8]])
    True

    >>> is_partition_noncrossing([[1, 10], [2, 5, 9], [3, 8], [6], [4, 7]])
    False
    """

    # Checks dimensionality
    if not all(isinstance(block, list) for block in partition):
        raise TypeError("The partition should be a list of lists. (At least) one of the blocks is not a list.")

    # Performs a shallow copy on the partition
    partition = copy(partition)
    
    # Loops over blocks
    i = 0
    while i < len(partition):
        # Moves to the next block if not crossing, otherwise remove crossing contribution
        if not _is_block_noncrossing(partition[i]):
            i+=1
        else:
            partition = remove_block(partition, partition[i])
            i = 0

    return True if partition == [] else False

def all_partitions(N : int) -> list[list[int]]:
    """
    Generates all the crossing partitions for the set {1, ..., N}. 
    
    The total number of elemens should follow the Bell numbers: https://en.wikipedia.org/wiki/Bell_number.

    >>> len(all_partitions(8))
    4140
    """
    
    natural_numbers = list(range(0, N))
    all_partitions = [block for num_blocks in range(1, N + 1) for block in more_itertools.set_partitions(natural_numbers, num_blocks)]
    
    return all_partitions

def generate_non_crossing(N : int) -> list[list[int]]:
    """
    Generates all the non-crossing partitions for the set {1, ..., N}
    
    >>> len(generate_non_crossing(8))
    1430
    """
    
    non_crossing_partitions = []

    for partition in all_partitions(N):
        if is_partition_noncrossing(partition):
            non_crossing_partitions.append(partition)
        
    return non_crossing_partitions