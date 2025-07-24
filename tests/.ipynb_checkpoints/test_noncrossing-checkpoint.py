# pytest tests/test_noncrossing.py
import pytest
import numpy as np
from ..noncrossing import *

def test_remove_block():
    partition = [[1, 10], [2, 5, 9], [3, 4], [6], [7, 8]]
    block = [6]
    new_partition = remove_block(partition, block)
    assert new_partition == [[1, 9], [2, 5, 8], [3, 4], [6, 7]]

    partition = [[1, 9], [2, 5, 8], [3, 4], [6, 7]]
    block = [3, 4]
    new_partition = remove_block(partition, block)
    assert new_partition == [[1, 7], [2, 3, 6], [4, 5]]

    with pytest.raises(ValueError):
        remove_block([[1, 6], [2, 3], [4, 5]], [1, 6])  # not non-crossing

    with pytest.raises(ValueError):
        remove_block([[1, 5], [1, 2, 4]], [9])  # not in partition

def test_is_partition_noncrossing():
    partition_nc = [[1, 10], [2, 5, 9], [3, 4], [6], [7, 8]]
    partition_crossing = [[1, 10], [2, 5, 9], [3, 8], [6], [4, 7]]
    assert is_partition_noncrossing(partition_nc) is True
    assert is_partition_noncrossing(partition_crossing) is False

def test_all_partitions():
    partitions = all_partitions(8)
    assert isinstance(partitions, list)
    assert all(isinstance(p, list) for p in partitions)
    assert len(partitions) == 4140  # Bell(8) = 4140

def test_generate_non_crossing():
    non_crossing = generate_non_crossing(8)
    assert isinstance(non_crossing, list)
    assert all(is_partition_noncrossing(p) for p in non_crossing)
    assert len(non_crossing) == 1430  # Known Catalan(8) = 1430