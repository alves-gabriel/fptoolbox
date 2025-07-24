import numpy as np
from typing import Any
from functools import reduce

from .noncrossing import generate_non_crossing

def centralize_operator(operator : Any, moment_type="infinite_temperature") -> Any:
    """Centralizes an operator for a given moment type."""
    
    phi = moment([operator], moment_type=moment_type) 
    
    return operator - phi

def infinite_temperature_trace(operator_list : list[Any]) -> float:
    """Returns Tr(ABC ... Z) given the operator list [A, B, C, ..., Z]."""
    dim = len(operator_list[0])
    
    return 1./dim * np.trace(reduce(np.matmul, operator_list))

def moment(operator_list : list[Any], moment_type = "infinite_temperature") -> float:
    """Chooses how the statistical momenta Φ are computed. The default choice corresponds to:

    Φ(•) = Tr(•)/D,

    where D is the operator dimensionality.
    """
    if moment_type == "infinite_temperature":
        return infinite_temperature_trace(operator_list)

    raise ValueError("Invalid moment choice.")

def _kappa_pi(operator_lst : list[Any], partition : list[list[int]]) -> float:
    """"Helper function for block-decomposition of cumulants in term of lower order ones."""
    
    k_pi = 1
    for block in partition:
        # Use block to implement index set
        operators = np.asarray(operator_lst)[np.array(block)]
        
        # Computes the cumulant
        k_pi *= _kappa_n(operators)   
    
    return k_pi
    
def _kappa_n(operator_lst : list[Any], moment_type="infinite_temperature") -> float:
    """
    Recursive computation of free cumulants using the formula:

    
    (*) κ_n(A_1, A_2, ..., A_n) = Φ(A_1, A_2, ..., A_n) - Σ κ_π(A_1, A_2, ..., A_n),


    where the summation is performed over non-crossing partitions and the quantity k_pi
    corresponds to the block-decomposition of cumulants in smaller orders.
    """

    # Statistical moment Φ(a_1 ... a_n)
    dim = len(operator_lst[0])
    momenent_n = moment(operator_lst, moment_type=moment_type)
    
    # Partition π = 1
    n = len(operator_lst)
    maximum_partition = [list(range(n))]

    # Recursively compute the cumulants
    sum_kappa = 0
    for partition in generate_non_crossing(n):

        # Skips the partition [[1, 2, ..., n]]
        if partition != maximum_partition:
            sum_kappa += _kappa_pi(operator_lst, partition)

    # Applies the formula (*)
    k_n = momenent_n - sum_kappa
    
    return k_n

"""Alias for computing the free cumulants. See the function _kappa_n."""
free_cumulant = _kappa_n