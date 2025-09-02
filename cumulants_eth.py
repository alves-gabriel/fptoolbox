import numpy as np 
import scipy as scp

####################
# CUMULANT ALGEBRA #
####################
    
def compute_eth_cumulant(order, A, B):
    """
    Compute the ETH cumulant of given order for operators A and B.

    Parameters
    ----------
    order : int
        The order of the cumulant. Currently only supports 2 and 4.
    A, B : np.ndarray
        Operators (square matrices).

    Returns
    -------
    float
        The computed ETH cumulant of the specified order.

    Raises
    ------
    ValueError
        If an invalid order is provided.
    """

    # Hilbert space dimension
    D = A.shape[0]
    
    if order == 2:
        return 1/D * np.einsum('ij, ji', A, B, optimize=True) 
    elif order == 4:
        return 1/D * (np.einsum('ij, jk, kl, li->', A, B, A, B, optimize=True)  # Full contraction
                    - np.einsum('ij, ji, il, li->', A, B, A, B, optimize=True)  # i = k, cactus 
                    - np.einsum('ij, jk, kj, ji->', A, B, A, B, optimize=True)  # j = l, cactus
                    + np.einsum('ij, ji, ij, ji->', A, B, A, B, optimize=True)) # i = k and j = l, crossing 
    else:
        raise ValueError("Invalid cumulant order.")

def compute_eth_diagram(diagram_type, A, B):
    """
    Compute individual ETH diagram contributions (e.g., crossing or cactus diagrams).

    Parameters
    ----------
    diagram_type : str
        The diagram type to compute. Valid options are "crossing" and "cactus".
    A, B : np.ndarray
        Operators (square matrices).

    Returns
    -------
    float
        The computed ETH diagram contribution of the specified type.

    Raises
    ------
    ValueError
        If an invalid diagram type is provided.
    """
    
    # Hilbert space dimension
    D = A.shape[0]

    if diagram_type == "crossing":
        return 1/D * np.einsum('ij, ji, ij, ji', A, B, A, B, optimize=True) # i = k and j = l
    elif diagram_type == "cactus":
        return 1/D * np.einsum('ij, ji, il, li', A, B, A, B, optimize=True) # i = k cactus
    else:
        raise ValueError("Invalid diagram type.")