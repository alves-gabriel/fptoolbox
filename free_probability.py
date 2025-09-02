import math
import numpy as np 
import scipy as scp
import matplotlib.pyplot as plt
from functools import reduce
from .cumulants import free_cumulant
from .cumulants_eth import *

##############################
# ETH/FREE PROBABILITY CLASS #
##############################

class FreeModel:
    """Compute meaningful quantities for ETH (Eigenstate Thermalization Hypothesis), Free Probability (FP) and Quantum chaos."""

    def __init__(self, model, unitary=True):

        # Parameters of the model
        self.matrix = model
        self.unitary = unitary
        self.D = model.shape[0]
        self.diagonalization = None
        self.omega = None

        # Level spacing statistics
        self.spacing_distribution = None
        self.spacing_ratio = None
        self.mean_spacing_ratio = None

        # Regularization used in the FT
        # self.eth_regularization = lambda omega : _delta_tau(omega, unitary=unitary)

    ############################
    # OPERATOR DIAGONALIZATION #
    ############################
    
    def diagonalize_operator(self, op, delete_diagonal=True, sorted_eigs=False, truncate=None):
        """
        Evaluates A_{mn} and  ω_{ij}, which are used in the calculation 
        of the cumulants. I.e., expresses the operator and the frequencies 
        in the eigenbasis of the model.
        
        Parameters
        ----------
        
        op : array 
            D x D operator of interest in the computational basis.

        sorted: bool, optional
            Sorts the eigenspectrum and, correspondingly, the eigenstates.
                              
        Returns
        -------
        
        (A, omega) : (np.array, np.array)
            Returns the operator in the eigenbasis, denoted by A_{mn}
            and the eigenspectra (differences) ω_{ij}.        
        """ 

        if truncate is not None:
            sorted_eigs = True
        
        self.operator = op
        
        # Performs the diagonalization (if unsolved)
        self.diagonalize()
        eig_vals, eig_states = self.diagonalization
        
        # Complex angles should be used in the case of unitary evolution
        if self.unitary:
            eig_vals = np.angle(eig_vals)
            
        # Order eigenstates
        if sorted_eigs:
            ordr = np.argsort(eig_vals)
            eig_vals = eig_vals[ordr]
            eig_states = eig_states[:, ordr]

            # Put eigenvalues in angle form and orders the eig vales and eig states
            self.diagonalization = eig_vals, eig_states

        # Phase differences
        omega = np.array([[Em - En for En in eig_vals] for Em in eig_vals])
    
        # Operator of interest in the eigenbasis
        self.A = eig_states.conj().T @ op @ eig_states
        self.A_diag = np.array(np.diag(self.A))
        
        # Manually sets the diagonal to zero in the eigenbasis (if desired)
        if delete_diagonal:
            
            for i in range(self.D):
                self.A[i][i] = 0

        # Restrict the phases to the interval [-pi, pi]
        self.omega = np.mod(omega + np.pi, 2*np.pi) - np.pi if self.unitary else omega

        # Truncate the spectrum and the dimensionality of the model accordingly
        if truncate:
            i_max = int(self.D*truncate)
            self.A = self.A[i_max:-i_max, i_max:-i_max] 
            self.omega = self.omega[i_max:-i_max, i_max:-i_max] 
            self.D = len(self.A)
            
        return self
        
    def diagonalize(self):
        """
        Diagonalizes the unitary/Hamiltonian. Uses Schur decomposition in the unitary case
        and eigh for the Hamiltonian case for efficiency.
        """

        # Skips the method if already diagonalized
        if self.diagonalization is not None:
            return self

        if self.unitary:
            schur_decomposition = scp.linalg.schur(self.matrix)

            # Extracts the eigenvalues in the diagonal
            self.diagonalization = np.diag(schur_decomposition[0]), schur_decomposition[1]
        else:
            self.diagonalization = np.linalg.eigh(self.matrix) 
            
        return self
        
    ############################
    # LEVEL SPACING STATISTICS #
    ############################

    def get_level_statistics(self):
        """
        Evaluates the level spacing (+ ratio) distribution
        
        Returns
        -------
        
        ([s_i], [r_i]) : (array, array)
            A tuple with two arrays. The first one contains the level spagin distribution s_i, while the second one
            contains the corresponding distribution for the ratio r_i.
        """ 
        
        # Performs the diagonalization (if unsolved)
        self.diagonalize()
    
        # Extracts eigenvalues and eigenstates
        eig_vals, _ = self.diagonalization
        
        # Spacing between the (ordered and consecutive) angles associated with the (complex) eigenvalues
        sorted_eigs = np.sort(eig_vals)
        spacing_distribution = np.array([theta1 - theta0 for theta0, theta1 in zip(sorted_eigs, sorted_eigs[1:])])
        
        # Normalization for the unitary case. No need to unfold. See Phys. Rev. Research 6, 023068 (2024)
        if self.unitary:
            self.spacing_distribution = spacing_distribution/(2*np.pi)

        # Unfolding for the non-unitary case. TODO
        else:
            # TODO UNFOLDING
            self.spacing_distribution = spacing_distribution/np.mean(spacing_distribution) 
        
        # Level spacing ratio distribution
        self.spacing_ratio = [min(s1, s0)/max(s1, s0) for s0, s1 in zip(spacing_distribution, spacing_distribution[1:])]

        # Mean spacing ratio
        self.mean_spacing_ratio = np.mean(self.spacing_ratio)

        return self
  
    ########################
    # DYNAMICAL QUANTITIES #
    ########################

    def heisenberg_evolution(self, A, t):
        """Returns the time-evolved operator A(t) in the Heisenberg picture at time t."""
        
        # Performs the diagonalization (if unsolved)
        self.diagonalize()
        eig_vals, eig_vecs = self.diagonalization  

        # Unitary at time t
        U_t = eig_vecs @ np.diag(eig_vals**t) @ eig_vecs.conj().T
        
        return U_t.conj().T @ A @ U_t
    
    def OTOC(self, A, B, t_min, t_max, k_max=1):
        """
        Computes mixed k-th OTOC between A and B.

        Parameters
        ----------
        
        A, B : np.array 
            Operators.
        
        t_min, t_max : float 
            Minimum and maximum time for the dynamics.
            
        k_max : int
            Maximum OTOC order k.
            
        Returns
        -------
        
        np.array:
        - Axis 0 (rows): time axis.
        - Axis 1 (columns): k-th OTOC at time t.
        """
        
        # Loops over time
        otoc = []
        for t in range(t_min, t_max):

            A_t = self.heisenberg_evolution(A, t)
    
            # Computes the k-th OTOC at time t, constructing the product <A(t)A ... A(t)A>
            prod, otoc_k = A_t @ B, []
            for k in range(k_max):
                otoc_k.append(np.trace(prod))
                prod = (A_t @ B) @ prod
                
            otoc.append(otoc_k)
            
        return np.array(otoc)/self.D

    def dynamical_cumulant(self, A, B, t_min, t_max, k_max=1, moment_type="infinite_temperature"):
        """
        Computes the 2k-th dynamical cumulants k(A(t), B, A(t), B,...) from the free probability definition.

        Parameters
        ----------
        
        A, B : np.array 
            Operators.
        
        t_min, t_max : float 
            Minimum and maximum time for the dynamics.
            
        k_max : int
            Maximum cumulant order k.
    
        moment_type: int
            Functional used to compute the moments. The default choices corresponds to Tr(A(t) B A(t) B....)/D,
            agreein with the (normalized) OTOC.
            
        Returns
        -------
        
        np.array:
        - Axis 0 (rows): time axis.
        - Axis 1 (columns): 2k-th cumulant at time t.
        """
        
        # Loops over time
        dynamical_cumulant = []
        for t in range(t_min, t_max):

            # Dynamics
            A_t = self.heisenberg_evolution(A, t)

            # Computes mixed cumulant up to order k_max
            cumulant_k = []
            for k in range(1, k_max + 1):
                cumulant_k.append(free_cumulant([A_t, B] * k, moment_type=moment_type))

            dynamical_cumulant.append(cumulant_k)
            
        return np.array(dynamical_cumulant)

#################
# ETH CUMULANTS #
#################

    # ----------------------------
    # ETH CUMULANTS IN TIME DOMAIN 
    # ----------------------------

    def _initialize_operator_eigenbasis(self, A, B, delete_diagonal):
        """Write down B in eigenbasis (or set it equal to A if undefined)."""
        
        _, eig_states = self.diagonalization

        if B is None:
            B = A

        # Basis transformation and deletes diagonal
        A_no_diag = eig_states.conj().T @ A @ eig_states
        B_no_diag = eig_states.conj().T @ B @ eig_states

        if delete_diagonal:
            for i in range(self.D):
                A_no_diag[i][i] = 0
                B_no_diag[i][i] = 0
            
        return A_no_diag, B_no_diag
                
    def eth_cumulant(self, order, t_lst, A, B=None, delete_diagonal=True):
        """
        Computes the mixed ETH cumulant of given order.

        Parameters
        ----------
        
        A, B : np.array 
            Operators. If B is none, takes B = A.
        
        t_lst : np.array
            List of time steps over which the computation is performed.
            
        Returns
        -------
        
        np.array:
            The cumulant from time t_min to t_max.
        """
        
        # Initializes 
        A, B = self._initialize_operator_eigenbasis(A, B, delete_diagonal)

        return np.array([compute_eth_cumulant(order, A * np.exp(self.omega*1j*t), B) for t in t_lst])

    def eth_diagram(self, diagram_type, t_lst, A, B=None, delete_diagonal=True):
        """See eth_cumulant. Computes eth diagrams, such as crossing and cactus diagrams."""

        # Initializes
        A, B = self._initialize_operator_eigenbasis(A, B, delete_diagonal)

        return np.array([compute_eth_diagram(diagram_type, A * np.exp(self.omega*1j*t), B) for t in t_lst])

    # --------------------------------- 
    # ETH CUMULANTS IN FREQUENCY DOMAIN 
    # ---------------------------------
    
    def eth_cumulant_freq(self, order, w_lst, A, B=None, delete_diagonal=True):
        """See eth_cumulant. Computes the mixed ETH cumulant of given order in the frequency domain, given the frequencies w_lst."""
        
        # Initializes
        A, B = self._initialize_operator_eigenbasis(A, B, delete_diagonal)
                
        return np.array([compute_eth_cumulant(order, A * _delta_tau(w - self.omega), B) for w in w_lst])

    def eth_diagram_freq(self, diagram_type, w_lst, A, B=None, delete_diagonal=True):
        """See eth_cumulant_freq. Computes eth diagrams in frequency domain."""
        
        # Initializes
        A, B = self._initialize_operator_eigenbasis(A, B, delete_diagonal)
                
        return np.array([compute_eth_diagram(diagram_type, A * _delta_tau(w - self.omega), B) for w in w_lst])

#######################
# AUXILIARY FUNCTIONS #
#######################

def _gaussian(x, mu, sigma):
    """Gaussian function with mean mu and variance sigma."""
    # return sigma/(sigma**2.5 + 2*sigma*(x - mu)**2)
    
    return 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-((x - mu)/sigma)**2/2)
    
def _delta_tau(w, w0=0, tau=25, unitary=True):
    """Gaussian approximating a Delta function. Note that this needs to have a period of 2pi for the unitary case, corresponding to a Dirac comb."""
    
    if unitary:        
        return 2 * np.pi * (_gaussian(w, w0, 1/tau) + _gaussian(w, w0 - 2*np.pi, 1/tau) + _gaussian(w, w0 + 2*np.pi, 1/tau))
    else:
        return 2 * np.pi * _gaussian(w, w0, 1/tau)