import math
import numpy as np 
import scipy as scp
import matplotlib.pyplot as plt
from functools import reduce

#######################
# AUXILIARY FUNCTIONS #
#######################

def _gaussian(x, mu, sig):
    """Gaussian function with mean mu and variance sig"""
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-((x - mu)/sig)**2/2)
    
    
def _delta_tau(w, w0=0, tau=10, unitary=True):
    """Gaussian approximating a Delta function. Note that this needs to have a period of 2 pi for the unitary case."""
    
    if unitary:
        return _gaussian(w, w0, 1/tau) + _gaussian(w, w0 - 2*np.pi, 1/tau) + _gaussian(w, w0 + 2*np.pi, 1/tau)
    else:
        return _gaussian(w, w0, 1/tau)
    
##################
# CUMULANT CLASS #
##################

class FreeModel:
    """Calculates the cumulants from FP theory and relevant quantities for Quantum Chaos"""

    def __init__(self, model, unitary=True):

        # Parameters of the model
        self.matrix = model
        self.unitary = unitary
        self.diagonalization = None
        self.A = None
        self.omega = None

        # Level spacing statistics
        self.spacing_distribution = None
        self.spacing_ratio = None
        self.mean_spacing_ratio = None

        # Correlations and cumulants
        self.otoc = {}
        self.k2 = {}
        self.k4 = {}
        self.cactus = {}
        self.crossing = {}

        # Correlations and cumulants in frequency domain
        self.otoc_freq = {}
        self.k2_freq = {}
        self.k4_freq = {}
        self.cactus_freq = {}
        self.crossing_freq = {}

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
        
        model : array 
            Floquet operator/unitary evolution/Hamiltonian.
            
        unitaryEvol : bool, optional
            ω_{ij} is defined as ω_{ij} := φ_i - φ_j when true and 
                              as ω_{ij} := E_i - E_j when false.

        sorted: bool, optional
            Sorts the eigenspectrum and, correspondingly, the eigenstates
                              
        Returns
        -------
        
        (A, omega) : (array, array)
            Returns the operator in the eigenbasis, denoted by A_{mn}
            and the eigenspectra (differences) ω_{ij}.        
        """ 

        if truncate is not None:
            sorted = True
        
        # Hilbert space dimension
        self.operator = op
        self.D = len(op)
        
        # Performs the diagonalization (if unsolved)
        if self.diagonalization is None:
            self.diagonalize()
    
        # Extracts eigenvalues and eigenstates
        eig_vals, eig_states = self.diagonalization
        
        # Complex angles should be used in the case of unitary evolution
        if self.unitary:
            eig_vals=np.angle(eig_vals)
            
        # Order eigenstates
        if sorted_eigs:
            ordr = np.argsort(eig_vals)
            eig_vals = eig_vals[ordr]
            eig_states = eig_states[:,ordr]

            # Put eigenvalues in angle form and orders the eig vales and eig states
            self.diagonalization = eig_vals, eig_states
            
        omega = np.array([[Em - En for En in eig_vals] for Em in eig_vals])
    
        # Operator of interest in the eigenbasis. TODO: transpose of this?
        A = eig_states.conj().T@op@eig_states
        
        # Manually sets the diagonal to zero in the eigenbasis
        if delete_diagonal:
            for i in range(self.D):
                A[i][i] = 0

        # Sets the frequencies and operator in the model eigenbasis
        self.A = A

        # IMPORTANT: remove unphysical phase differences. 
        # We should restrict the phases to the intervals [0, 2pi]
        self.omega = np.mod(omega, 2*np.pi) if self.unitary else omega

        if truncate:
            i_max = int(self.D*truncate)
            self.A  = self.A[i_max:-i_max, i_max:-i_max] 
            self.omega = self.omega[i_max:-i_max, i_max:-i_max] 
            self.D = len(self.A)
            
        return self
        
    def diagonalize(self):
        """
        Diagonalizes the unitary/Hamiltonian
        """
        if self.unitary:
            # Schur decomposition
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
        if self.diagonalization is None:
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
    
    ###############
    # TIME DOMAIN #
    ###############

    #------#
    # OTOC #
    #------#
    
    def _compute_OTOC(self, t):
        """Evaluates the OTOC <A(t) A A(t) A>."""
        if t in self.otoc.keys():
            return self.otoc[t]
            
        At = self.A * np.exp(self.omega*1j*t)
        self.otoc[t] = np.trace(At@self.A@At@self.A)/self.D
        
        return self.otoc[t]

    def compute_OTOC(self, t):
        """Vectorizes the method."""
        return np.vectorize(self._compute_OTOC, excluded='self')(t)

    #----------------#
    # Cumulant K2(t) #
    #----------------#
    
    def _compute_K2(self, t):
        """
        Evaluates the second order cumulant k2(t):=Σ e^(i ω_{ij} t) |A_{ij}|^2
        at time t. Summed over i != j.
        """
        
        if t in self.k2.keys():
            return self.k2[t]
            
        # Computes the cumulant
        omega_t = np.exp(self.omega*1j*t)
        At = self.A * omega_t
        k2 = np.einsum('ij,ji', At, self.A, optimize=True)                           
        self.k2[t] = k2/self.D 
        
        return self.k2[t]
        
    def compute_K2(self, t):
        """Vectorizes the method."""
        return np.vectorize(self._compute_K2, excluded='self')(t)

    #----------------#
    # Cactus diagram #
    #----------------#
    
    def _compute_cactus(self, t1, t2=None):
        """Evaluates the cactus diagram at times t1 and t2."""
               
        if t2 is None:
            t2 = t1
            
        if (t1, t2) in self.cactus.keys():
            return self.cactus[(t1, t2)]

        # Computes the cactus diagram
        At1 = self.A * np.exp(self.omega*1j*t1)   
        At2 = self.A * np.exp(self.omega*1j*t2)   
        cactus = np.einsum('ij,ji,ik,ki', At2, self.A, At1, self.A, optimize=True)
        self.cactus[(t1, t2)] = cactus/self.D - self.compute_crossing(t1 + t2)

        return self.cactus[(t1, t2)]
        
    def compute_cactus(self, t1, t2=None):
        """Vectorizes the method."""
        return np.vectorize(self._compute_cactus, excluded='self')(t1, t2)

    # ------------------#
    # Crossing diagrams #
    # ----------------- #
    
    def _compute_crossing(self, t):
        """Evaluates the crossing diagrams k2(t):=Σ |A_{ij}|^4."""
        
        if t in self.crossing.keys():
            return self.crossing[t]
            
        # Computes the diagram
        omega_t = np.exp(self.omega*1j*t)
        At = self.A * omega_t  
        crossing = np.einsum('ij,ji,ij,ji', At, self.A, At, self.A, optimize=True)
        self.crossing[t] = crossing/self.D 
                                     
        return self.crossing[t]
        
    def compute_crossing(self, t):
        """Vectorizes the method"""
        return np.vectorize(self._compute_crossing, excluded='self')(t)

    # -------------- #
    # Cumulant K4(t) #
    # -------------- #
    
    def _compute_K4(self, t):
        """Evaluates the fourth order cumulant."""

        # Be sure that everything here is being computed for the same t
        # CAUTION: check this
        self.k4[t] = self.compute_OTOC(t) + self.compute_crossing(t) - 2*self.compute_cactus(t)
            
        return self.k4[t]
        
    def compute_K4(self, t):
        """Vectorizes the method."""
        return np.vectorize(self._compute_K4, excluded='self')(t)

    #-----------------#
    # Other functions #
    #-----------------#
    
    def factorization_ratio(self, t=0):
        """
        Evaluates factorization parameter r(t) := cactus(t)/|k_2(t)|². 
        If t=None computes r(t) for the current time
        """
        
        return self.compute_cactus(t)/self.compute_K2(t)**2

    def plotETH(self, t_lst, show_cactus=False, show_crossing=False):
        """
        Plots the OTOCs and the diagrams for the model as a function of time.

        Parameters
        ----------
        
        t_lst : array 
            List of points in time where the evaluations should be performed.
        
        show_cactus : bool , optional.
            Computes the cacus diagram cac(t, t)
            
        show_crossing : bool, optional
            Computes the crossing partition.
        """
        
        # Plots
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot()

        # Optional cactus and crossing partitions
        if show_cactus:
            ax.plot(t_lst, np.abs(self.compute_cactus(t_lst))**2, color="magenta", label=r'$\mathrm{Cactus}$', marker='s', markersize=6)

        if show_crossing:
            ax.plot(t_lst, np.abs(self.compute_crossing(t_lst))**2, color="orange", label=r'$\mathrm{Crossing}$', marker='x', markersize=10)
                    
        # K2
        ax.plot(t_lst, 2*np.abs(self.compute_K2(t_lst))**2
                , marker='*', markersize=10, markeredgewidth=1, markeredgecolor="black", label=r'$2|k_2(t)|^2$')

        # OTOC
        ax.plot(t_lst, np.real(self.compute_OTOC(t_lst))
                , color="red", marker='d', markersize=5, markeredgewidth=1, markeredgecolor="black", label=r'$\langle A(t)AA(t)A\rangle$')

        # K4
        ax.plot(t_lst, np.real(self.compute_K4(t_lst))       
                , color="green", marker='o', markersize=5, markeredgewidth=1, markeredgecolor="black", label='$k_4(t)$')

        # K2 + K4
        ax.plot(t_lst, 2*np.abs(self.compute_K2(t_lst))**2 + np.real(self.compute_K4(t_lst))
                , color="black", linestyle='dashed', label=r'$2|k_2(t)|^2+k_4(t)$')

        # Labels
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\mathrm{Contribution}$')
        ax.legend(loc="upper right")    
        
        plt.show()

    ####################
    # FREQUENCY DOMAIN #
    ####################
    
    #------#
    # OTOC #
    #------#
    
    def _compute_OTOC_freq(self, w1, w2=None, w3=None):
        """
        Evaluates the OTOC in the frequency domain.
        """

        # Avoids double computation
        if (w1, w2, w3) in self.otoc_freq.keys():
            return self.otoc_freq[(w1, w2, w3)]

        if w2 is None:
            w2 = w1
            
        if w3 is None:
            w3 = -w1   

        # Computes the OTOC
        Aw3 = self.A * _delta_tau(w3 - self.omega)
        Aw2 = self.A * _delta_tau(w2 - self.omega)
        Aw1 = self.A * _delta_tau(w1 - self.omega)
        self.otoc_freq[(w1, w2, w3)] = np.trace(Aw1@Aw2@Aw3@self.A)/self.D
        
        return self.otoc_freq[(w1, w2, w3)]
        
    def compute_OTOC_freq(self, w1, w2=None, w3=None):
        """Vectorizes the method."""
        return np.vectorize(self._compute_OTOC_freq, excluded='self')(w1, w2, w3)
    
    #----------------#
    # Cumulant K2(t) #
    #----------------#
    
    def _compute_K2_freq(self, w):
        """
        Evaluates the second order cumulant in the frequency domain.
        """

        # Avoids double computation
        if w in self.k2_freq.keys():
            return self.k2_freq[w]
            
        # Computes the cumulant. Uses lambda function for efficient mapping
        #f = lambda x: _delta_tau(x, 0, 10**(-5))
        Aw = self.A * _delta_tau(w - self.omega, unitary=self.unitary)
        k2 = np.einsum('ij,ji', Aw, self.A, optimize=True)                           
        self.k2_freq[w] = k2/self.D 
        
        return self.k2_freq[w]

    def compute_K2_freq(self, w):
        """Vectorizes the method."""
        return np.vectorize(self._compute_K2_freq, excluded='self')(w)
        
    def _compute_cactus_freq(self, w1, w2=None):
        """
        Evaluates the cactus diagram at frequencies w1 and w2.
        """

        # Avoids double computation
        if w2 is None:
            w2 = w1
        
        if (w1, w2) in self.cactus_freq.keys():
            return self.cactus_freq[(w1, w2)]
        
        # Do I need the conjugates?
        # Computes the cactus diagram
        Aw1 = self.A * _delta_tau(w1 - self.omega, unitary=self.unitary)
        Aw2 = self.A * _delta_tau(w2 - self.omega, unitary=self.unitary)
        cactus = np.einsum('ij,ji,ik,ki', Aw1, self.A, Aw2, self.A, optimize=True) \
                - np.einsum('ij,ji,ij,ji', Aw1, self.A, Aw2, self.A, optimize=True)
        self.cactus_freq[(w1, w2)] = cactus/self.D
        
        return self.cactus_freq[(w1, w2)]
        
    def compute_cactus_freq(self, w1, w2=None):
        """Vectorizes the method."""
        return np.vectorize(self._compute_cactus_freq, excluded='self')(w1, w2)

    # ------------------#
    # Crossing diagrams #
    # ----------------- #
    
    def _compute_crossing_freq(self, w):
        """
        Evaluates the crossing diagrams in the frequency domain.
        """

        # Avoids double computation
        if w in self.crossing_freq.keys():
            return self.crossing_freq[w]
        
        # Computes the diagram
        Aw1 = self.A * _delta_tau(w - self.omega, unitary=self.unitary)
        crossing = np.einsum('ij,ji,ij,ji', Aw1, self.A, self.A, self.A, optimize=False)
        self.crossing_freq[w] = crossing/self.D 
                                     
        return self.crossing_freq[w]
    
    def compute_crossing_freq(self, w):
        """Vectorizes the method."""
        return np.vectorize(self._compute_crossing_freq, excluded='self')(w)

    #----------------#
    # Cumulant K4(t) #
    #----------------#
    
    def _compute_K4_freq(self, w1, w2=None, w3=None):
        """
        Evaluates the fourth order cumulant.
        """
        
        if w2 is None:
            w2 = w1
            
        if w3 is None:
            w3 = -w1     

        Aw1 = self.A * _delta_tau(w1 - self.omega, unitary=self.unitary)
        Aw2 = self.A * _delta_tau(w2 - self.omega, unitary=self.unitary)
        Aw3 = self.A * _delta_tau(w3 - self.omega, unitary=self.unitary)

        # All indices combinations; k = i; j = l; k = i & j = i, respectively
        k4 = np.einsum('ij,jk,kl,li', Aw1, Aw2, Aw3, self.A, optimize=True)\
        - np.einsum('ij,ji,il,li', Aw1, Aw2, Aw3, self.A, optimize=True)\
        - np.einsum('ij,jk,kj,ji', Aw1, Aw2, Aw3, self.A, optimize=True)\
        + np.einsum('ij,ji,ij,ji', Aw1, Aw2, Aw3, self.A, optimize=True) 

        self.k4_freq[(w1, w2, w3)] = k4/self.D 
                                    
        return self.k4_freq[(w1, w2, w3)]

    def compute_K4_freq(self, w1, w2=None, w3=None):
        """Vectorizes the method."""
        return np.vectorize(self._compute_K4_freq, excluded='self')(w1, w2, w3)
        
    ########################
    # ARBITRARY CUMULANTS  #
    ########################

    def _operator_lst(self, op_lst, t_lst):
        """Evaluates the time dependent operators A_1(t_1), A_2(t_2), ..., A_q(t_q)"""

        q = len(op_lst)      
        A_lst = [[]] * q

        # Performs the diagonalization (if unsolved)
        if self.diagonalization is None:
            self.diagonalize()
    
        # Extracts eigenvalues and eigenstates
        _, eig_states = self.diagonalization        
        
        for i in range(q):
            A_eig = eig_states.conj().T @ op_lst[i] @ eig_states
            
            for _ in range(self.D):
                A_eig[_][_] = 0
                
            A_lst[i] =  A_eig * np.exp(self.omega*1j*t_lst[i])
            
        return A_lst    
    
    def compute_OTOC_mixed(self, op_lst, t_lst):
        """Evaluates the OTOC <A_1(t_1) A_2(t_2) ... A_q(t_q)>."""
     
        A_lst = self._operator_lst(op_lst, t_lst)
        
        return np.trace(reduce(np.matmul, A_lst))/self.D

    #----------------#
    # Cactus diagram #
    #----------------#
    
    def compute_cactus_mixed(self, op_lst, t_lst):
               
        A_lst = self._operator_lst(op_lst, t_lst)
        return 1/self.D * np.einsum('ij,ji,ik,ki', *A_lst, optimize=True)

    # ------------------#
    # Crossing diagrams #
    # ----------------- #
    
    def compute_crossing_mixed(self, op_lst, t_lst):
        """Evaluates the crossing diagrams k2(t):=Σ |A_{ij}|^4."""
        
        A_lst = self._operator_lst(op_lst, t_lst) 
        return 1/self.D *  np.einsum('ij,ji,ij,ji', *A_lst, optimize=True)
        
    # -------------- #
    # Cumulant K4(t) #
    # -------------- #
    
    def compute_K4_mixed(self, op_lst, t_lst):

        A_lst = self._operator_lst(op_lst, t_lst) 
        return self.compute_OTOC(t) + self.compute_crossing(t) - 2*self.compute_cactus(t)
        
class FreeMatrix(FreeModel):
    """Calculates the cumulants for RMT theory - with no frequency structure. Normalized by default. Inherits from the FreeModel class."""
    
    def __init__(self, random_matrix, unitary=True, normalized=True):

        # Explicitly using super() to inherit FreeModel.__init__(self, model, unitary) 
        super().__init__(random_matrix, unitary)

        # Overrides these attributes with the given parameters
        self.A = random_matrix
        self.D = len(random_matrix)

        # Standardizes the variance of the matrix
        if normalized:
            self.A = self.A/np.sqrt(self.D)

        # Removes the frequency/time dependence
        self.omega = np.full((self.D, self.D), 0)

# try:
#    myClassInstance.methodName()
# except AttributeError:  
#     raise AttributeError('The cumulants have not been computed')