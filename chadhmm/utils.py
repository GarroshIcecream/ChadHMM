from typing import Tuple, Optional, List, Union, Literal
from dataclasses import dataclass, field

import torch
import matplotlib.pyplot as plt # type: ignore

DECODERS = frozenset(('viterbi', 'map'))
INFORM_CRITERIA = frozenset(('AIC', 'BIC', 'HQC'))
SUPPORTED_A = frozenset(('semi','left-to-right','ergodic'))


@dataclass
class Observations:
    """Dataclass for a sequence of observations."""
    data: torch.Tensor
    n_samples: int
    log_probs: torch.Tensor
    start_idx: torch.Tensor
    end_idx: torch.Tensor
    lengths: List[int]
    n_sequences: int = field(default=1)

@dataclass
class ContextualVariables:
    """Dataclass for contextual variables."""
    n_context: int
    X: Tuple[torch.Tensor,...]
    time_dependent: bool = field(default=False)

class SeedGenerator:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed_gen = torch.random.manual_seed(seed)
        else:
            initial_seed = torch.random.seed()
            self.seed_gen = torch.random.manual_seed(initial_seed)

    def __repr__(self) -> str:
        return f"SeedGenerator(seed={self()})"

    def __call__(self) -> int:
        return self.seed_gen.seed()
    
    @property
    def seed(self) -> int:
        return self.seed_gen.initial_seed()
    
    @seed.setter
    def seed(self, seed: int) -> None:
        self.seed_gen = torch.random.manual_seed(seed)


class ConvergenceHandler:
    """
    Convergence Monitor
    ----------
    Convergence monitor for HMM training. Stores the score at each iteration and checks for convergence.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.
    n_init : int
        Number of initializations.
    tol : float
        Convergence threshold.
    post_conv_iter : int
        Number of iterations to run after convergence.
    verbose : bool
        Print convergence information.
    """

    def __init__(self, 
                 max_iter:int,
                 n_init:int, 
                 tol:float, 
                 post_conv_iter:int,
                 verbose:bool = True):
        
        self.tol = tol
        self.verbose = verbose
        self.post_conv_iter = post_conv_iter
        self.max_iter = max_iter
        self.score = torch.full(size=(max_iter+1,n_init),
                                fill_value=float('nan'), 
                                dtype=torch.float64)
        self.delta = self.score.clone()

    def __repr__(self):
        return f"""
        ConvergenceHandler(tol={self.tol},
                            n_iters = {self.max_iter+1},
                            post_conv_iter={self.post_conv_iter},
                            converged={self.converged},
                            verbose={self.verbose})
                """
    
    def push_pull(self, new_score:torch.Tensor, iter:int, rank:int) -> bool:
        """Push a new score and check for convergence."""
        self.push(new_score, iter, rank)
        return self.converged(iter, rank)
        
    def push(self, new_score:torch.Tensor, iter:int, rank:int):
        """Update the iteration count."""
        self.score[iter,rank] = new_score
        self.delta[iter,rank] = new_score - self.score[iter-1,rank]

    def converged(self, iter:int, rank:int) -> bool:
        """Check if the model has converged and update the convergence monitor."""
        conv_lag = iter-self.post_conv_iter

        if conv_lag < 0:
            self.is_converged = False
        elif torch.all(self.delta[conv_lag:iter, rank] < self.tol):
            self.is_converged = True
        else:
            self.is_converged = False

        if self.verbose:
            score = self.score[iter,rank].item()
            delta = self.delta[iter,rank].item()

            if self.is_converged:
                print(f'Model converged after {iter} iterations with log-likelihood: {score:.2f}')
            elif iter == 0:
                print(f"Run {rank+1} | Initialization | Score: {score:.2f}")
            else:
                print(f"Run {rank+1} | " +
                      f"Iteration: {iter} | " + 
                      f"Score: {score:.2f} | " +
                      f"Delta: {delta:.2f} | " +
                      f"Converged = {self.is_converged}"
                    )

        return self.is_converged
    
    def plot_convergence(self):
        # Define input for plot
        labels = [f'Log-likelihood - Run #{i+1}' for i in range(self.score.shape[1])]

        # Plot setting
        plt.style.use('ggplot')
        _, ax = plt.subplots(figsize=(10, 7))
        ax.plot(torch.arange(self.max_iter+1), 
                self.score.cpu(), 
                linewidth=2, 
                marker='o', 
                markersize=5, 
                label=labels)
        
        ax.set_title('HMM Model Log-Likelihood Convergence')
        ax.set_xlabel('# Iterations')
        ax.set_ylabel('Log-likelihood')
        ax.legend(loc='lower right')
        plt.show()

def sample_probs(prior:float, 
                 target_size:Union[Tuple[int,...],torch.Size]) -> torch.Tensor:
    """Initialize a matrix of probabilities"""
    alphas = torch.full(size=target_size,
                        fill_value=prior,
                        dtype=torch.float64)
    
    probs = torch.distributions.Dirichlet(alphas).sample()
    return probs

def sample_A(prior:float, 
             n_states:int,
             A_type:Literal['semi','left-to-right','ergodic']='ergodic') -> torch.Tensor:
    """Initialize Transition Matrix from Dirichlet distribution, prior of 1 refers to Uniform sampling"""
    if A_type not in SUPPORTED_A:
        raise NotImplementedError(f'This type of Transition matrix is not supported {A_type} please use {SUPPORTED_A}')
    
    probs = sample_probs(prior, (n_states,n_states))
    if A_type == 'ergodic':
        pass
    elif A_type == 'semi':
        probs.fill_diagonal_(0)
        probs /= probs.sum(dim=-1,keepdim=True)
    elif A_type == 'left-to-right':
        probs = torch.triu(probs)
        probs /= probs.sum(dim=-1,keepdim=True)
    else:
        raise NotImplementedError(f'This type of Transition matrix is not supported: {A_type}')

    return probs

def is_valid_A(logits:torch.Tensor,
               A_type:str='ergodic') -> bool:
    """Check the constraints on the Transition Matrix given its type"""
    if A_type not in SUPPORTED_A:
        raise NotImplementedError(f'This type of Transition matrix is not supported {A_type} please use {SUPPORTED_A}')
    
    return {
        'semi': bool(torch.all(logits.exp().diagonal() == 0)),
        'ergodic': bool(torch.all(logits.exp() > 0.0)),
        'left-to-right': bool(torch.all(logits.exp() > 0.0))
    }[A_type]

def log_normalize(matrix:torch.Tensor, dim:Union[int,Tuple[int,...]]=1) -> torch.Tensor:
    """Normalize a posterior probability matrix"""
    return matrix - matrix.logsumexp(dim,True)    

def validate_covars(covars: torch.Tensor, 
                    covariance_type: str, 
                    n_features: int) -> torch.Tensor:
    """Do basic checks on matrix covariance sizes and values"""
    if covariance_type == 'spherical':
        if len(covars) != n_features:
            raise ValueError("'spherical' covars have length n_features")
        elif torch.any(covars <= 0): 
            raise ValueError("'spherical' covars must be positive")
    elif covariance_type == 'tied':
        if covars.shape[0] != covars.shape[1]:
            raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
        elif (not torch.allclose(covars, covars.T) or torch.any(covars.symeig(eigenvectors=False).eigenvalues <= 0)):
            raise ValueError("'tied' covars must be symmetric, positive-definite")
    elif covariance_type == 'diag':
        if len(covars.shape) != 2:
            raise ValueError("'diag' covars must have shape (n_features, n_dim)")
        elif torch.any(covars <= 0):
            raise ValueError("'diag' covars must be positive")
    elif covariance_type == 'full':
        if len(covars.shape) != 3:
            raise ValueError("'full' covars must have shape (n_features, n_dim, n_dim)")
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError("'full' covars must have shape (n_features, n_dim, n_dim)")
        for n, cv in enumerate(covars):
            eig_vals, _ = torch.linalg.eigh(cv)
            if (not torch.allclose(cv, cv.T) or torch.any(eig_vals <= 0)):
                raise ValueError(f"component {n} of 'full' covars must be symmetric, positive-definite")
    else:
        raise NotImplementedError(f"This covariance type is not implemented: {covariance_type}")
    
    return covars
       
def init_covars(tied_cv: torch.Tensor, 
                covariance_type: str, 
                n_states: int) -> torch.Tensor:
    """Initialize covars to a given covariance type"""
    if covariance_type == 'spherical':
        return tied_cv.mean() * torch.ones((n_states,))
    elif covariance_type == 'tied':
        return tied_cv
    elif covariance_type == 'diag':
        return tied_cv.diag().unsqueeze(0).expand(n_states, -1)
    elif covariance_type == 'full':
        return tied_cv.unsqueeze(0).expand(n_states, -1, -1)
    else:
        raise NotImplementedError(f"This covariance type is not implemented: {covariance_type}")
    
def fill_covars(covars: torch.Tensor, 
                covariance_type: str, 
                n_states: int, 
                n_features: int,
                n_components: Optional[int]=None) -> torch.Tensor:
    """Fill in missing values for covars"""
    if covariance_type == 'full':
        return covars
    elif covariance_type == 'diag':
        return torch.stack([torch.diag(covar) for covar in covars])
    elif covariance_type == 'tied':
        return covars.unsqueeze(0).expand(n_states, -1, -1)
    elif covariance_type == 'spherical':
        eye = torch.eye(n_features).unsqueeze(0)
        return eye * covars.unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError(f"This covariance type is not implemented: {covariance_type}")

