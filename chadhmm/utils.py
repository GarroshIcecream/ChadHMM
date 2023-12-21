from typing import Generator, Sequence, Tuple, Optional, List, Generator, Dict, Callable, Union
from dataclasses import dataclass, field

import torch
from torch.multiprocessing import Process, Queue
import matplotlib.pyplot as plt # type: ignore
from prettytable import PrettyTable # type: ignore

MAT_TYPES = frozenset(('Transition','Emission','Duration','Weights','Matrix','Tensor','Initial','Vector'))
DECODERS = frozenset(('viterbi', 'map'))
INFORM_CRITERIA = frozenset(('AIC', 'BIC', 'HQC'))


@dataclass
class FittedModel:
    """Dataclass for a fitted model"""
    model: str
    n_fit_params: Dict[str,int]
    DOF: int
    converged: bool
    log_likelihood: float
    AIC: float
    params: Dict[str,torch.Tensor]

@dataclass
class Observations:
    """Dataclass for a sequence of observations."""
    n_samples: int
    X: List[torch.Tensor]
    log_probs: List[torch.Tensor]
    lengths: List[int]
    n_sequences: int = field(default=1)

@dataclass
class ContextualVariables:
    """Dataclass for contextual variables."""
    n_context: int
    matrix: List[torch.Tensor]
    time_dependent: bool = field(default=False)


class Multiprocessor:

    def __init__(self):
        self.processes = []
        self.queue = Queue()

    @staticmethod
    def _wrapper(func:Callable, queue:Queue, args, kwargs):
        ret = func(*args, **kwargs)
        queue.put(ret)

    def run(self, func:Callable, name:str, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = Process(target=self._wrapper, name=name, args=args2)
        self.processes.append(p)
        p.start()

    def wait(self):
        rets = []
        for p in self.processes:
            ret = self.queue.get()
            rets.append(ret)
        for p in self.processes:
            p.join()
        return rets

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
                 device:torch.device, 
                 verbose:bool = True):
        
        self.tol = tol
        self.verbose = verbose
        self.post_conv_iter = post_conv_iter
        self.device = device
        self.max_iter = max_iter
        self.score = torch.full(size=(max_iter+1,n_init),
                                fill_value=float('nan'), 
                                dtype=torch.float64,
                                device=self.device)
        self.delta = self.score.clone()

    def __repr__(self):
        return f"""
        ConvergenceHandler(tol={self.tol},
                            n_iters = {self.iter+1},
                            post_conv_iter={self.post_conv_iter},
                            converged={self.converged},
                            verbose={self.verbose})
                """
    
    def push_pull(self, new_score:float, iter:int, rank:int) -> bool:
        """Push a new score and check for convergence."""
        self.push(new_score, iter, rank)
        return self.converged(iter, rank)
        
    def push(self, new_score:float, iter:int, rank:int):
        """Update the iteration count."""
        self.score[iter,rank] = new_score
        self.delta[iter,rank] = new_score - self.score[iter-1,rank]

    def converged(self, iter:int, rank:int) -> bool:
        """Check if the model has converged and update the convergence monitor."""
        conv_lag = iter-self.post_conv_iter

        if conv_lag < 0:
            self.is_converged = False
        elif all(self.delta[conv_lag:iter, rank] < self.tol):
            self.is_converged = True
        else:
            self.is_converged = False

        if self.verbose:
            score = self.score[iter,rank].item()
            delta = self.delta[iter,rank].item()

            if iter == 0:
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


def laplace_smoothing(log_matrix: torch.Tensor, k: float) -> torch.Tensor:
    """Laplace smoothing of a log probability matrix"""

    if k <= 0.0:  
        return log_matrix
    else:
        real_matrix = torch.exp(log_matrix) 
        smoothed_log_matrix = torch.log((real_matrix + k) / (1 + k * real_matrix.shape[-1]))
        return smoothed_log_matrix

def validate_logits(matrix:torch.Tensor, name:str) -> torch.Tensor:    

    if torch.any(torch.isnan(matrix)):
        raise ValueError("Matrix must not contain NaNs")
    elif torch.any(0 <= matrix):
        raise ValueError("Logits must be <= 0")
    
    if name in ('Transition','Emission','Duration','Weights','Matrix'):
        assert matrix.ndim == 2, ValueError(f'Probability Matrix of type {name} must be 2-dimensional')
        if name == 'Transition':
            n_rows, n_cols = matrix.shape
            assert n_cols == n_rows, ValueError(f'For Transition Matrix specify rectangular probability matrix not {n_rows}x{n_cols}') 
    elif name in ('Initial','Vector'):
        assert matrix.ndim == 1, ValueError(f'Probability Vector of type {name} must be 1-dimensional')
    elif name == 'Tensor':
        assert matrix.ndim > 2, ValueError(f'Probability Tensor of type {name} must have more than 2-dimensions')
    else:
        raise NotImplementedError(f'This matrix type is not supported, please use one of {MAT_TYPES}')
    
    if not torch.all(matrix.logsumexp(-1) < 1e-3):
        raise ValueError('Probabilities alongside last dimension must sum up to 1')
    else:
        return matrix

def sample_logits(prior: float, target_size: Union[Tuple[int,...],torch.Size], device:torch.device, semi: bool = False) -> torch.Tensor:
    """Initialize a matrix of probabilities"""
    alphas = torch.full(size=target_size,
                        fill_value=prior,
                        dtype=torch.float64,
                        device=device)
    
    probs = torch.distributions.Dirichlet(alphas).sample()
    if semi:
        # Usable only for 2D
        probs.fill_diagonal_(0)
        probs /= probs.sum(dim=-1, keepdim=True)

    return probs.log()

def log_normalize(matrix: torch.Tensor, dim:int=0) -> torch.Tensor:
    """Normalize a posterior probability matrix"""
    return matrix - matrix.logsumexp(dim,True)

def sequence_generator(X: Observations) -> Generator[Tuple[int,torch.Tensor,torch.Tensor], None, None]:
    for X_len,seq,log_probs in zip(X.lengths,X.X,X.log_probs):
        yield X_len, seq, log_probs

def validate_sequence(sequence:torch.Tensor, 
                      discrete:bool, 
                      n_features:int) -> torch.Tensor:
    """Do basic checks on sequence dimensions and values. Return a dataclass with the sequence and its properties."""
    if discrete:
        if (n_dim:=sequence.ndim) != 1:
            raise ValueError(f'Sequence must be 1-dimensional, got shape {n_dim}')
        elif sequence.dtype != torch.long:
            raise ValueError(f'Sequence must be of type torch.long, got {sequence.dtype}')
        elif torch.max(sequence).item() > (n_features-1):
            raise ValueError('Invalid emission in sequence')
        else:
            return sequence
    else:
        if (n_dim:=sequence.ndim) != 2:
            raise ValueError(f'Sequence must be 2-dimensional, got {n_dim} dims.')
        elif sequence.dtype != torch.double:
            raise ValueError(f'Sequence must be of type torch.float, got {sequence.dtype}')
        elif (T:=sequence.shape[1]) != n_features:
            raise ValueError(f'Second dimension corresponds to {n_features} model features. Got {T} features.')
        else:
            return sequence           
    
def validate_lambdas(lambdas: torch.Tensor, n_states: int, n_features: int) -> torch.Tensor:
    """Do basic checks on matrix mean sizes and values"""
    
    if len(lambdas.shape) != 2:
        raise ValueError("lambdas must have shape (n_states, n_features)")
    elif lambdas.shape[0] != n_states:
        raise ValueError("lambdas must have shape (n_states, n_features)")
    elif lambdas.shape[1] != n_features:
        raise ValueError("lambdas must have shape (n_states, n_features)")
    elif torch.any(torch.isnan(lambdas)):
        raise ValueError("lambdas must not contain NaNs")
    elif torch.any(torch.isinf(lambdas)):
        raise ValueError("lambdas must not contain infinities")
    elif torch.any(lambdas <= 0):
        raise ValueError("lambdas must be positive")
    else:
        return lambdas

def validate_covars(covars: torch.Tensor, 
                    covariance_type: str, 
                    n_states: int, 
                    n_features: int,
                    n_components: Optional[int]=None) -> torch.Tensor:
    """Do basic checks on matrix covariance sizes and values"""
    if n_components is None:
        valid_shape = torch.Size((n_states, n_features, n_features))
    else:
        valid_shape = torch.Size((n_states, n_components, n_features, n_features))    

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

def print_table(rows: list, header: list, title: str):
    """
    Helper method for the pretty print function. It prints the parameters
    as a nice table.
    """
    t = PrettyTable(title=title, 
                    field_names=header, 
                    header_style='upper',
                    padding_width=1, 
                    title_style='upper')
    
    for row in rows:
        t.add_row(row)
    
    print(t)

