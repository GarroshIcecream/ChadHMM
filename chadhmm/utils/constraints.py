import numpy as np
import torch
from typing import Tuple, Union, Optional

from chadhmm.schemas.common import Transitions, InformCriteria, CovarianceType


def sample_probs(
    prior: float, target_size: Tuple[int, ...] | torch.Size
) -> torch.Tensor:
    """Initialize a matrix of probabilities"""
    alphas = torch.full(size=target_size, fill_value=prior, dtype=torch.float64)

    probs = torch.distributions.Dirichlet(alphas).sample()
    return probs


def sample_A(prior: float, n_states: int, A_type: Transitions) -> torch.Tensor:
    """Initialize Transition Matrix from Dirichlet distribution, prior of 1
    refers to Uniform sampling"""
    probs = sample_probs(prior, (n_states, n_states))
    if A_type == Transitions.ERGODIC:
        pass
    elif A_type == Transitions.SEMI:
        probs.fill_diagonal_(0)
        probs /= probs.sum(dim=-1, keepdim=True)
    elif A_type == Transitions.LEFT_TO_RIGHT:
        probs = torch.triu(probs)
        probs /= probs.sum(dim=-1, keepdim=True)
    else:
        raise NotImplementedError(
            f"This type of Transition matrix is not supported: {A_type}"
        )

    return probs


def compute_information_criteria(
    samples: int, log_likelihood: torch.Tensor, dof: int, criterion: InformCriteria
) -> torch.Tensor:
    """Compute the information criteria for a given model."""
    match criterion:
        case InformCriteria.AIC:
            return -2.0 * log_likelihood + 2.0 * dof
        case InformCriteria.BIC:
            return -2.0 * log_likelihood + dof * np.log(samples)
        case InformCriteria.HQC:
            return -2.0 * log_likelihood + 2.0 * dof * np.log(np.log(samples))


def is_valid_A(logits: torch.Tensor, A_type: Transitions) -> bool:
    """Check the constraints on the Transition Matrix given its type"""
    if A_type == Transitions.ERGODIC:
        return bool(torch.all(logits.exp() > 0.0))
    elif A_type == Transitions.SEMI:
        return bool(torch.all(logits.exp().diagonal() == 0))
    elif A_type == Transitions.LEFT_TO_RIGHT:
        return bool(torch.all(logits.exp() > 0.0))
    else:
        raise NotImplementedError(
            f"This type of Transition matrix is not supported: {A_type}"
        )


def log_normalize(matrix: torch.Tensor, dim: Union[int, Tuple[int, ...]]) -> torch.Tensor:
    """Normalize a posterior probability matrix"""
    return matrix - matrix.logsumexp(dim, True)


def validate_lambdas(
    lambdas: torch.Tensor, n_states: int, n_features: int
) -> torch.Tensor:
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


def validate_covars(
    covars: torch.Tensor,
    covariance_type: CovarianceType,
    n_states: int,
    n_features: int,
    n_components: Optional[int] = None,
) -> torch.Tensor:
    """Do basic checks on matrix covariance sizes and values"""
    if n_components is None:
        valid_shape = torch.Size((n_states, n_features, n_features))
    else:
        valid_shape = torch.Size((n_states, n_components, n_features, n_features))

    match covariance_type:
        case CovarianceType.SPHERICAL:
            if len(covars) != n_features:
                raise ValueError("'spherical' covars have length n_features")
            elif torch.any(covars <= 0):
                raise ValueError("'spherical' covars must be positive")
        case CovarianceType.TIED:
            if covars.shape[0] != covars.shape[1]:
                raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
            elif not torch.allclose(covars, covars.T) or torch.any(
                covars.symeig(eigenvectors=False).eigenvalues <= 0
            ):
                raise ValueError("'tied' covars must be symmetric, positive-definite")
        case CovarianceType.DIAG:
            if len(covars.shape) != 2:
                raise ValueError("'diag' covars must have shape (n_features, n_dim)")
            elif torch.any(covars <= 0):
                raise ValueError("'diag' covars must be positive")
        case CovarianceType.FULL:
            if covars.shape != valid_shape:
                raise ValueError(
                    "'full' covars must have shape (n_features, n_dim, n_dim)"
                )
            elif covars.shape[1] != covars.shape[2]:
                raise ValueError(
                    "'full' covars must have shape (n_features, n_dim, n_dim)"
                )
            for n, cv in enumerate(covars):
                eig_vals, _ = torch.linalg.eigh(cv)
                if not torch.allclose(cv, cv.T) or torch.any(eig_vals <= 0):
                    raise ValueError(
                        f"component {n} of 'full' covars must be symmetric, "
                        f"positive-definite"
                    )

    return covars


def init_covars(
    tied_cv: torch.Tensor, covariance_type: CovarianceType, n_states: int
) -> torch.Tensor:
    """Initialize covars to a given covariance type"""
    if covariance_type == CovarianceType.SPHERICAL:
        return tied_cv.mean() * torch.ones((n_states,))
    elif covariance_type == CovarianceType.TIED:
        return tied_cv
    elif covariance_type == CovarianceType.DIAG:
        return tied_cv.diag().unsqueeze(0).expand(n_states, -1)
    elif covariance_type == CovarianceType.FULL:
        return tied_cv.unsqueeze(0).expand(n_states, -1, -1)
    else:
        raise NotImplementedError(
            f"This covariance type is not implemented: {covariance_type}"
        )


def fill_covars(
    covars: torch.Tensor,
    covariance_type: CovarianceType,
    n_states: int,
    n_features: int,
    n_components: Optional[int] = None,
) -> torch.Tensor:
    """Fill in missing values for covars"""
    if covariance_type == CovarianceType.FULL:
        return covars
    elif covariance_type == CovarianceType.DIAG:
        return torch.stack([torch.diag(covar) for covar in covars])
    elif covariance_type == CovarianceType.TIED:
        return covars.unsqueeze(0).expand(n_states, -1, -1)
    elif covariance_type == CovarianceType.SPHERICAL:
        eye = torch.eye(n_features).unsqueeze(0)
        return eye * covars.unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError(
            f"This covariance type is not implemented: {covariance_type}"
        )
