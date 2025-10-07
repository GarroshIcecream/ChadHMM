"""
ChadHMM
======

Ultra Chad Implementation of Hidden Markov Models in Pytorch
(available only to true sigma males)

But seriously this package needs you to help me make it better.
I'm not a professional programmer, I'm just a guy who likes to code.
If you have any suggestions, please let me know. I'm open to all ideas.
"""

from .hmm import (
    GaussianHMM, 
    GaussianMixtureHMM, 
    MultinomialHMM, 
    PoissonHMM
)
from .hsmm import (
    GaussianHSMM, 
    GaussianMixtureHSMM, 
    MultinomialHSMM, 
    PoissonHSMM
)
from .schemas import (
    Observations, 
    ContextualVariables, 
    DecodingAlgorithm, 
    InformCriteria, 
    Transitions, 
    CovarianceType
)
from .utils import ConvergenceHandler, SeedGenerator

__all__ = [
    "MultinomialHMM",
    "MultinomialHSMM",
    "GaussianHMM",
    "GaussianHSMM",
    "PoissonHMM",
    "PoissonHSMM",
    "GaussianMixtureHMM",
    "GaussianMixtureHSMM",
    "Observations",
    "ContextualVariables",
    "DecodingAlgorithm",
    "InformCriteria",
    "Transitions",
    "CovarianceType",
    "constraints",
    "SeedGenerator",
    "ConvergenceHandler",
]
