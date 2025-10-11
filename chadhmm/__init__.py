"""
ChadHMM
======

Ultra Chad Implementation of Hidden Markov Models in Pytorch
(available only to true sigma males)

But seriously this package needs you to help me make it better.
I'm not a professional programmer, I'm just a guy who likes to code.
If you have any suggestions, please let me know. I'm open to all ideas.
"""

from .hmm import HMM
from .hsmm import GaussianHSMM, GaussianMixtureHSMM, MultinomialHSMM, PoissonHSMM
from .schemas import (
    ContextualVariables,
    CovarianceType,
    DecodingAlgorithm,
    InformCriteria,
    Observations,
    Transitions,
)
from .utils import ConvergenceHandler, SeedGenerator

__all__ = [
    "HMM",
    "GaussianHSMM",
    "PoissonHSMM",
    "MultinomialHSMM",
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
