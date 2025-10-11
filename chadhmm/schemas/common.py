from enum import Enum


class Transitions(Enum):
    SEMI = "Semi-Markov"
    ERGODIC = "Ergodic"
    LEFT_TO_RIGHT = "Left-to-Right"


class InformCriteria(Enum):
    AIC = "Akaike Information Criteria"
    BIC = "Bayesian Information Criteria"
    HQC = "Hannan-Quinn Information Criteria"


class CovarianceType(Enum):
    FULL = "Full"
    DIAG = "Diagonal"
    TIED = "Tied"
    SPHERICAL = "Spherical"


class DecodingAlgorithm(Enum):
    VITERBI = "Viterbi"
    MAP = "MAP"
