from enum import Enum


class Transitions(Enum):
    SEMI = "semi"
    ERGODIC = "ergodic"
    LEFT_TO_RIGHT = "left-to-right"


class InformCriteria(Enum):
    AIC = "AIC"
    BIC = "BIC"
    HQC = "HQC"


class CovarianceType(Enum):
    FULL = "full"
    DIAG = "diag"
    TIED = "tied"
    SPHERICAL = "spherical"


class DecodingAlgorithm(Enum):
    VITERBI = "viterbi"
    MAP = "map"
