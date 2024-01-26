"""
ChadHMM
======

Ultra Chad Implementation of Hidden Markov Models in Pytorch (available only to true sigma males)

But seriously this package needs you to help me make it better. I'm not a professional programmer, I'm just a guy who likes to code. 
If you have any suggestions, please let me know. I'm open to all ideas.
"""

__all__ = ['MultinomialHMM', 
           'MultinomialHSMM', 
           'GaussianHMM',
           'GaussianHSMM',
           'PoissonHMM',
           'PoissonHSMM',
           'GaussianMixtureHMM',
           'GaussianMixtureHSMM']

# Import HMM objects
from .hmm import MultinomialHMM, GaussianHMM, GaussianMixtureHMM, PoissonHMM
from .hsmm import MultinomialHSMM, GaussianHSMM, GaussianMixtureHSMM, PoissonHSMM