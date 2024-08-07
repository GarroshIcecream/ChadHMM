# Chad Hidden Markov Models (ChadHMM)
[![GitHub license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/drkostas/pyemail-sender/master/LICENSE)
![Downloads](https://img.shields.io/pypi/dm/chadhmm?color=purple)

> **NOTE:**
> This package is still in its early stages, documentation might not reflect every method mentioned above, please feel free to contribute and make this more coherent.

## Table of Contents

- [Chad Hidden Markov Models (ChadHMM)](#chad-hidden-markov-models-chadhmm)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
  - [Usage ](#usage-)
  - [Roadmap ](#roadmap-)
  - [Unit Tests ](#unit-tests-)
  - [References ](#references-)
  - [License ](#license-)

## About <a name = "about"></a>

This repository was created as an attempt to learn and recreate the parameter estimation for Hidden Markov Models using PyTorch library. Included are models with Categorical and Gaussian emissions for both Hidden Markov Models (`HMM`) and Hidden Semi-Markov Models(`HSMM`). As en extension I am trying to include models where the parameter estimation depends on certain set of external variables, these models are referred to as Contextual HMM or Parametric/Conditional HMM where the emission probabilities/distribution parameters are influenced by the context either time dependent or independent.

The documentation on the parameter estimation and model description is captured in - now empty - [docs](https://github.com/GarroshIcecream/ChadHMM//tree/master/docs) folder. Furthermore, there are [examples](https://github.com/GarroshIcecream/ChadHMM//tree/master/tests) of the usage, especially on the financial time series, focusing on the sequence prediction but also on the possible interpretation of the model parameters.

## Getting Started <a name = "getting_started"></a>

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

1. Install from PyPi server
   ```bash
   $ pip install chadhmm
   ```
2. Clone the repo
   ```bash
   $ git clone https://github.com/GarroshIcecream/ChadHMM.git
   ```

## Usage <a name = "usage"></a>

Please refer to the [docs](https://github.com/GarroshIcecream/ChadHMM//tree/master/docs) for more detailed guide on how to create, train and predict sequences using Hidden Markov Models. There is also a section dedicated to visualizing the model parameters as well as its sequence predictions.

See below example of training and inference using `MultinomialHMM`:
```python
from chadhmm import MultinomialHMM
from chadhmm.utilities import constraints
import torch

# Initialize Multinomial HMM with 6 states and 4 emissions
hmm = MultinomialHMM(
  n_states=6,
  n_features=4,
  n_trials=2,
  transitions=constraints.Transitions.ERGODIC
)

# Mock the example data and one hot encode
train_seq = torch.randint(0,hmm.n_features,(1000,))
one_hot = hmm.n_trials * torch.nn.functional.one_hot(train_seq,4)

# fit the model using EM algorithm assuming two sequences of lenghts 400 and 600
# also fit the model only once (n_init=1)
hmm.fit(X=one_hot,max_iter=5,lengths=[400,600],n_init=1,verbose=False)

# Compute log likelihood of generated sequence (set by_sample=False for joint log likelihood)
log_likes = hmm.score(
  one_hot,
  lengths=[400,500,100], 
  by_sample=True
)
print(log_likes)

# Compute Akeike Information criteria for each sequence (AIC, BIC or HQC)
ics = hmm.ic(
  one_hot,
  lengths=[400,500,100],
  criterion=constraints.InformCriteria.AIC
)
print(ics)

# Get the most likely sequence using Viterbi algorithm (MAP also available)
viterbi_path = hmm.predict(
  one_hot,
  lengths=[400,500,100],
  algorithm='viterbi'
)
print(viterbi_path)
```

## Roadmap <a name = "roadmap"></a>

- [x] Hidden Semi Markov Models
  - [x] Fix computation of posteriors
  - [ ] Implementation of Viterbi algorithm for HSMM
- [ ] Integration of contextual models
  - [ ] Time dependent context
  - [ ] Contextual Variables for covariances using GEM (Generalized Expectation Maximization algorithm)
  - [ ] Contextual variables for Multinomial emissions
- [ ] Use Wrapped Distributions instead of Tensors with parameters - instead of Tensor with logits use Categorical distribution  
  - [ ] Implement different types of covariance matrices
  - [ ] Connect that with degrees of freedom
- [ ] Improve the docs with examples
    - [ ] Application on financial time series prediction
- [ ] Support for CUDA training
- [x] Support different types of Transition Matrices - semi, left-to-right and ergodic
- [x] Support for wider range of emissions distributions
- [x] K-Means for Gaussian means initialization
- [x] Code base refactor, abstractions might be confusing

See the [open issues](https://github.com/GarroshIcecream/ChadHMM/issues) for a full list of proposed features (and known issues).

## Unit Tests <a name = "unit_tests"></a>

If you want to run the unit tests, execute the following command:

```ShellSession
$ make tests
```

## References <a name = "references"></a>

Implementations are based on:

- Hidden Markov Models (HMM):
   - ["A tutorial on hidden Markov models and selected applications in speech recognition"](https://ieeexplore.ieee.org/document/18626) by Lawrence Rabiner from Rutgers University

- Hidden Semi-Markov Models (HSMM):
   - ["An efficient forward-backward algorithm for an explicit-duration hidden Markov model"](https://www.researchgate.net/publication/3342828_An_efficient_forward-backward_algorithm_for_an_explicit-duration_hidden_Markov_model) by Hisashi Kobayashi from Princeton University

- Contextual HMM and HSMM:
  - ["Contextual Hidden Markov Models"](https://www.researchgate.net/publication/261490802_Contextual_Hidden_Markov_Models) by Thierry Artieres from Ecole Centrale de Marseille

## License <a name = "license"></a>

This project is licensed under the MIT License - see
the [LICENSE](https://github.com/GarroshIcecream/ChadHMM/blob/master/LICENSE) file for details.


<a href="https://www.buymeacoffee.com/adpesek13n" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>