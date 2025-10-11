from chadhmm.distributions import MultinomialDistribution
from chadhmm.hmm.BaseHMM import BaseHMM
from chadhmm.schemas import Transitions


class MultinomialHMM(BaseHMM):
    """
    Multinomial Hidden Markov Model using composition pattern.

    Hidden Markov model with multinomial (discrete) emissions.

    Special cases:
    - If n_trials=1 and n_features=2: Bernoulli distribution
    - If n_trials=1 and n_features>2: Categorical distribution
    - If n_trials>1 and n_features=2: Binomial distribution
    - If n_trials>1 and n_features>2: Multinomial distribution

    Parameters:
    ----------
    n_states : int
        Number of hidden states in the model.
    n_features : int
        Number of possible emission values (categories).
    n_trials : int
        Number of trials in the multinomial distribution.
    alpha : float
        Dirichlet concentration parameter for the prior over initial
        distribution, transition and emission probabilities.
    seed : int | None
        Random seed for reproducibility.

    Example:
    --------
    >>> # Categorical emissions (n_trials=1)
    >>> model = MultinomialHMM(n_states=3, n_features=5, n_trials=1)
    >>> model.fit(X)
    >>> predictions = model.predict(X, algorithm='viterbi')
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        transitions=None,  # For backward compatibility
        n_trials: int = 1,
        alpha: float = 1.0,
        seed: int | None = None,
    ):
        # Store parameters needed for distribution initialization
        self.n_features = n_features
        self.n_trials = n_trials

        # Handle transitions parameter (backward compatibility)
        if transitions is None:
            transitions = Transitions.ERGODIC

        # Create initial emission distribution
        emission_dist = MultinomialDistribution.sample_distribution(
            n_components=n_states,
            n_features=n_features,
            X=None,
            n_trials=n_trials,
            alpha=alpha,
        )

        # Initialize BaseHMM with the emission distribution
        super().__init__(
            n_states=n_states,
            emission_dist=emission_dist,
            alpha=alpha,
            transitions=transitions,
            seed=seed,
        )

    def __str__(self) -> str:
        """Returns a string representation of the model."""
        return (
            f"MultinomialHMM(n_states={self.n_states}, "
            f"n_features={self.n_features}, "
            f"n_trials={self.n_trials})"
        )

    def __repr__(self) -> str:
        """Returns a detailed representation of the model."""
        return self.__str__()
