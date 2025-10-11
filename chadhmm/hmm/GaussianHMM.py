from chadhmm.distributions import GaussianDistribution
from chadhmm.hmm.BaseHMM import BaseHMM
from chadhmm.schemas import CovarianceType, Transitions


class GaussianHMM(BaseHMM):
    """
    Gaussian Hidden Markov Model using composition pattern.

    This model assumes that the data follows a multivariate Gaussian distribution.
    The model parameters (initial state probabilities, transition probabilities,
    emission means, and emission covariances) are learned using the
    Baum-Welch algorithm.

    Parameters:
    ----------
    n_states : int
        Number of hidden states in the model.
    n_features : int
        Number of features in the emission data.
    covariance_type : CovarianceType
        Type of covariance parameters to use for the emission distributions.
    k_means : bool
        Whether to use k-means clustering to initialize the emission means.
    alpha : float
        Dirichlet concentration parameter for the prior over initial state
        probabilities and transition probabilities.
    min_covar : float
        Floor value for covariance matrices.
    seed : int | None
        Random seed to use for reproducible results.

    Example:
    --------
    >>> model = GaussianHMM(n_states=3, n_features=2, covariance_type='full')
    >>> model.fit(X)
    >>> predictions = model.predict(X, algorithm='viterbi')
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        covariance_type: CovarianceType,
        alpha: float = 1.0,
        transitions: Transitions = Transitions.ERGODIC,
        k_means: bool = False,
        min_covar: float = 1e-3,
        seed: int | None = None,
    ):
        # Store parameters needed for distribution initialization
        self.n_features = n_features
        self.k_means = k_means
        self.min_covar = min_covar
        self.covariance_type = covariance_type

        # Create initial emission distribution
        emission_dist = GaussianDistribution.sample_distribution(
            n_components=n_states,
            n_features=n_features,
            X=None,
            k_means=k_means,
            min_covar=min_covar,
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
            f"GaussianHMM(n_states={self.n_states}, "
            f"n_features={self.n_features}, "
            f"covariance_type={self.covariance_type})"
        )

    def __repr__(self) -> str:
        """Returns a detailed representation of the model."""
        return self.__str__()
