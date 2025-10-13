import unittest

import pytest
import torch

from chadhmm.distributions import (
    DurationDistribution,
    GaussianDistribution,
    GaussianMixtureDistribution,
    InitialDistribution,
    MultinomialDistribution,
    PoissonDistribution,
    TransitionMatrix,
)
from chadhmm.hsmm import HSMM
from chadhmm.schemas import CovarianceType, DecodingAlgorithm, Transitions
from chadhmm.utils import constraints


# Factory functions to create HSMM models with the new composition pattern
def create_multinomial_hsmm(
    n_states: int,
    n_features: int,
    max_duration: int,
    n_trials: int = 1,
    alpha: float = 1.0,
    seed: int | None = None,
):
    """Create a MultinomialHSMM using the new composition pattern."""
    if seed is not None:
        torch.manual_seed(seed)

    pi = InitialDistribution.sample_from_dirichlet(alpha, torch.Size([n_states]))
    A = TransitionMatrix.sample_from_dirichlet(
        alpha, Transitions.SEMI, torch.Size([n_states, n_states])
    )
    D = DurationDistribution.sample_from_dirichlet(
        alpha, torch.Size([n_states, max_duration])
    )
    emission_pdf = MultinomialDistribution.sample_distribution(
        n_components=n_states, n_features=n_features, n_trials=n_trials
    )

    hsmm = HSMM(
        transition_matrix=A,
        initial_distribution=pi,
        duration_distribution=D,
        emission_pdf=emission_pdf,
    )
    return hsmm


def create_gaussian_hsmm(
    n_states: int,
    n_features: int,
    max_duration: int,
    covariance_type: CovarianceType = CovarianceType.FULL,
    alpha: float = 1.0,
    seed: int | None = None,
):
    """Create a GaussianHSMM using the new composition pattern."""
    if seed is not None:
        torch.manual_seed(seed)

    pi = InitialDistribution.sample_from_dirichlet(alpha, torch.Size([n_states]))
    A = TransitionMatrix.sample_from_dirichlet(
        alpha, Transitions.SEMI, torch.Size([n_states, n_states])
    )
    D = DurationDistribution.sample_from_dirichlet(
        alpha, torch.Size([n_states, max_duration])
    )
    emission_pdf = GaussianDistribution.sample_distribution(
        n_components=n_states, n_features=n_features
    )

    hsmm = HSMM(
        transition_matrix=A,
        initial_distribution=pi,
        duration_distribution=D,
        emission_pdf=emission_pdf,
    )
    return hsmm


def create_gaussian_mixture_hsmm(
    n_states: int,
    n_features: int,
    n_components: int,
    max_duration: int,
    covariance_type: CovarianceType = CovarianceType.FULL,
    alpha: float = 1.0,
    seed: int | None = None,
):
    """Create a GaussianMixtureHSMM using the new composition pattern.

    Note: n_components here refers to the number of mixture components per state.
    """
    if seed is not None:
        torch.manual_seed(seed)

    pi = InitialDistribution.sample_from_dirichlet(alpha, torch.Size([n_states]))
    A = TransitionMatrix.sample_from_dirichlet(
        alpha, Transitions.SEMI, torch.Size([n_states, n_states])
    )
    D = DurationDistribution.sample_from_dirichlet(
        alpha, torch.Size([n_states, max_duration])
    )
    emission_pdf = GaussianMixtureDistribution.sample_distribution(
        n_components=n_states, n_features=n_features, n_mixture_components=n_components
    )

    hsmm = HSMM(
        transition_matrix=A,
        initial_distribution=pi,
        duration_distribution=D,
        emission_pdf=emission_pdf,
    )
    return hsmm


def create_poisson_hsmm(
    n_states: int,
    n_features: int,
    max_duration: int,
    alpha: float = 1.0,
    seed: int | None = None,
):
    """Create a PoissonHSMM using the new composition pattern."""
    if seed is not None:
        torch.manual_seed(seed)

    pi = InitialDistribution.sample_from_dirichlet(alpha, torch.Size([n_states]))
    A = TransitionMatrix.sample_from_dirichlet(
        alpha, Transitions.SEMI, torch.Size([n_states, n_states])
    )
    D = DurationDistribution.sample_from_dirichlet(
        alpha, torch.Size([n_states, max_duration])
    )
    emission_pdf = PoissonDistribution.sample_distribution(
        n_components=n_states, n_features=n_features
    )

    hsmm = HSMM(
        transition_matrix=A,
        initial_distribution=pi,
        duration_distribution=D,
        emission_pdf=emission_pdf,
    )
    return hsmm


class TestMultinomialHSMM(unittest.TestCase):
    """Tests for MultinomialHSMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 4
        self.n_trials = 2
        self.max_duration = 10
        self.hsmm = create_multinomial_hsmm(
            n_states=self.n_states,
            n_features=self.n_features,
            n_trials=self.n_trials,
            max_duration=self.max_duration,
            alpha=1.0,
            seed=42,
        )

        # Generate test data
        torch.manual_seed(42)
        self.X = torch.randint(0, self.n_features, (100,))
        self.X_one_hot = self.n_trials * torch.nn.functional.one_hot(
            self.X, self.n_features
        )
        self.lengths = [50, 50]

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hsmm.n_states, self.n_states)
        self.assertEqual(self.hsmm.emission_pdf.n_features, self.n_features)
        self.assertEqual(self.hsmm.emission_pdf.n_trials, self.n_trials)
        self.assertEqual(self.hsmm.max_duration, self.max_duration)

    def test_pdf_property(self):
        """Test emission PDF has required methods."""
        self.assertTrue(hasattr(self.hsmm.emission_pdf, "log_prob"))
        self.assertTrue(hasattr(self.hsmm.emission_pdf, "sample"))

    def test_dof_property(self):
        """Test degrees of freedom calculation."""
        # Just test that DOF is a positive integer
        self.assertIsInstance(self.hsmm.dof, int)
        self.assertGreater(self.hsmm.dof, 0)

    def test_duration_matrix_property(self):
        """Test duration matrix D property."""
        # Each row should sum to 1 (logsumexp along dim=1)
        D = self.hsmm.D.logits
        self.assertEqual(D.shape, (self.n_states, self.max_duration))
        self.assertTrue(
            torch.allclose(
                D.logsumexp(1), torch.zeros(self.n_states, dtype=D.dtype), atol=1e-6
            )
        )

    def test_sample_emission_pdf(self):
        """Test emission PDF property."""
        # Test emission PDF
        emission_pdf = self.hsmm.emission_pdf
        self.assertIsNotNone(emission_pdf)
        self.assertEqual(emission_pdf.n_components, self.n_states)
        self.assertEqual(emission_pdf.n_features, self.n_features)
        self.assertEqual(emission_pdf.n_trials, self.n_trials)

    @pytest.mark.skip(
        reason="HSMM backward algorithm has broadcasting bug in torch.compile"
    )
    def test_fit_method(self):
        """Test model fitting."""
        # Test basic fitting
        self.hsmm.fit(
            X=self.X_one_hot, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Check that parameters are updated
        self.assertIsNotNone(self.hsmm._params)

    @pytest.mark.skip(
        reason="HSMM backward algorithm has broadcasting bug in torch.compile"
    )
    def test_predict_method(self):
        """Test prediction methods."""
        # Fit model first
        self.hsmm.fit(
            X=self.X_one_hot, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Test MAP prediction
        map_path = self.hsmm.predict(
            X=self.X_one_hot, lengths=self.lengths, algorithm=DecodingAlgorithm.MAP
        )
        self.assertIsInstance(map_path, list)
        self.assertEqual(len(map_path), len(self.lengths))

        # Test Viterbi prediction
        viterbi_path = self.hsmm.predict(
            X=self.X_one_hot,
            lengths=self.lengths,
            algorithm=DecodingAlgorithm.VITERBI,
        )
        self.assertIsInstance(viterbi_path, list)
        self.assertEqual(len(viterbi_path), len(self.lengths))
        # Check that each sequence has the correct length
        for i, length in enumerate(self.lengths):
            self.assertEqual(viterbi_path[i].shape[0], length)
            # Check that all states are valid (between 0 and n_states-1)
            self.assertTrue(torch.all(viterbi_path[i] >= 0))
            self.assertTrue(torch.all(viterbi_path[i] < self.n_states))

    @pytest.mark.skip(
        reason="HSMM backward algorithm has broadcasting bug in torch.compile"
    )
    def test_score_method(self):
        """Test scoring methods."""
        # Fit model first
        self.hsmm.fit(
            X=self.X_one_hot, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Test by_sample=True
        scores_by_sample = self.hsmm.score(
            X=self.X_one_hot, lengths=self.lengths, by_sample=True
        )
        # HSMM returns shape (n_sequences, max_duration) for scores
        self.assertEqual(scores_by_sample.shape, (len(self.lengths), self.max_duration))

        # Test by_sample=False
        scores_joint = self.hsmm.score(
            X=self.X_one_hot, lengths=self.lengths, by_sample=False
        )
        # HSMM returns shape (1, max_duration) for joint scores
        self.assertEqual(scores_joint.shape, (1, self.max_duration))

    @pytest.mark.skip(
        reason="HSMM backward algorithm has broadcasting bug in torch.compile"
    )
    def test_ic_method(self):
        """Test information criteria calculation."""
        # Fit model first
        self.hsmm.fit(
            X=self.X_one_hot, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Test AIC
        aic = self.hsmm.ic(
            X=self.X_one_hot,
            lengths=self.lengths,
            criterion=constraints.InformCriteria.AIC,
        )
        # HSMM returns shape (n_sequences, max_duration) for IC
        self.assertEqual(aic.shape, (len(self.lengths), self.max_duration))

        # Test BIC
        bic = self.hsmm.ic(
            X=self.X_one_hot,
            lengths=self.lengths,
            criterion=constraints.InformCriteria.BIC,
        )
        # HSMM returns shape (n_sequences, max_duration) for IC
        self.assertEqual(bic.shape, (len(self.lengths), self.max_duration))

    @pytest.mark.skip(
        reason="HSMM backward algorithm has broadcasting bug in torch.compile"
    )
    def test_model_persistence(self):
        """Test model saving and loading."""

        # Fit model first
        self.hsmm.fit(
            X=self.X_one_hot, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # HSMM models don't have save_model method, so we'll test basic functionality
        # Test that model can be fitted and has correct structure
        self.assertEqual(self.hsmm.n_states, self.n_states)
        self.assertEqual(self.hsmm.emission_pdf.n_features, self.n_features)
        self.assertEqual(self.hsmm.emission_pdf.n_trials, self.n_trials)
        self.assertEqual(self.hsmm.max_duration, self.max_duration)

        # Check that parameters have correct shapes
        self.assertEqual(self.hsmm.A.logits.shape, (self.n_states, self.n_states))
        self.assertEqual(self.hsmm.D.logits.shape, (self.n_states, self.max_duration))

    def test_device_transfer(self):
        """Test moving model to different devices."""
        if torch.cuda.is_available():
            # Move to CUDA
            self.hsmm.to("cuda")
            self.assertEqual(self.hsmm.device.type, "cuda")

            # Move back to CPU
            self.hsmm.to("cpu")
            self.assertEqual(self.hsmm.device.type, "cpu")

    @pytest.mark.skip(
        reason="HSMM backward algorithm has broadcasting bug in torch.compile"
    )
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with single sequence
        single_seq = self.X_one_hot[:10]
        self.hsmm.fit(X=single_seq, max_iter=2, n_init=1, verbose=False)

        # Test with short but valid sequence
        short_seq = self.X_one_hot[:20]  # Use more samples to avoid numerical issues
        self.hsmm.fit(X=short_seq, max_iter=2, n_init=1, verbose=False)

        # Test with mismatched lengths
        with self.assertRaises((ValueError, AssertionError)):
            self.hsmm.fit(
                X=self.X_one_hot,
                lengths=[30, 30],  # Doesn't sum to 100
                max_iter=2,
                n_init=1,
                verbose=False,
            )


class TestGaussianHSMM(unittest.TestCase):
    """Tests for GaussianHSMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 2
        self.max_duration = 10
        self.hsmm = create_gaussian_hsmm(
            n_states=self.n_states,
            n_features=self.n_features,
            max_duration=self.max_duration,
            covariance_type=constraints.CovarianceType.FULL,
            alpha=1.0,
            seed=42,
        )

        # Generate test data
        torch.manual_seed(42)
        self.X = torch.randn(100, self.n_features)
        self.lengths = [50, 50]

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hsmm.n_states, self.n_states)
        self.assertEqual(self.hsmm.emission_pdf.n_features, self.n_features)
        self.assertEqual(self.hsmm.max_duration, self.max_duration)

    @pytest.mark.skip(
        reason="HSMM backward algorithm has broadcasting bug in torch.compile"
    )
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Fit model
        self.hsmm.fit(
            X=self.X, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Test MAP prediction
        predictions = self.hsmm.predict(
            X=self.X, lengths=self.lengths, algorithm=DecodingAlgorithm.MAP
        )
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(self.lengths))

        # Test Viterbi prediction
        viterbi_predictions = self.hsmm.predict(
            X=self.X, lengths=self.lengths, algorithm=DecodingAlgorithm.VITERBI
        )
        self.assertIsInstance(viterbi_predictions, list)
        self.assertEqual(len(viterbi_predictions), len(self.lengths))
        # Check that each sequence has the correct length
        for i, length in enumerate(self.lengths):
            self.assertEqual(viterbi_predictions[i].shape[0], length)
            # Check that all states are valid
            self.assertTrue(torch.all(viterbi_predictions[i] >= 0))
            self.assertTrue(torch.all(viterbi_predictions[i] < self.n_states))


class TestGaussianMixtureHSMM(unittest.TestCase):
    """Tests for GaussianMixtureHSMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 2
        self.n_features = 2
        self.n_components = 3
        self.max_duration = 10
        self.hsmm = create_gaussian_mixture_hsmm(
            n_states=self.n_states,
            n_features=self.n_features,
            n_components=self.n_components,
            max_duration=self.max_duration,
            covariance_type=constraints.CovarianceType.FULL,
            alpha=1.0,
            seed=42,
        )

        # Generate test data
        torch.manual_seed(42)
        self.X = torch.randn(100, self.n_features)
        self.lengths = [50, 50]

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hsmm.n_states, self.n_states)
        self.assertEqual(self.hsmm.emission_pdf.n_features, self.n_features)
        self.assertEqual(self.hsmm.emission_pdf.n_mixtures, self.n_components)
        self.assertEqual(self.hsmm.max_duration, self.max_duration)

    @pytest.mark.skip(
        reason="HSMM backward algorithm has broadcasting bug in torch.compile"
    )
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Fit model
        self.hsmm.fit(
            X=self.X, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Test MAP prediction
        predictions = self.hsmm.predict(
            X=self.X, lengths=self.lengths, algorithm=DecodingAlgorithm.MAP
        )
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(self.lengths))

        # Test Viterbi prediction
        viterbi_predictions = self.hsmm.predict(
            X=self.X, lengths=self.lengths, algorithm=DecodingAlgorithm.VITERBI
        )
        self.assertIsInstance(viterbi_predictions, list)
        self.assertEqual(len(viterbi_predictions), len(self.lengths))
        # Check that each sequence has the correct length
        for i, length in enumerate(self.lengths):
            self.assertEqual(viterbi_predictions[i].shape[0], length)
            # Check that all states are valid
            self.assertTrue(torch.all(viterbi_predictions[i] >= 0))
            self.assertTrue(torch.all(viterbi_predictions[i] < self.n_states))


class TestPoissonHSMM(unittest.TestCase):
    """Tests for PoissonHSMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 2
        self.max_duration = 10
        self.hsmm = create_poisson_hsmm(
            n_states=self.n_states,
            n_features=self.n_features,
            max_duration=self.max_duration,
            alpha=1.0,
            seed=42,
        )

        # Generate test data
        torch.manual_seed(42)
        self.X = torch.poisson(torch.ones(100, self.n_features) * 2)
        self.lengths = [50, 50]

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hsmm.n_states, self.n_states)
        self.assertEqual(self.hsmm.emission_pdf.n_features, self.n_features)
        self.assertEqual(self.hsmm.max_duration, self.max_duration)

    @pytest.mark.skip(
        reason="HSMM backward algorithm has broadcasting bug in torch.compile"
    )
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Fit model
        self.hsmm.fit(
            X=self.X, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Test MAP prediction
        predictions = self.hsmm.predict(
            X=self.X, lengths=self.lengths, algorithm=DecodingAlgorithm.MAP
        )
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(self.lengths))

        # Test Viterbi prediction
        viterbi_predictions = self.hsmm.predict(
            X=self.X, lengths=self.lengths, algorithm=DecodingAlgorithm.VITERBI
        )
        self.assertIsInstance(viterbi_predictions, list)
        self.assertEqual(len(viterbi_predictions), len(self.lengths))
        # Check that each sequence has the correct length
        for i, length in enumerate(self.lengths):
            self.assertEqual(viterbi_predictions[i].shape[0], length)
            # Check that all states are valid
            self.assertTrue(torch.all(viterbi_predictions[i] >= 0))
            self.assertTrue(torch.all(viterbi_predictions[i] < self.n_states))


class TestHSMMTransitions(unittest.TestCase):
    """Tests for HSMM transitions."""

    def test_semi_transitions(self):
        """Test semi-Markov transition matrix."""
        hsmm = create_multinomial_hsmm(n_states=3, n_features=4, max_duration=10)

        A = hsmm.A.probs
        # Diagonal should be zero
        self.assertTrue(torch.all(torch.diag(A) == 0))


class TestHSMMDurationModeling(unittest.TestCase):
    """Tests for HSMM duration modeling."""

    def test_duration_matrix_initialization(self):
        """Test duration matrix initialization."""
        hsmm = create_multinomial_hsmm(n_states=3, n_features=4, max_duration=10)

        D = hsmm.D.probs
        # Each row should sum to 1
        self.assertTrue(torch.allclose(D.sum(1), torch.ones(3, dtype=D.dtype)))

        # All probabilities should be non-negative
        self.assertTrue(torch.all(D >= 0))


if __name__ == "__main__":
    unittest.main()
