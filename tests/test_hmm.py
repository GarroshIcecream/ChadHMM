import unittest

import pytest
import torch

from chadhmm.distributions import (
    GaussianDistribution,
    GaussianMixtureDistribution,
    InitialDistribution,
    MultinomialDistribution,
    PoissonDistribution,
    TransitionMatrix,
)
from chadhmm.hmm import HMM
from chadhmm.schemas import DecodingAlgorithm, Transitions
from chadhmm.utils import constraints


# Factory functions to create HMM models with the new composition pattern
def create_multinomial_hmm(
    n_states: int,
    n_features: int,
    transitions: Transitions = Transitions.ERGODIC,
    n_trials: int = 1,
    alpha: float = 1.0,
    seed: int | None = None,
):
    """Create a MultinomialHMM using the new composition pattern."""
    if seed is not None:
        torch.manual_seed(seed)

    pi = InitialDistribution.sample_from_dirichlet(alpha, torch.Size([n_states]))
    A = TransitionMatrix.sample_from_dirichlet(
        alpha, transitions, torch.Size([n_states, n_states])
    )
    emission_pdf = MultinomialDistribution.sample_distribution(
        n_components=n_states, n_features=n_features, n_trials=n_trials
    )

    hmm = HMM(
        transition_matrix=A,
        initial_distribution=pi,
        emission_pdf=emission_pdf,
    )
    return hmm


def create_gaussian_hmm(
    n_states: int,
    n_features: int,
    transitions: Transitions = Transitions.ERGODIC,
    alpha: float = 1.0,
    seed: int | None = None,
):
    """Create a GaussianHMM using the new composition pattern."""
    if seed is not None:
        torch.manual_seed(seed)

    pi = InitialDistribution.sample_from_dirichlet(alpha, torch.Size([n_states]))
    A = TransitionMatrix.sample_from_dirichlet(
        alpha, transitions, torch.Size([n_states, n_states])
    )
    emission_pdf = GaussianDistribution.sample_distribution(
        n_components=n_states, n_features=n_features
    )

    hmm = HMM(
        transition_matrix=A,
        initial_distribution=pi,
        emission_pdf=emission_pdf,
    )
    return hmm


def create_gaussian_mixture_hmm(
    n_states: int,
    n_features: int,
    n_components: int,
    transitions: Transitions = Transitions.ERGODIC,
    alpha: float = 1.0,
    seed: int | None = None,
):
    """Create a GaussianMixtureHMM using the new composition pattern.

    Note: n_components here refers to the number of mixture components per state.
    """
    if seed is not None:
        torch.manual_seed(seed)

    pi = InitialDistribution.sample_from_dirichlet(alpha, torch.Size([n_states]))
    A = TransitionMatrix.sample_from_dirichlet(
        alpha, transitions, torch.Size([n_states, n_states])
    )

    emission_pdf = GaussianMixtureDistribution.sample_distribution(
        n_components=n_states, n_features=n_features, n_mixture_components=n_components
    )

    hmm = HMM(
        transition_matrix=A,
        initial_distribution=pi,
        emission_pdf=emission_pdf,
    )
    return hmm


def create_poisson_hmm(
    n_states: int,
    n_features: int,
    transitions: Transitions = Transitions.ERGODIC,
    alpha: float = 1.0,
    seed: int | None = None,
):
    """Create a PoissonHMM using the new composition pattern."""
    if seed is not None:
        torch.manual_seed(seed)

    pi = InitialDistribution.sample_from_dirichlet(alpha, torch.Size([n_states]))
    A = TransitionMatrix.sample_from_dirichlet(
        alpha, transitions, torch.Size([n_states, n_states])
    )
    emission_pdf = PoissonDistribution.sample_distribution(
        n_components=n_states, n_features=n_features
    )

    hmm = HMM(
        transition_matrix=A,
        initial_distribution=pi,
        emission_pdf=emission_pdf,
    )
    return hmm


class TestBasicMultinomialHMM(unittest.TestCase):
    """Basic tests for MultinomialHMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.hmm = create_multinomial_hmm(2, 3, transitions=Transitions.ERGODIC)

    def test_pdf_subclass(self):
        """Test that emission PDF has log_prob method."""
        self.assertTrue(hasattr(self.hmm.emission_pdf, "log_prob"))

    def test_dof_is_int(self):
        """Test that degrees of freedom is an integer."""
        self.assertIsInstance(self.hmm.dof, int)

    def test_emission_pdf(self):
        """Test emission PDF property."""
        emission_pdf = self.hmm.emission_pdf
        self.assertIsNotNone(emission_pdf)

    def test_emission_pdf_with_data(self):
        """Test emission PDF with data."""
        # For multinomial distributions, we need one-hot encoded data
        X_raw = torch.tensor([0, 1, 2, 0, 1])
        X = torch.nn.functional.one_hot(X_raw, 3).float()
        # Just test that we can create observations
        obs = self.hmm.to_observations(X)
        self.assertIsNotNone(obs)


class TestMultinomialHMM(unittest.TestCase):
    """Tests for MultinomialHMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 4
        self.n_trials = 2
        self.hmm = create_multinomial_hmm(
            n_states=self.n_states,
            n_features=self.n_features,
            n_trials=self.n_trials,
            transitions=Transitions.ERGODIC,
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
        self.assertEqual(self.hmm.n_states, self.n_states)
        self.assertEqual(self.hmm.emission_pdf.n_features, self.n_features)
        self.assertEqual(self.hmm.emission_pdf.n_trials, self.n_trials)

    def test_pdf_property(self):
        """Test emission PDF has required methods."""
        self.assertTrue(hasattr(self.hmm.emission_pdf, "log_prob"))
        self.assertTrue(hasattr(self.hmm.emission_pdf, "sample"))

    def test_dof_property(self):
        """Test degrees of freedom calculation."""
        expected_dof = (
            self.n_states**2 + self.n_states * self.n_features - self.n_states - 1
        )
        self.assertEqual(self.hmm.dof, expected_dof)

    def test_device_property(self):
        """Test device property."""
        # The HMM doesn't have a device property directly, check a parameter instead
        self.assertEqual(self.hmm.pi.device, torch.device("cpu"))

    def test_transition_matrix_properties(self):
        """Test transition matrix A and initial probabilities pi."""
        # Test A property - each row should sum to 1 (logsumexp along dim=1)
        A = self.hmm.A.logits
        self.assertEqual(A.shape, (self.n_states, self.n_states))
        self.assertTrue(
            torch.allclose(
                A.logsumexp(1), torch.zeros(self.n_states, dtype=A.dtype), atol=1e-6
            )
        )

        # Test pi property - vector should sum to 1 (logsumexp along dim=0)
        pi = self.hmm.pi.logits
        self.assertEqual(pi.shape, (self.n_states,))
        self.assertTrue(
            torch.allclose(pi.logsumexp(0), torch.zeros(1, dtype=pi.dtype), atol=1e-6)
        )

    def test_sample_emission_pdf(self):
        """Test emission PDF property."""
        # Test emission PDF
        emission_pdf = self.hmm.emission_pdf
        self.assertIsNotNone(emission_pdf)
        self.assertEqual(emission_pdf.n_components, self.n_states)
        self.assertEqual(emission_pdf.n_features, self.n_features)
        self.assertEqual(emission_pdf.n_trials, self.n_trials)

    @pytest.mark.skip(reason="HMM fit requires setuptools for torch.compile")
    def test_predict_method(self):
        """Test prediction methods."""
        # Fit model first
        self.hmm.fit(
            X=self.X_one_hot, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Test Viterbi prediction
        viterbi_path = self.hmm.predict(
            X=self.X_one_hot, lengths=self.lengths, algorithm=DecodingAlgorithm.VITERBI
        )
        self.assertIsInstance(viterbi_path, list)
        self.assertEqual(len(viterbi_path), len(self.lengths))

        # Test MAP prediction
        map_path = self.hmm.predict(
            X=self.X_one_hot, lengths=self.lengths, algorithm=DecodingAlgorithm.MAP
        )
        self.assertIsInstance(map_path, list)
        self.assertEqual(len(map_path), len(self.lengths))

    @pytest.mark.skip(reason="HMM fit requires setuptools for torch.compile")
    def test_score_method(self):
        """Test scoring methods."""
        # Fit model first
        self.hmm.fit(
            X=self.X_one_hot, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Test by_sample=True
        scores_by_sample = self.hmm.score(
            X=self.X_one_hot, lengths=self.lengths, by_sample=True
        )
        self.assertEqual(scores_by_sample.shape, (len(self.lengths),))

        # Test by_sample=False
        scores_joint = self.hmm.score(
            X=self.X_one_hot, lengths=self.lengths, by_sample=False
        )
        self.assertEqual(scores_joint.shape, (1,))

    @pytest.mark.skip(reason="HMM fit requires setuptools for torch.compile")
    def test_ic_method(self):
        """Test information criteria calculation."""
        # Fit model first
        self.hmm.fit(
            X=self.X_one_hot, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Test AIC
        aic = self.hmm.ic(
            X=self.X_one_hot,
            lengths=self.lengths,
            criterion=constraints.InformCriteria.AIC,
        )
        self.assertEqual(aic.shape, (len(self.lengths),))

        # Test BIC
        bic = self.hmm.ic(
            X=self.X_one_hot,
            lengths=self.lengths,
            criterion=constraints.InformCriteria.BIC,
        )
        self.assertEqual(bic.shape, (len(self.lengths),))

    def test_sample_method(self):
        """Test sampling from the model."""
        sample, states = self.hmm.sample(n_samples=100)
        self.assertEqual(sample.shape, torch.Size([100, self.n_features]))
        self.assertEqual(states.shape, torch.Size([100]))
        self.assertTrue(torch.all(states >= 0))
        self.assertTrue(torch.all(states < self.n_states))

    @pytest.mark.skip(reason="HMM fit requires setuptools for torch.compile")
    def test_model_persistence(self):
        """Test model saving and loading."""
        import os
        import tempfile

        # Fit model first
        self.hmm.fit(
            X=self.X_one_hot, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
            # Test that save doesn't raise error
            torch.save(self.hmm.state_dict(), tmp.name)
            self.assertTrue(os.path.exists(tmp.name))

            # Load model
            new_hmm = create_multinomial_hmm(
                n_states=self.n_states,
                n_features=self.n_features,
                transitions=constraints.Transitions.ERGODIC,
                n_trials=self.n_trials,
            )
            # Test that load doesn't raise error
            new_hmm.load_state_dict(torch.load(tmp.name))

            # Check that loaded model has correct structure
            self.assertEqual(new_hmm.n_states, self.n_states)
            self.assertEqual(new_hmm.emission_pdf.n_features, self.n_features)
            self.assertEqual(new_hmm.emission_pdf.n_trials, self.n_trials)

            # Check that parameters have correct shapes
            self.assertEqual(new_hmm.A.logits.shape, (self.n_states, self.n_states))
            self.assertEqual(new_hmm.pi.logits.shape, (self.n_states,))

            # Clean up
            os.unlink(tmp.name)


class TestGaussianHMM(unittest.TestCase):
    """Tests for GaussianHMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 2
        self.hmm = create_gaussian_hmm(
            n_states=self.n_states,
            n_features=self.n_features,
            transitions=constraints.Transitions.ERGODIC,
            alpha=1.0,
            seed=42,
        )

        # Generate test data
        torch.manual_seed(42)
        self.X = torch.randn(100, self.n_features)
        self.lengths = [50, 50]

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hmm.n_states, self.n_states)
        self.assertEqual(self.hmm.emission_pdf.n_features, self.n_features)

    def test_pdf_property(self):
        """Test emission PDF has required methods."""
        self.assertTrue(hasattr(self.hmm.emission_pdf, "log_prob"))
        self.assertTrue(hasattr(self.hmm.emission_pdf, "sample"))

    def test_dof_property(self):
        """Test degrees of freedom calculation."""
        # Just test that DOF is a positive integer
        self.assertIsInstance(self.hmm.dof, int)
        self.assertGreater(self.hmm.dof, 0)


class TestGaussianMixtureHMM(unittest.TestCase):
    """Tests for GaussianMixtureHMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 2
        self.n_features = 2
        self.n_components = 3
        self.hmm = create_gaussian_mixture_hmm(
            n_states=self.n_states,
            n_features=self.n_features,
            n_components=self.n_components,
            transitions=constraints.Transitions.ERGODIC,
            alpha=1.0,
            seed=42,
        )

        # Generate test data
        torch.manual_seed(42)
        self.X = torch.randn(100, self.n_features)
        self.lengths = [50, 50]

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hmm.n_states, self.n_states)
        self.assertEqual(self.hmm.emission_pdf.n_features, self.n_features)
        self.assertEqual(self.hmm.emission_pdf.n_mixtures, self.n_components)


class TestPoissonHMM(unittest.TestCase):
    """Tests for PoissonHMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 2
        self.hmm = create_poisson_hmm(
            n_states=self.n_states,
            n_features=self.n_features,
            transitions=constraints.Transitions.ERGODIC,
            alpha=1.0,
            seed=42,
        )

        # Generate test data
        torch.manual_seed(42)
        self.X = torch.poisson(torch.ones(100, self.n_features) * 2)
        self.lengths = [50, 50]

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hmm.n_states, self.n_states)
        self.assertEqual(self.hmm.emission_pdf.n_features, self.n_features)

    def test_pdf_property(self):
        """Test emission PDF has required methods."""
        self.assertTrue(hasattr(self.hmm.emission_pdf, "log_prob"))
        self.assertTrue(hasattr(self.hmm.emission_pdf, "sample"))


class TestHMMTransitions(unittest.TestCase):
    """Test different transition matrix types."""

    def test_ergodic_transitions(self):
        """Test ergodic transition matrix."""
        hmm = create_multinomial_hmm(
            n_states=3, n_features=4, transitions=constraints.Transitions.ERGODIC
        )

        A = hmm.A.probs
        # All transitions should be possible
        self.assertTrue(torch.all(A > 0))

    def test_left_to_right_transitions(self):
        """Test left-to-right transition matrix."""
        hmm = create_multinomial_hmm(
            n_states=3, n_features=4, transitions=constraints.Transitions.LEFT_TO_RIGHT
        )

        A = hmm.A.probs
        # Should be upper triangular
        self.assertTrue(torch.all(torch.tril(A, diagonal=-1) == 0))

    def test_semi_transitions(self):
        """Test semi-Markov transition matrix."""
        hmm = create_multinomial_hmm(
            n_states=3, n_features=4, transitions=constraints.Transitions.SEMI
        )

        A = hmm.A.probs
        # Diagonal should be zero
        self.assertTrue(torch.all(torch.diag(A) == 0))


if __name__ == "__main__":
    unittest.main()
