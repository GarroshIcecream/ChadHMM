import unittest

import torch
from torch.distributions import (
    Distribution,
    Independent,
    Multinomial,
    MultivariateNormal,
)

from chadhmm.hmm import GaussianHMM, GaussianMixtureHMM, MultinomialHMM, PoissonHMM
from chadhmm.schemas import DecodingAlgorithm, Transitions
from chadhmm.utils import constraints


class TestBasicMultinomialHMM(unittest.TestCase):
    """Basic tests for MultinomialHMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.hmm = MultinomialHMM(2, 3, transitions=Transitions.ERGODIC)

    def test_pdf_subclass(self):
        """Test that PDF is a subclass of Distribution."""
        self.assertTrue(issubclass(type(self.hmm.pdf), Distribution))

    def test_dof_is_int(self):
        """Test that degrees of freedom is an integer."""
        self.assertIsInstance(self.hmm.dof, int)

    def test_emission_pdf(self):
        """Test emission PDF sampling."""
        emission_pdf = self.hmm.sample_emission_pdf()
        self.assertIsInstance(emission_pdf, Distribution)

    def test_emission_pdf_with_data(self):
        """Test emission PDF sampling with data."""
        X = torch.tensor([0, 1, 2, 0, 1])
        emission_pdf = self.hmm.sample_emission_pdf(X)
        self.assertIsInstance(emission_pdf, Distribution)


class TestMultinomialHMM(unittest.TestCase):
    """Tests for MultinomialHMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 4
        self.n_trials = 2
        self.hmm = MultinomialHMM(
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
        self.assertEqual(self.hmm.n_features, self.n_features)
        self.assertEqual(self.hmm.n_trials, self.n_trials)
        self.assertEqual(self.hmm.seed, 42)

    def test_pdf_property(self):
        """Test PDF property returns correct distribution type."""
        self.assertIsInstance(self.hmm.pdf, Multinomial)

    def test_dof_property(self):
        """Test degrees of freedom calculation."""
        expected_dof = (
            self.n_states**2 + self.n_states * self.n_features - self.n_states - 1
        )
        self.assertEqual(self.hmm.dof, expected_dof)

    def test_device_property(self):
        """Test device property."""
        self.assertEqual(self.hmm.device, torch.device("cpu"))

    def test_transition_matrix_properties(self):
        """Test transition matrix A and initial probabilities pi."""
        # Test A property
        A = self.hmm.A
        self.assertEqual(A.shape, (self.n_states, self.n_states))
        self.assertTrue(
            torch.allclose(A.logsumexp(1), torch.zeros(self.n_states, dtype=A.dtype))
        )

        # Test pi property
        pi = self.hmm.pi
        self.assertEqual(pi.shape, (self.n_states,))
        self.assertTrue(torch.allclose(pi.logsumexp(0), torch.zeros(1, dtype=pi.dtype)))

    def test_sample_emission_pdf(self):
        """Test emission PDF sampling."""
        # Test without data
        pdf = self.hmm.sample_emission_pdf()
        self.assertIsInstance(pdf, Multinomial)
        self.assertEqual(pdf.total_count, self.n_trials)
        self.assertEqual(pdf.logits.shape, (self.n_states, self.n_features))

        # Test with data
        pdf_with_data = self.hmm.sample_emission_pdf(self.X)
        self.assertIsInstance(pdf_with_data, Multinomial)

    def test_fit_method(self):
        """Test model fitting."""
        # Test basic fitting
        self.hmm.fit(
            X=self.X_one_hot, lengths=self.lengths, max_iter=5, n_init=1, verbose=False
        )

        # Check that parameters are updated
        self.assertIsNotNone(self.hmm._params)

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
        sample = self.hmm.sample(size=100)
        self.assertEqual(sample.shape, (100,))
        self.assertTrue(torch.all(sample >= 0))
        self.assertTrue(torch.all(sample < self.n_states))

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
            self.hmm.save_model(tmp.name)
            self.assertTrue(os.path.exists(tmp.name))

            # Load model
            new_hmm = MultinomialHMM(
                n_states=self.n_states,
                n_features=self.n_features,
                transitions=constraints.Transitions.ERGODIC,
                n_trials=self.n_trials,
            )
            # Test that load doesn't raise error
            new_hmm.load_model(tmp.name)

            # Check that loaded model has correct structure
            self.assertEqual(new_hmm.n_states, self.n_states)
            self.assertEqual(new_hmm.n_features, self.n_features)
            self.assertEqual(new_hmm.n_trials, self.n_trials)

            # Check that parameters have correct shapes
            self.assertEqual(new_hmm.A.shape, (self.n_states, self.n_states))
            self.assertEqual(new_hmm.pi.shape, (self.n_states,))

            # Clean up
            os.unlink(tmp.name)


class TestGaussianHMM(unittest.TestCase):
    """Tests for GaussianHMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 2
        self.hmm = GaussianHMM(
            n_states=self.n_states,
            n_features=self.n_features,
            transitions=constraints.Transitions.ERGODIC,
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
        self.assertEqual(self.hmm.n_states, self.n_states)
        self.assertEqual(self.hmm.n_features, self.n_features)

    def test_pdf_property(self):
        """Test PDF property returns correct distribution type."""
        self.assertIsInstance(self.hmm.pdf, MultivariateNormal)

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
        self.hmm = GaussianMixtureHMM(
            n_states=self.n_states,
            n_features=self.n_features,
            n_components=self.n_components,
            transitions=constraints.Transitions.ERGODIC,
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
        self.assertEqual(self.hmm.n_states, self.n_states)
        self.assertEqual(self.hmm.n_features, self.n_features)
        self.assertEqual(self.hmm.n_components, self.n_components)


class TestPoissonHMM(unittest.TestCase):
    """Tests for PoissonHMM."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 2
        self.hmm = PoissonHMM(
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
        self.assertEqual(self.hmm.n_features, self.n_features)

    def test_pdf_property(self):
        """Test PDF property returns correct distribution type."""
        self.assertIsInstance(self.hmm.pdf, Independent)


class TestHMMTransitions(unittest.TestCase):
    """Test different transition matrix types."""

    def test_ergodic_transitions(self):
        """Test ergodic transition matrix."""
        hmm = MultinomialHMM(
            n_states=3, n_features=4, transitions=constraints.Transitions.ERGODIC
        )

        A = hmm.A.exp()
        # All transitions should be possible
        self.assertTrue(torch.all(A > 0))

    def test_left_to_right_transitions(self):
        """Test left-to-right transition matrix."""
        hmm = MultinomialHMM(
            n_states=3, n_features=4, transitions=constraints.Transitions.LEFT_TO_RIGHT
        )

        A = hmm.A.exp()
        # Should be upper triangular
        self.assertTrue(torch.all(torch.tril(A, diagonal=-1) == 0))

    def test_semi_transitions(self):
        """Test semi-Markov transition matrix."""
        hmm = MultinomialHMM(
            n_states=3, n_features=4, transitions=constraints.Transitions.SEMI
        )

        A = hmm.A.exp()
        # Diagonal should be zero
        self.assertTrue(torch.all(torch.diag(A) == 0))


if __name__ == "__main__":
    unittest.main()
