"""Tests for custom distribution classes."""

import unittest

import torch

from chadhmm.distributions import (
    DurationDistribution,
    InitialDistribution,
    TransitionMatrix,
)
from chadhmm.schemas.common import Transitions


class TestInitialDistribution(unittest.TestCase):
    """Tests for InitialDistribution class."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.logits = torch.log(torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64))

    def test_initialization(self):
        """Test initialization with logits."""
        pi = InitialDistribution(logits=self.logits)
        self.assertEqual(pi.n_states, self.n_states)
        self.assertTrue(
            torch.allclose(pi.logits.sum(), torch.tensor(1.0, dtype=torch.float64))
        )

    def test_initialization_with_probs(self):
        """Test initialization with probabilities."""
        probs = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
        pi = InitialDistribution(probs=probs)
        self.assertEqual(pi.n_states, self.n_states)
        self.assertTrue(torch.allclose(pi.logits, probs))

    def test_invalid_dimensions(self):
        """Test that 2D logits raise an error."""
        with self.assertRaises(ValueError):
            InitialDistribution(logits=torch.randn(3, 3))

    def test_sampling(self):
        """Test sampling from the distribution."""
        pi = InitialDistribution(logits=self.logits)
        sample = pi.sample()
        self.assertTrue(0 <= sample < self.n_states)

    def test_repr(self):
        """Test string representation."""
        pi = InitialDistribution(logits=self.logits)
        self.assertIn("InitialDistribution", repr(pi))
        self.assertIn(f"n_states={self.n_states}", repr(pi))


class TestTransitionMatrix(unittest.TestCase):
    """Tests for TransitionMatrix class."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3

    def test_ergodic_initialization(self):
        """Test initialization of ergodic transition matrix."""
        logits = torch.log(
            torch.tensor(
                [[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]], dtype=torch.float64
            )
        )
        A = TransitionMatrix(logits=logits, transition_type=Transitions.ERGODIC)

        self.assertEqual(A.n_states, self.n_states)
        self.assertEqual(A.transition_type, Transitions.ERGODIC)
        # Check that each row sums to 1
        self.assertTrue(
            torch.allclose(
                A.probs.sum(dim=1), torch.ones(self.n_states, dtype=torch.float64)
            )
        )

    def test_semi_markov_initialization(self):
        """Test initialization of semi-Markov transition matrix."""
        logits = torch.log(
            torch.tensor(
                [[0.0, 0.6, 0.4], [0.5, 0.0, 0.5], [0.3, 0.7, 0.0]], dtype=torch.float64
            )
        )
        A = TransitionMatrix(logits=logits, transition_type=Transitions.SEMI)

        self.assertEqual(A.transition_type, Transitions.SEMI)
        # Check that diagonal is zero
        self.assertTrue(
            torch.allclose(
                torch.diag(A.probs), torch.zeros(self.n_states, dtype=torch.float64)
            )
        )

    def test_left_to_right_initialization(self):
        """Test initialization of left-to-right transition matrix."""
        logits = torch.log(
            torch.tensor(
                [[0.5, 0.3, 0.2], [0.0, 0.6, 0.4], [0.0, 0.0, 1.0]], dtype=torch.float64
            )
        )
        A = TransitionMatrix(logits=logits, transition_type=Transitions.LEFT_TO_RIGHT)

        self.assertEqual(A.transition_type, Transitions.LEFT_TO_RIGHT)

    def test_invalid_dimensions(self):
        """Test that non-square matrices raise an error."""
        with self.assertRaises(ValueError):
            TransitionMatrix(
                transition_type=Transitions.ERGODIC, logits=torch.randn(3, 4)
            )

    def test_sample_next_state(self):
        """Test sampling next state."""
        logits = torch.log(
            torch.tensor(
                [[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]], dtype=torch.float64
            )
        )
        A = TransitionMatrix(logits=logits, transition_type=Transitions.ERGODIC)

        current_state = torch.tensor(0)
        next_state = A.sample_next_state(current_state)
        self.assertTrue(0 <= next_state < self.n_states)


class TestDurationDistribution(unittest.TestCase):
    """Tests for DurationDistribution class."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.max_duration = 5
        self.logits = torch.log(
            torch.tensor(
                [
                    [0.4, 0.3, 0.2, 0.08, 0.02],
                    [0.1, 0.2, 0.3, 0.25, 0.15],
                    [0.2, 0.2, 0.2, 0.2, 0.2],
                ],
                dtype=torch.float64,
            )
        )

    def test_initialization(self):
        """Test initialization with logits."""
        D = DurationDistribution(logits=self.logits)

        self.assertEqual(D.n_states, self.n_states)
        self.assertEqual(D.max_duration, self.max_duration)
        # Check that each row sums to 1
        self.assertTrue(
            torch.allclose(
                D.probs.sum(dim=1), torch.ones(self.n_states, dtype=torch.float64)
            )
        )

    def test_initialization_with_probs(self):
        """Test initialization with probabilities."""
        probs = self.logits.exp()
        D = DurationDistribution(probs=probs)

        self.assertEqual(D.n_states, self.n_states)
        self.assertEqual(D.max_duration, self.max_duration)

    def test_invalid_dimensions(self):
        """Test that 1D logits raise an error."""
        with self.assertRaises(ValueError):
            DurationDistribution(logits=torch.randn(5))

    def test_sample_duration(self):
        """Test sampling duration for a state."""
        D = DurationDistribution(logits=self.logits)

        state = torch.tensor(0)
        duration = D.sample_duration(state)
        self.assertTrue(0 <= duration < self.max_duration)

    def test_mean_duration(self):
        """Test computing mean duration."""
        D = DurationDistribution(logits=self.logits)

        # Mean for all states
        mean_all = D.mean_duration()
        self.assertEqual(mean_all.shape, (self.n_states,))

        # Mean for specific state
        mean_state_0 = D.mean_duration(0)
        self.assertIsInstance(mean_state_0.item(), float)

        # Check that state 0 has lower mean than state 1
        # (based on our test distribution)
        self.assertLess(mean_all[0], mean_all[1])

    def test_repr(self):
        """Test string representation."""
        D = DurationDistribution(logits=self.logits)

        self.assertIn("DurationDistribution", repr(D))
        self.assertIn(f"n_states={self.n_states}", repr(D))
        self.assertIn(f"max_duration={self.max_duration}", repr(D))


class TestDistributionIntegration(unittest.TestCase):
    """Test integration with HMM/HSMM models."""

    def test_hmm_uses_custom_distributions(self):
        """Test that HMM uses custom distribution classes."""
        from chadhmm.hmm import GaussianHMM
        from chadhmm.utils.constraints import CovarianceType

        hmm = GaussianHMM(
            n_states=3,
            n_features=2,
            transitions=Transitions.ERGODIC,
            covariance_type=CovarianceType.FULL,
            seed=42,
        )

        self.assertIsInstance(hmm._params.pi, InitialDistribution)
        self.assertIsInstance(hmm._params.A, TransitionMatrix)
        self.assertEqual(hmm._params.A.transition_type, Transitions.ERGODIC)

    def test_hsmm_uses_custom_distributions(self):
        """Test that HSMM uses custom distribution classes."""
        from chadhmm.hsmm import GaussianHSMM
        from chadhmm.utils.constraints import CovarianceType

        hsmm = GaussianHSMM(
            n_states=3,
            n_features=2,
            max_duration=5,
            covariance_type=CovarianceType.FULL,
            seed=42,
        )

        self.assertIsInstance(hsmm._params.pi, InitialDistribution)
        self.assertIsInstance(hsmm._params.A, TransitionMatrix)
        self.assertIsInstance(hsmm._params.D, DurationDistribution)
        self.assertEqual(hsmm._params.A.transition_type, Transitions.SEMI)


if __name__ == "__main__":
    unittest.main()
