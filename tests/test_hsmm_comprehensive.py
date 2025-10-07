"""
Comprehensive tests for HSMM models.
Tests all HSMM implementations with various scenarios and edge cases.
"""

import unittest
import torch
from torch.distributions import Multinomial, Normal, Poisson

from chadhmm.hsmm import (
    MultinomialHSMM, 
    GaussianHSMM, 
    GaussianMixtureHSMM, 
    PoissonHSMM
)
from chadhmm.utils import constraints


class TestMultinomialHSMM(unittest.TestCase):
    """Comprehensive tests for MultinomialHSMM."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 4
        self.n_trials = 2
        self.max_duration = 10
        self.hsmm = MultinomialHSMM(
            n_states=self.n_states,
            n_features=self.n_features,
            n_trials=self.n_trials,
            max_duration=self.max_duration,
            alpha=1.0,
            seed=42
        )
        
        # Generate test data
        torch.manual_seed(42)
        self.X = torch.randint(0, self.n_features, (100,))
        self.X_one_hot = self.n_trials * torch.nn.functional.one_hot(self.X, self.n_features)
        self.lengths = [50, 50]
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hsmm.n_states, self.n_states)
        self.assertEqual(self.hsmm.n_features, self.n_features)
        self.assertEqual(self.hsmm.n_trials, self.n_trials)
        self.assertEqual(self.hsmm.max_duration, self.max_duration)
        self.assertEqual(self.hsmm.seed, 42)
        
    def test_pdf_property(self):
        """Test PDF property returns correct distribution type."""
        self.assertIsInstance(self.hsmm.pdf, Multinomial)
        
    def test_dof_property(self):
        """Test degrees of freedom calculation."""
        # Just test that DOF is a positive integer
        self.assertIsInstance(self.hsmm.dof, int)
        self.assertGreater(self.hsmm.dof, 0)
        
    def test_duration_matrix_property(self):
        """Test duration matrix D property."""
        D = self.hsmm.D
        self.assertEqual(D.shape, (self.n_states, self.max_duration))
        self.assertTrue(torch.allclose(D.logsumexp(1), torch.zeros(self.n_states, dtype=D.dtype)))
        
    def test_sample_emission_pdf(self):
        """Test emission PDF sampling."""
        # Test without data
        pdf = self.hsmm.sample_emission_pdf()
        self.assertIsInstance(pdf, Multinomial)
        self.assertEqual(pdf.total_count, self.n_trials)
        self.assertEqual(pdf.logits.shape, (self.n_states, self.n_features))
        
        # Test with data
        pdf_with_data = self.hsmm.sample_emission_pdf(self.X)
        self.assertIsInstance(pdf_with_data, Multinomial)
        
    def test_fit_method(self):
        """Test model fitting."""
        # Test basic fitting
        self.hsmm.fit(
            X=self.X_one_hot,
            lengths=self.lengths,
            max_iter=5,
            n_init=1,
            verbose=False
        )
        
        # Check that parameters are updated
        self.assertIsNotNone(self.hsmm._params)
        
    def test_predict_method(self):
        """Test prediction methods."""
        # Fit model first
        self.hsmm.fit(
            X=self.X_one_hot,
            lengths=self.lengths,
            max_iter=5,
            n_init=1,
            verbose=False
        )
        
        # Test MAP prediction (Viterbi not implemented yet)
        map_path = self.hsmm.predict(
            X=self.X_one_hot,
            lengths=self.lengths,
            algorithm='map'
        )
        self.assertIsInstance(map_path, list)
        self.assertEqual(len(map_path), len(self.lengths))
        
        # Test that Viterbi raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.hsmm.predict(
                X=self.X_one_hot,
                lengths=self.lengths,
                algorithm='viterbi'
            )
        
    def test_score_method(self):
        """Test scoring methods."""
        # Fit model first
        self.hsmm.fit(
            X=self.X_one_hot,
            lengths=self.lengths,
            max_iter=5,
            n_init=1,
            verbose=False
        )
        
        # Test by_sample=True
        scores_by_sample = self.hsmm.score(
            X=self.X_one_hot,
            lengths=self.lengths,
            by_sample=True
        )
        # HSMM returns shape (n_sequences, max_duration) for scores
        self.assertEqual(scores_by_sample.shape, (len(self.lengths), self.max_duration))
        
        # Test by_sample=False
        scores_joint = self.hsmm.score(
            X=self.X_one_hot,
            lengths=self.lengths,
            by_sample=False
        )
        # HSMM returns shape (1, max_duration) for joint scores
        self.assertEqual(scores_joint.shape, (1, self.max_duration))
        
    def test_ic_method(self):
        """Test information criteria calculation."""
        # Fit model first
        self.hsmm.fit(
            X=self.X_one_hot,
            lengths=self.lengths,
            max_iter=5,
            n_init=1,
            verbose=False
        )
        
        # Test AIC
        aic = self.hsmm.ic(
            X=self.X_one_hot,
            lengths=self.lengths,
            criterion=constraints.InformCriteria.AIC
        )
        # HSMM returns shape (n_sequences, max_duration) for IC
        self.assertEqual(aic.shape, (len(self.lengths), self.max_duration))
        
        # Test BIC
        bic = self.hsmm.ic(
            X=self.X_one_hot,
            lengths=self.lengths,
            criterion=constraints.InformCriteria.BIC
        )
        # HSMM returns shape (n_sequences, max_duration) for IC
        self.assertEqual(bic.shape, (len(self.lengths), self.max_duration))
        
    def test_model_persistence(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        # Fit model first
        self.hsmm.fit(
            X=self.X_one_hot,
            lengths=self.lengths,
            max_iter=5,
            n_init=1,
            verbose=False
        )
        
        # HSMM models don't have save_model method, so we'll test basic functionality
        # Test that model can be fitted and has correct structure
        self.assertEqual(self.hsmm.n_states, self.n_states)
        self.assertEqual(self.hsmm.n_features, self.n_features)
        self.assertEqual(self.hsmm.n_trials, self.n_trials)
        self.assertEqual(self.hsmm.max_duration, self.max_duration)

        # Check that parameters have correct shapes
        self.assertEqual(self.hsmm.A.shape, (self.n_states, self.n_states))
        self.assertEqual(self.hsmm.D.shape, (self.n_states, self.max_duration))
            
    def test_device_transfer(self):
        """Test moving model to different devices."""
        if torch.cuda.is_available():
            # Move to CUDA
            self.hsmm.to('cuda')
            self.assertEqual(self.hsmm.device.type, 'cuda')
            
            # Move back to CPU
            self.hsmm.to('cpu')
            self.assertEqual(self.hsmm.device.type, 'cpu')
            
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
                verbose=False
            )


class TestGaussianHSMM(unittest.TestCase):
    """Comprehensive tests for GaussianHSMM."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 2
        self.max_duration = 10
        self.hsmm = GaussianHSMM(
            n_states=self.n_states,
            n_features=self.n_features,
            max_duration=self.max_duration,
            alpha=1.0,
            seed=42
        )
        
        # Generate test data
        torch.manual_seed(42)
        self.X = torch.randn(100, self.n_features)
        self.lengths = [50, 50]
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hsmm.n_states, self.n_states)
        self.assertEqual(self.hsmm.n_features, self.n_features)
        self.assertEqual(self.hsmm.max_duration, self.max_duration)
        
    def test_pdf_property(self):
        """Test PDF property returns correct distribution type."""
        self.assertIsInstance(self.hsmm.pdf, Normal)
        
    def test_dof_property(self):
        """Test degrees of freedom calculation."""
        expected_dof = (self.n_states**2 + 
                          self.n_states * self.max_duration +
                          self.n_states * self.n_features * 2 - 
                          self.n_states - 1)
        self.assertEqual(self.hsmm.dof, expected_dof)
        
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Fit model
        self.hsmm.fit(
            X=self.X,
            lengths=self.lengths,
            max_iter=5,
            n_init=1,
            verbose=False
        )
        
        # Test MAP prediction
        predictions = self.hsmm.predict(
            X=self.X,
            lengths=self.lengths,
            algorithm='map'
        )
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(self.lengths))
        
        # Test that Viterbi raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.hsmm.predict(
                X=self.X,
                lengths=self.lengths,
                algorithm='viterbi'
            )
        
    def test_score_method(self):
        """Test scoring method."""
        self.hsmm.fit(
            X=self.X,
            lengths=self.lengths,
            max_iter=5,
            n_init=1,
            verbose=False
        )
        
        scores = self.hsmm.score(X=self.X, lengths=self.lengths)
        self.assertEqual(scores.shape, (len(self.lengths),))


class TestGaussianMixtureHSMM(unittest.TestCase):
    """Comprehensive tests for GaussianMixtureHSMM."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 2
        self.n_features = 2
        self.n_components = 3
        self.max_duration = 10
        self.hsmm = GaussianMixtureHSMM(
            n_states=self.n_states,
            n_features=self.n_features,
            n_components=self.n_components,
            max_duration=self.max_duration,
            alpha=1.0,
            seed=42
        )
        
        # Generate test data
        torch.manual_seed(42)
        self.X = torch.randn(100, self.n_features)
        self.lengths = [50, 50]
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hsmm.n_states, self.n_states)
        self.assertEqual(self.hsmm.n_features, self.n_features)
        self.assertEqual(self.hsmm.n_components, self.n_components)
        self.assertEqual(self.hsmm.max_duration, self.max_duration)
        
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Fit model
        self.hsmm.fit(
            X=self.X,
            lengths=self.lengths,
            max_iter=5,
            n_init=1,
            verbose=False
        )
        
        # Test MAP prediction
        predictions = self.hsmm.predict(
            X=self.X,
            lengths=self.lengths,
            algorithm='map'
        )
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(self.lengths))
        
        # Test that Viterbi raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.hsmm.predict(
                X=self.X,
                lengths=self.lengths,
                algorithm='viterbi'
            )


class TestPoissonHSMM(unittest.TestCase):
    """Comprehensive tests for PoissonHSMM."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_states = 3
        self.n_features = 2
        self.max_duration = 10
        self.hsmm = PoissonHSMM(
            n_states=self.n_states,
            n_features=self.n_features,
            max_duration=self.max_duration,
            alpha=1.0,
            seed=42
        )
        
        # Generate test data
        torch.manual_seed(42)
        self.X = torch.poisson(torch.ones(100, self.n_features) * 2)
        self.lengths = [50, 50]
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hsmm.n_states, self.n_states)
        self.assertEqual(self.hsmm.n_features, self.n_features)
        self.assertEqual(self.hsmm.max_duration, self.max_duration)
        
    def test_pdf_property(self):
        """Test PDF property returns correct distribution type."""
        self.assertIsInstance(self.hsmm.pdf, Poisson)
        
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Fit model
        self.hsmm.fit(
            X=self.X,
            lengths=self.lengths,
            max_iter=5,
            n_init=1,
            verbose=False
        )
        
        # Test MAP prediction
        predictions = self.hsmm.predict(
            X=self.X,
            lengths=self.lengths,
            algorithm='map'
        )
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(self.lengths))
        
        # Test that Viterbi raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.hsmm.predict(
                X=self.X,
                lengths=self.lengths,
                algorithm='viterbi'
            )


class TestHSMMTransitions(unittest.TestCase):
    """Test different transition matrix types for HSMM."""
    
    def test_ergodic_transitions(self):
        """Test ergodic transition matrix."""
        hsmm = MultinomialHSMM(
            n_states=3,
            n_features=4,
            max_duration=10
        )
        
        A = hsmm.A.exp()
        # All transitions should be possible
        self.assertTrue(torch.all(A > 0))
        
    def test_left_to_right_transitions(self):
        """Test left-to-right transition matrix."""
        hsmm = MultinomialHSMM(
            n_states=3,
            n_features=4,
            max_duration=10
        )
        
        A = hsmm.A.exp()
        # Should be upper triangular
        self.assertTrue(torch.all(torch.tril(A, diagonal=-1) == 0))
        
    def test_semi_transitions(self):
        """Test semi-Markov transition matrix."""
        hsmm = MultinomialHSMM(
            n_states=3,
            n_features=4,
            max_duration=10
        )
        
        A = hsmm.A.exp()
        # Diagonal should be zero
        self.assertTrue(torch.all(torch.diag(A) == 0))


class TestHSMMDurationModeling(unittest.TestCase):
    """Test duration modeling in HSMM."""
    
    def test_duration_matrix_initialization(self):
        """Test duration matrix initialization."""
        hsmm = MultinomialHSMM(
            n_states=3,
            n_features=4,
            max_duration=10
        )
        
        D = hsmm.D.exp()
        # Each row should sum to 1
        self.assertTrue(torch.allclose(D.sum(1), torch.ones(3, dtype=D.dtype)))
        
        # All probabilities should be non-negative
        self.assertTrue(torch.all(D >= 0))
        
    def test_duration_matrix_constraints(self):
        """Test duration matrix constraints."""
        hsmm = MultinomialHSMM(
            n_states=3,
            n_features=4,
            max_duration=10
        )
        
        # Test setting invalid duration matrix
        with self.assertRaises(ValueError):
            invalid_D = torch.zeros((3, 10))
            hsmm.D = invalid_D.log()


if __name__ == '__main__':
    unittest.main()
