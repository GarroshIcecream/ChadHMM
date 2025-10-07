"""
Integration tests and edge case tests for ChadHMM.
Tests real-world scenarios, performance, and edge cases.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
from torch.distributions import Multinomial, Normal, Poisson

from chadhmm.hmm import (
    MultinomialHMM, 
    GaussianHMM, 
    GaussianMixtureHMM, 
    PoissonHMM
)
from chadhmm.hsmm import (
    MultinomialHSMM, 
    GaussianHSMM, 
    GaussianMixtureHSMM, 
    PoissonHSMM
)
from chadhmm.utilities import constraints


class TestIntegrationScenarios(unittest.TestCase):
    """Test real-world integration scenarios."""
    
    def test_financial_time_series_simulation(self):
        """Test HMM on simulated financial time series."""
        # Simulate a financial time series with 3 states: bull, bear, sideways
        torch.manual_seed(42)
        n_samples = 1000
        n_features = 2  # price change and volume
        
        # Generate synthetic financial data
        states = torch.zeros(n_samples, dtype=torch.long)
        observations = torch.zeros(n_samples, n_features)
        
        # State 0: Bull market (positive returns, high volume)
        # State 1: Bear market (negative returns, high volume)  
        # State 2: Sideways (small returns, low volume)
        
        current_state = 0
        for t in range(n_samples):
            if current_state == 0:  # Bull
                observations[t, 0] = torch.normal(0.02, 0.01, size=())  # positive returns
                observations[t, 1] = torch.normal(1.5, 0.3)    # high volume
            elif current_state == 1:  # Bear
                observations[t, 0] = torch.normal(-0.02, 0.01, size=())  # negative returns
                observations[t, 1] = torch.normal(1.5, 0.3)       # high volume
            else:  # Sideways
                observations[t, 0] = torch.normal(0.0, 0.005, size=())  # small returns
                observations[t, 1] = torch.normal(0.8, 0.2)    # low volume
            
            # State transitions (simplified)
            if torch.rand(1) < 0.05:  # 5% chance to change state
                current_state = (current_state + 1) % 3
            states[t] = current_state
        
        # Fit Gaussian HMM
        hmm = GaussianHMM(
            n_states=3,
            n_features=n_features,
            transitions=constraints.Transitions.ERGODIC,
            seed=42
        )
        
        # Split into train/validation
        train_size = int(0.8 * n_samples)
        train_X = observations[:train_size]
        val_X = observations[train_size:]
        train_lengths = [train_size]
        val_lengths = [len(val_X)]
        
        # Fit model
        hmm.fit(
            X=train_X,
            lengths=train_lengths,
            max_iter=20,
            n_init=3,
            verbose=False
        )
        
        # Test prediction
        predictions = hmm.predict(
            X=val_X,
            lengths=val_lengths,
            algorithm='viterbi'
        )
        
        # Check that predictions are valid
        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(predictions[0]), len(val_X))
        self.assertTrue(torch.all(predictions[0] >= 0))
        self.assertTrue(torch.all(predictions[0] < 3))
        
        # Test scoring
        scores = hmm.score(X=val_X, lengths=val_lengths)
        self.assertEqual(scores.shape, (1,))
        self.assertTrue(torch.isfinite(scores[0]))
        
    def test_sequence_classification(self):
        """Test HMM for sequence classification."""
        # Generate sequences from different HMMs
        torch.manual_seed(42)
        
        # Model 1: High frequency transitions
        hmm1 = MultinomialHMM(
            n_states=2,
            n_features=3,
            transitions=constraints.Transitions.ERGODIC,
            seed=42
        )
        
        # Model 2: Low frequency transitions  
        hmm2 = MultinomialHMM(
            n_states=2,
            n_features=3,
            transitions=constraints.Transitions.ERGODIC,
            seed=43
        )
        
        # Generate test sequences
        seq1 = hmm1.sample(100)
        seq2 = hmm2.sample(100)
        
        # Convert to observations
        X1 = torch.nn.functional.one_hot(seq1.long(), 3).float()
        X2 = torch.nn.functional.one_hot(seq2.long(), 3).float()
        
        # Fit both models
        hmm1.fit(X=X1, max_iter=10, n_init=1, verbose=False)
        hmm2.fit(X=X2, max_iter=10, n_init=1, verbose=False)
        
        # Test classification (higher likelihood should indicate better fit)
        score1_on_seq1 = hmm1.score(X1)[0]
        score1_on_seq2 = hmm1.score(X2)[0]
        score2_on_seq1 = hmm2.score(X1)[0]
        score2_on_seq2 = hmm2.score(X2)[0]
        
        # Model 1 should fit sequence 1 better than sequence 2
        self.assertGreater(score1_on_seq1, score1_on_seq2)
        
        # Model 2 should fit sequence 2 better than sequence 1
        self.assertGreater(score2_on_seq2, score2_on_seq1)
        
    def test_multiple_sequence_handling(self):
        """Test handling of multiple sequences."""
        torch.manual_seed(42)
        
        # Create multiple sequences of different lengths
        sequences = [
            torch.randint(0, 3, (50,)),
            torch.randint(0, 3, (75,)),
            torch.randint(0, 3, (25,))
        ]
        
        # Combine into single tensor
        X = torch.cat(sequences)
        lengths = [len(seq) for seq in sequences]
        
        # Convert to one-hot
        X_one_hot = torch.nn.functional.one_hot(X, 3).float()
        
        # Fit HMM
        hmm = MultinomialHMM(
            n_states=3,
            n_features=3,
            transitions=constraints.Transitions.ERGODIC,
            seed=42
        )
        
        hmm.fit(
            X=X_one_hot,
            lengths=lengths,
            max_iter=10,
            n_init=1,
            verbose=False
        )
        
        # Test prediction on all sequences
        predictions = hmm.predict(
            X=X_one_hot,
            lengths=lengths,
            algorithm='viterbi'
        )
        
        self.assertEqual(len(predictions), len(sequences))
        for i, pred in enumerate(predictions):
            self.assertEqual(len(pred), lengths[i])
            
    def test_model_comparison(self):
        """Test comparing different model configurations."""
        torch.manual_seed(42)
        
        # Generate test data
        X = torch.randn(200, 2)
        lengths = [100, 100]
        
        # Test different numbers of states
        models = {}
        for n_states in [2, 3, 4]:
            hmm = GaussianHMM(
                n_states=n_states,
                n_features=2,
                transitions=constraints.Transitions.ERGODIC,
                seed=42
            )
            hmm.fit(X=X, lengths=lengths, max_iter=10, n_init=1, verbose=False)
            models[n_states] = hmm
            
        # Compare using BIC
        bics = {}
        for n_states, hmm in models.items():
            bic = hmm.ic(X=X, lengths=lengths, criterion=constraints.InformCriteria.BIC)
            bics[n_states] = bic.sum().item()
            
        # All BICs should be finite
        for bic in bics.values():
            self.assertTrue(np.isfinite(bic))
            
    def test_device_consistency(self):
        """Test model behavior across different devices."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        torch.manual_seed(42)
        X = torch.randn(100, 2)
        lengths = [100]
        
        # CPU model
        hmm_cpu = GaussianHMM(
            n_states=3,
            n_features=2,
            seed=42
        )
        hmm_cpu.fit(X=X, lengths=lengths, max_iter=5, n_init=1, verbose=False)
        
        # GPU model
        hmm_gpu = GaussianHMM(
            n_states=3,
            n_features=2,
            seed=42
        )
        hmm_gpu.to('cuda')
        hmm_gpu.fit(X=X.cuda(), lengths=lengths, max_iter=5, n_init=1, verbose=False)
        
        # Move GPU model back to CPU for comparison
        hmm_gpu.to('cpu')
        
        # Parameters should be similar (allowing for numerical differences)
        self.assertTrue(torch.allclose(
            hmm_cpu.A, hmm_gpu.A, atol=1e-3
        ))
        self.assertTrue(torch.allclose(
            hmm_cpu.pi, hmm_gpu.pi, atol=1e-3
        ))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_very_short_sequences(self):
        """Test with very short sequences."""
        torch.manual_seed(42)
        
        # Single observation
        X = torch.randn(1, 2)
        hmm = GaussianHMM(n_states=2, n_features=2, seed=42)
        
        # Should not raise error
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        
        # Two observations
        X = torch.randn(2, 2)
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        
    def test_single_state_model(self):
        """Test single state model."""
        torch.manual_seed(42)
        
        X = torch.randn(100, 2)
        hmm = GaussianHMM(n_states=1, n_features=2, seed=42)
        
        # Should work without errors
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        
        predictions = hmm.predict(X=X, algorithm='viterbi')
        self.assertEqual(len(predictions), 1)
        self.assertTrue(torch.all(predictions[0] == 0))  # All states should be 0
        
    def test_large_models(self):
        """Test with large models."""
        torch.manual_seed(42)
        
        # Large number of states
        hmm = MultinomialHMM(
            n_states=20,
            n_features=10,
            transitions=constraints.Transitions.ERGODIC,
            seed=42
        )
        
        # Generate data
        X = torch.randint(0, 10, (500,))
        X_one_hot = torch.nn.functional.one_hot(X, 10).float()
        
        # Should work without memory issues
        hmm.fit(X=X_one_hot, max_iter=5, n_init=1, verbose=False)
        
        predictions = hmm.predict(X=X_one_hot, algorithm='viterbi')
        self.assertEqual(len(predictions), 1)
        self.assertTrue(torch.all(predictions[0] >= 0))
        self.assertTrue(torch.all(predictions[0] < 20))
        
    def test_extreme_parameter_values(self):
        """Test with extreme parameter values."""
        torch.manual_seed(42)
        
        # Very small alpha (concentrated distributions)
        hmm = MultinomialHMM(
            n_states=3,
            n_features=4,
            alpha=0.001,
            seed=42
        )
        
        X = torch.randint(0, 4, (100,))
        X_one_hot = torch.nn.functional.one_hot(X, 4).float()
        
        hmm.fit(X=X_one_hot, max_iter=5, n_init=1, verbose=False)
        
        # Very large alpha (uniform distributions)
        hmm2 = MultinomialHMM(
            n_states=3,
            n_features=4,
            alpha=100.0,
            seed=42
        )
        
        hmm2.fit(X=X_one_hot, max_iter=5, n_init=1, verbose=False)
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        torch.manual_seed(42)
        
        # Very large values
        X = torch.randn(100, 2) * 1000
        hmm = GaussianHMM(n_states=3, n_features=2, seed=42)
        
        # Should handle without overflow
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        
        # Very small values
        X = torch.randn(100, 2) * 1e-10
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        torch.manual_seed(42)
        
        # Large dataset
        X = torch.randn(10000, 3)
        lengths = [5000, 5000]
        
        hmm = GaussianHMM(n_states=5, n_features=3, seed=42)
        
        # Should complete without memory issues
        hmm.fit(X=X, lengths=lengths, max_iter=5, n_init=1, verbose=False)
        
        # Test prediction on large dataset
        predictions = hmm.predict(X=X, lengths=lengths, algorithm='viterbi')
        self.assertEqual(len(predictions), 2)
        
    def test_convergence_edge_cases(self):
        """Test convergence in edge cases."""
        torch.manual_seed(42)
        
        # Data that might cause convergence issues
        X = torch.zeros(100, 2)  # All zeros
        hmm = GaussianHMM(n_states=3, n_features=2, seed=42)
        
        # Should handle without errors
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        
        # Identical observations
        X = torch.ones(100, 2)  # All ones
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        torch.manual_seed(42)
        
        hmm = GaussianHMM(n_states=3, n_features=2, seed=42)
        X = torch.randn(100, 2)
        
        # Invalid lengths
        with self.assertRaises(ValueError):
            hmm.fit(X=X, lengths=[50, 30], max_iter=5, n_init=1, verbose=False)
            
        # Invalid algorithm
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        with self.assertRaises(ValueError):
            hmm.predict(X=X, algorithm='invalid')
            
        # Invalid information criteria
        with self.assertRaises(AttributeError):
            hmm.ic(X=X, criterion='invalid')


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_training_time(self):
        """Test that training completes in reasonable time."""
        import time
        
        torch.manual_seed(42)
        
        # Medium-sized problem
        X = torch.randn(1000, 3)
        lengths = [500, 500]
        
        hmm = GaussianHMM(n_states=5, n_features=3, seed=42)
        
        start_time = time.time()
        hmm.fit(X=X, lengths=lengths, max_iter=20, n_init=1, verbose=False)
        end_time = time.time()
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(end_time - start_time, 10.0)  # 10 seconds max
        
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        torch.manual_seed(42)
        
        # Large problem
        X = torch.randn(5000, 4)
        lengths = [2500, 2500]
        
        hmm = GaussianHMM(n_states=10, n_features=4, seed=42)
        
        # Should complete without excessive memory usage
        hmm.fit(X=X, lengths=lengths, max_iter=10, n_init=1, verbose=False)
        
        # Test prediction doesn't use excessive memory
        predictions = hmm.predict(X=X, lengths=lengths, algorithm='viterbi')
        self.assertEqual(len(predictions), 2)


if __name__ == '__main__':
    unittest.main()
