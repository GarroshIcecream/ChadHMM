"""
Pytest-style tests for ChadHMM using modern testing practices.
Demonstrates pytest fixtures, parametrization, and advanced testing patterns.
"""

import pytest
import torch
import numpy as np
from torch.distributions import Multinomial, Normal, Poisson

from chadhmm.hmm import MultinomialHMM, GaussianHMM, GaussianMixtureHMM, PoissonHMM
from chadhmm.hsmm import MultinomialHSMM, GaussianHSMM
from chadhmm.utilities import constraints


class TestHMMBasic:
    """Basic HMM tests using pytest style."""
    
    def test_multinomial_hmm_initialization(self, sample_models):
        """Test MultinomialHMM initialization."""
        hmm = sample_models['multinomial_hmm']
        assert hmm.n_states == 3
        assert hmm.n_features == 4
        assert hmm.n_trials == 2
        assert hmm.seed == 42
    
    def test_gaussian_hmm_initialization(self, sample_models):
        """Test GaussianHMM initialization."""
        hmm = sample_models['gaussian_hmm']
        assert hmm.n_states == 3
        assert hmm.n_features == 2
        assert hmm.seed == 42
    
    @pytest.mark.parametrize("n_states,n_features", [
        (2, 3), (3, 4), (5, 2), (1, 1)
    ])
    def test_multinomial_hmm_creation(self, n_states, n_features):
        """Test creating MultinomialHMM with different parameters."""
        hmm = MultinomialHMM(n_states=n_states, n_features=n_features)
        assert hmm.n_states == n_states
        assert hmm.n_features == n_features
    
    def test_model_device_property(self, sample_models):
        """Test model device property."""
        hmm = sample_models['gaussian_hmm']
        assert hmm.device == torch.device('cpu')
    
    def test_model_seed_property(self, sample_models):
        """Test model seed property."""
        hmm = sample_models['multinomial_hmm']
        assert hmm.seed == 42


class TestHMMFitting:
    """Test HMM fitting functionality."""
    
    def test_multinomial_hmm_fitting(self, sample_models, sample_data):
        """Test MultinomialHMM fitting."""
        hmm = sample_models['multinomial_hmm']
        X = sample_data['multinomial_data']
        X_one_hot = torch.nn.functional.one_hot(X, 4).float()
        
        # Should not raise error
        hmm.fit(X=X_one_hot, max_iter=5, n_init=1, verbose=False)
        
        # Check that parameters are updated
        assert hmm._params is not None
    
    def test_gaussian_hmm_fitting(self, sample_models, sample_data):
        """Test GaussianHMM fitting."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        assert hmm._params is not None
    
    def test_fitting_with_lengths(self, sample_models, sample_data):
        """Test fitting with multiple sequences."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        lengths = sample_data['lengths']
        
        hmm.fit(X=X, lengths=lengths, max_iter=5, n_init=1, verbose=False)
        assert hmm._params is not None
    
    @pytest.mark.parametrize("max_iter", [1, 5, 10])
    def test_fitting_iterations(self, sample_models, sample_data, max_iter):
        """Test fitting with different iteration counts."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        
        hmm.fit(X=X, max_iter=max_iter, n_init=1, verbose=False)
        assert hmm._params is not None


class TestHMMPrediction:
    """Test HMM prediction functionality."""
    
    def test_viterbi_prediction(self, sample_models, sample_data):
        """Test Viterbi prediction."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        
        # Fit first
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        
        # Predict
        predictions = hmm.predict(X=X, algorithm='viterbi')
        assert len(predictions) == 1
        assert len(predictions[0]) == len(X)
        assert torch.all(predictions[0] >= 0)
        assert torch.all(predictions[0] < hmm.n_states)
    
    def test_map_prediction(self, sample_models, sample_data):
        """Test MAP prediction."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        predictions = hmm.predict(X=X, algorithm='map')
        
        assert len(predictions) == 1
        assert len(predictions[0]) == len(X)
    
    def test_prediction_with_lengths(self, sample_models, sample_data):
        """Test prediction with multiple sequences."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        lengths = sample_data['lengths']
        
        hmm.fit(X=X, lengths=lengths, max_iter=5, n_init=1, verbose=False)
        predictions = hmm.predict(X=X, lengths=lengths, algorithm='viterbi')
        
        assert len(predictions) == len(lengths)
        for pred, length in zip(predictions, lengths):
            assert len(pred) == length
    
    def test_invalid_algorithm(self, sample_models, sample_data):
        """Test invalid prediction algorithm."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        
        with pytest.raises(ValueError):
            hmm.predict(X=X, algorithm='invalid')


class TestHMMScoring:
    """Test HMM scoring functionality."""
    
    def test_score_by_sample(self, sample_models, sample_data):
        """Test scoring by sample."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        lengths = sample_data['lengths']
        
        hmm.fit(X=X, lengths=lengths, max_iter=5, n_init=1, verbose=False)
        scores = hmm.score(X=X, lengths=lengths, by_sample=True)
        
        assert scores.shape == (len(lengths),)
        assert torch.isfinite(scores).all()
    
    def test_score_joint(self, sample_models, sample_data):
        """Test joint scoring."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        scores = hmm.score(X=X, by_sample=False)
        
        assert scores.shape == (1,)
        assert torch.isfinite(scores).all()
    
    def test_information_criteria(self, sample_models, sample_data, criterion):
        """Test information criteria calculation."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        ic = hmm.ic(X=X, criterion=criterion)
        
        assert ic.shape == (1,)
        assert torch.isfinite(ic).all()


class TestHMMTransitions:
    """Test different transition matrix types."""
    
    def test_transition_types(self, transition_type):
        """Test different transition matrix types."""
        hmm = MultinomialHMM(
            n_states=3,
            n_features=4,
            transitions=transition_type,
            seed=42
        )
        
        A = hmm.A.exp()
        assert torch.allclose(A.sum(1), torch.ones(3, dtype=A.dtype))
        
        if transition_type == constraints.Transitions.LEFT_TO_RIGHT:
            # Should be upper triangular
            assert torch.all(torch.tril(A, diagonal=-1) == 0)
        elif transition_type == constraints.Transitions.SEMI:
            # Diagonal should be zero
            assert torch.all(torch.diag(A) == 0)


class TestHMMSampling:
    """Test HMM sampling functionality."""
    
    def test_sampling(self, sample_models):
        """Test sampling from HMM."""
        hmm = sample_models['multinomial_hmm']
        sample = hmm.sample(size=100)
        
        assert sample.shape == (100,)
        assert torch.all(sample >= 0)
        assert torch.all(sample < hmm.n_states)
    
    def test_sampling_different_sizes(self, sample_models):
        """Test sampling with different sizes."""
        hmm = sample_models['gaussian_hmm']
        
        for size in [1, 10, 100, 1000]:
            sample = hmm.sample(size=size)
            assert sample.shape == (size,)
            assert torch.all(sample >= 0)
            assert torch.all(sample < hmm.n_states)


class TestHMMPersistence:
    """Test HMM model persistence."""
    
    def test_save_load_model(self, sample_models, sample_data, temp_file):
        """Test model saving and loading."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        
        # Fit model
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        
        # Save model
        hmm.save_model(temp_file)
        
        # Create new model and load
        new_hmm = GaussianHMM(n_states=3, n_features=2, seed=42)
        new_hmm.load_model(temp_file)
        
        # Check parameters are the same
        assert torch.allclose(hmm.A, new_hmm.A)
        assert torch.allclose(hmm.pi, new_hmm.pi)
    
    def test_save_load_with_predictions(self, sample_models, sample_data, temp_file):
        """Test that loaded model produces same predictions."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        
        # Fit and predict
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        original_predictions = hmm.predict(X=X, algorithm='viterbi')
        
        # Save and load
        hmm.save_model(temp_file)
        new_hmm = GaussianHMM(n_states=3, n_features=2, seed=42)
        new_hmm.load_model(temp_file)
        
        # Predict with loaded model
        loaded_predictions = new_hmm.predict(X=X, algorithm='viterbi')
        
        # Should be identical
        assert torch.equal(original_predictions[0], loaded_predictions[0])


class TestHMMEdgeCases:
    """Test HMM edge cases and error conditions."""
    
    def test_single_observation(self, sample_models):
        """Test with single observation."""
        hmm = sample_models['gaussian_hmm']
        X = torch.randn(1, 2)
        
        # Should not raise error
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        predictions = hmm.predict(X=X, algorithm='viterbi')
        assert len(predictions) == 1
        assert len(predictions[0]) == 1
    
    def test_single_state_model(self):
        """Test single state model."""
        hmm = GaussianHMM(n_states=1, n_features=2, seed=42)
        X = torch.randn(100, 2)
        
        hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        predictions = hmm.predict(X=X, algorithm='viterbi')
        
        # All predictions should be state 0
        assert torch.all(predictions[0] == 0)
    
    def test_invalid_lengths(self, sample_models, sample_data):
        """Test invalid lengths parameter."""
        hmm = sample_models['gaussian_hmm']
        X = sample_data['gaussian_data']
        
        with pytest.raises(ValueError):
            hmm.fit(X=X, lengths=[50, 30], max_iter=5, n_init=1, verbose=False)
    
    def test_empty_sequence(self, sample_models):
        """Test with empty sequence."""
        hmm = sample_models['gaussian_hmm']
        X = torch.randn(0, 2)
        
        with pytest.raises(ValueError):
            hmm.fit(X=X, max_iter=5, n_init=1, verbose=False)


class TestHSMMBasic:
    """Basic HSMM tests."""
    
    def test_multinomial_hsmm_initialization(self, sample_models):
        """Test MultinomialHSMM initialization."""
        hsmm = sample_models['multinomial_hsmm']
        assert hsmm.n_states == 3
        assert hsmm.n_features == 4
        assert hsmm.n_trials == 2
        assert hsmm.max_duration == 10
    
    def test_gaussian_hsmm_initialization(self, sample_models):
        """Test GaussianHSMM initialization."""
        hsmm = sample_models['gaussian_hsmm']
        assert hsmm.n_states == 3
        assert hsmm.n_features == 2
        assert hsmm.max_duration == 10
    
    def test_hsmm_duration_matrix(self, sample_models):
        """Test HSMM duration matrix."""
        hsmm = sample_models['multinomial_hsmm']
        D = hsmm.D.exp()
        
        # Each row should sum to 1
        assert torch.allclose(D.sum(1), torch.ones(3, dtype=D.dtype))
        # All probabilities should be non-negative
        assert torch.all(D >= 0)
    
    def test_hsmm_fitting(self, sample_models, sample_data):
        """Test HSMM fitting."""
        hsmm = sample_models['gaussian_hsmm']
        X = sample_data['gaussian_data']
        
        hsmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        assert hsmm._params is not None
    
    def test_hsmm_map_prediction(self, sample_models, sample_data):
        """Test HSMM MAP prediction."""
        hsmm = sample_models['gaussian_hsmm']
        X = sample_data['gaussian_data']
        
        hsmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        predictions = hsmm.predict(X=X, algorithm='map')
        
        assert len(predictions) == 1
        assert len(predictions[0]) == len(X)
    
    def test_hsmm_viterbi_not_implemented(self, sample_models, sample_data):
        """Test that HSMM Viterbi is not implemented."""
        hsmm = sample_models['gaussian_hsmm']
        X = sample_data['gaussian_data']
        
        hsmm.fit(X=X, max_iter=5, n_init=1, verbose=False)
        
        with pytest.raises(NotImplementedError):
            hsmm.predict(X=X, algorithm='viterbi')


class TestFinancialData:
    """Test with financial time series data."""
    
    def test_financial_hmm_fitting(self, financial_data):
        """Test HMM fitting on financial data."""
        hmm = GaussianHMM(n_states=3, n_features=2, seed=42)
        X = financial_data['X']
        lengths = financial_data['lengths']
        
        hmm.fit(X=X, lengths=lengths, max_iter=20, n_init=3, verbose=False)
        
        # Test prediction
        predictions = hmm.predict(X=X, lengths=lengths, algorithm='viterbi')
        assert len(predictions) == len(lengths)
        
        # Test scoring
        scores = hmm.score(X=X, lengths=lengths)
        assert scores.shape == (len(lengths),)
        assert torch.isfinite(scores).all()
    
    def test_financial_state_interpretation(self, financial_data):
        """Test that financial states are interpretable."""
        hmm = GaussianHMM(n_states=3, n_features=2, seed=42)
        X = financial_data['X']
        
        hmm.fit(X=X, max_iter=20, n_init=3, verbose=False)
        predictions = hmm.predict(X=X, algorithm='viterbi')
        
        # Check that we get different states
        unique_states = torch.unique(predictions[0])
        assert len(unique_states) > 1, "Should have multiple states"


class TestPerformance:
    """Performance tests."""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self, large_dataset):
        """Test performance with large dataset."""
        hmm = GaussianHMM(n_states=5, n_features=3, seed=42)
        X = large_dataset['X']
        lengths = large_dataset['lengths']
        
        # Should complete without errors
        hmm.fit(X=X, lengths=lengths, max_iter=10, n_init=1, verbose=False)
        predictions = hmm.predict(X=X, lengths=lengths, algorithm='viterbi')
        
        assert len(predictions) == len(lengths)
    
    @pytest.mark.slow
    def test_high_dimensional_performance(self, high_dimensional_data):
        """Test performance with high-dimensional data."""
        hmm = GaussianHMM(n_states=10, n_features=10, seed=42)
        X = high_dimensional_data['X']
        lengths = high_dimensional_data['lengths']
        
        hmm.fit(X=X, lengths=lengths, max_iter=10, n_init=1, verbose=False)
        predictions = hmm.predict(X=X, lengths=lengths, algorithm='viterbi')
        
        assert len(predictions) == len(lengths)


if __name__ == '__main__':
    pytest.main([__file__])
