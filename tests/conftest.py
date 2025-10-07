"""
Pytest configuration for ChadHMM tests.
Provides fixtures and configuration for comprehensive testing.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from chadhmm.hmm import MultinomialHMM, GaussianHMM, GaussianMixtureHMM, PoissonHMM
from chadhmm.hsmm import MultinomialHSMM, GaussianHSMM, GaussianMixtureHSMM, PoissonHSMM
from chadhmm.utils import constraints


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    torch.manual_seed(42)
    return {
        'multinomial_data': torch.randint(0, 4, (100,)),
        'gaussian_data': torch.randn(100, 2),
        'poisson_data': torch.poisson(torch.ones(100, 2) * 2),
        'lengths': [50, 50],
        'single_length': [100]
    }


@pytest.fixture
def sample_models():
    """Provide sample models for testing."""
    return {
        'multinomial_hmm': MultinomialHMM(
            n_states=3, n_features=4, n_trials=2, seed=42
        ),
        'gaussian_hmm': GaussianHMM(
            n_states=3, n_features=2, seed=42
        ),
        'gaussian_mixture_hmm': GaussianMixtureHMM(
            n_states=2, n_features=2, n_components=3, seed=42
        ),
        'poisson_hmm': PoissonHMM(
            n_states=3, n_features=2, seed=42
        ),
        'multinomial_hsmm': MultinomialHSMM(
            n_states=3, n_features=4, n_trials=2, max_duration=10, seed=42
        ),
        'gaussian_hsmm': GaussianHSMM(
            n_states=3, n_features=2, max_duration=10, seed=42
        )
    }


@pytest.fixture
def temp_file():
    """Provide a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def device():
    """Provide device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def transition_types():
    """Provide different transition types for testing."""
    return [
        constraints.Transitions.ERGODIC,
        constraints.Transitions.LEFT_TO_RIGHT,
        constraints.Transitions.SEMI
    ]


@pytest.fixture
def information_criteria():
    """Provide different information criteria for testing."""
    return [
        constraints.InformCriteria.AIC,
        constraints.InformCriteria.BIC,
        constraints.InformCriteria.HQC
    ]


@pytest.fixture
def covariance_types():
    """Provide different covariance types for testing."""
    return [
        constraints.CovarianceType.FULL,
        constraints.CovarianceType.DIAG,
        constraints.CovarianceType.TIED,
        constraints.CovarianceType.SPHERICAL
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file names
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        else:
            item.add_marker(pytest.mark.unit)
            
        # Add slow marker for tests that might take time
        if any(keyword in item.nodeid.lower() for keyword in 
               ['large', 'stress', 'memory', 'time']):
            item.add_marker(pytest.mark.slow)


# Test data generators
@pytest.fixture
def financial_data():
    """Generate synthetic financial time series data."""
    torch.manual_seed(42)
    n_samples = 1000
    
    # Simulate 3 market states: bull, bear, sideways
    states = torch.zeros(n_samples, dtype=torch.long)
    returns = torch.zeros(n_samples)
    volumes = torch.zeros(n_samples)
    
    current_state = 0
    for t in range(n_samples):
        if current_state == 0:  # Bull market
            returns[t] = torch.normal(0.02, 0.01, size=())
            volumes[t] = torch.normal(1.5, 0.3)
        elif current_state == 1:  # Bear market
            returns[t] = torch.normal(-0.02, 0.01, size=())
            volumes[t] = torch.normal(1.5, 0.3)
        else:  # Sideways market
            returns[t] = torch.normal(0.0, 0.005, size=())
            volumes[t] = torch.normal(0.8, 0.2)
        
        # State transitions
        if torch.rand(1) < 0.05:
            current_state = (current_state + 1) % 3
        states[t] = current_state
    
    # Combine into feature matrix
    X = torch.stack([returns, volumes], dim=1)
    
    return {
        'X': X,
        'states': states,
        'lengths': [500, 500]
    }


@pytest.fixture
def sequence_data():
    """Generate sequence data for testing."""
    torch.manual_seed(42)
    
    # Generate sequences of different lengths
    sequences = [
        torch.randint(0, 3, (50,)),
        torch.randint(0, 3, (75,)),
        torch.randint(0, 3, (25,))
    ]
    
    # Combine and convert to one-hot
    X = torch.cat(sequences)
    X_one_hot = torch.nn.functional.one_hot(X, 3).float()
    lengths = [len(seq) for seq in sequences]
    
    return {
        'X': X_one_hot,
        'X_raw': X,
        'lengths': lengths,
        'sequences': sequences
    }


# Performance testing fixtures
@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    torch.manual_seed(42)
    return {
        'X': torch.randn(5000, 3),
        'lengths': [2500, 2500]
    }


@pytest.fixture
def high_dimensional_data():
    """Generate high-dimensional data for testing."""
    torch.manual_seed(42)
    return {
        'X': torch.randn(1000, 10),
        'lengths': [500, 500]
    }


# Error testing fixtures
@pytest.fixture
def invalid_data():
    """Provide invalid data for error testing."""
    return {
        'wrong_shape': torch.randn(100, 3),  # Wrong number of features
        'negative_values': torch.randn(100, 2) - 5,  # Negative values for Poisson
        'nan_values': torch.tensor([[1.0, float('nan')], [2.0, 3.0]]),
        'inf_values': torch.tensor([[1.0, float('inf')], [2.0, 3.0]])
    }


# Utility functions for tests
def assert_tensor_properties(tensor, shape=None, dtype=None, finite=True):
    """Assert tensor properties for testing."""
    if shape is not None:
        assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
    if dtype is not None:
        assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"
    if finite:
        assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"


def assert_model_properties(model, n_states=None, n_features=None):
    """Assert model properties for testing."""
    if n_states is not None:
        assert model.n_states == n_states, f"Expected {n_states} states, got {model.n_states}"
    if n_features is not None:
        assert model.n_features == n_features, f"Expected {n_features} features, got {model.n_features}"


def assert_predictions_valid(predictions, lengths, n_states):
    """Assert that predictions are valid."""
    assert len(predictions) == len(lengths), "Number of predictions doesn't match number of sequences"
    for i, (pred, length) in enumerate(zip(predictions, lengths)):
        assert len(pred) == length, f"Prediction {i} length {len(pred)} doesn't match expected {length}"
        assert torch.all(pred >= 0), f"Predictions contain negative values: {pred}"
        assert torch.all(pred < n_states), f"Predictions contain invalid states: {pred}"


# Pytest configuration for test discovery
def pytest_generate_tests(metafunc):
    """Generate parametrized tests based on fixtures."""
    if 'transition_type' in metafunc.fixturenames:
        metafunc.parametrize('transition_type', [
            constraints.Transitions.ERGODIC,
            constraints.Transitions.LEFT_TO_RIGHT,
            constraints.Transitions.SEMI
        ])
    
    if 'criterion' in metafunc.fixturenames:
        metafunc.parametrize('criterion', [
            constraints.InformCriteria.AIC,
            constraints.InformCriteria.BIC,
            constraints.InformCriteria.HQC
        ])
