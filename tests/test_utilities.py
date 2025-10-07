"""
Comprehensive tests for utility functions and classes.
Tests constraints, convergence handling, seed generation, and utility functions.
"""

import unittest
import torch
import numpy as np

from chadhmm.utils import (
    constraints,
    ConvergenceHandler,
    SeedGenerator
)
from chadhmm.schemas import (
    Transitions,
    CovarianceType,
    InformCriteria,
    Observations,
    ContextualVariables
)


class TestConstraints(unittest.TestCase):
    """Test constraint utility functions."""
    
    def test_sample_probs(self):
        """Test probability sampling."""
        # Test basic sampling
        probs = constraints.sample_probs(1.0, (3, 4))
        self.assertEqual(probs.shape, (3, 4))
        self.assertTrue(torch.allclose(probs.sum(1), torch.ones(3, dtype=probs.dtype)))
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))
        
        # Test with different prior
        probs2 = constraints.sample_probs(0.5, (2, 3))
        self.assertEqual(probs2.shape, (2, 3))
        self.assertTrue(torch.allclose(probs2.sum(1), torch.ones(2, dtype=probs2.dtype)))
        
    def test_sample_A_ergodic(self):
        """Test ergodic transition matrix sampling."""
        A = constraints.sample_A(1.0, 3, Transitions.ERGODIC)
        self.assertEqual(A.shape, (3, 3))
        self.assertTrue(torch.allclose(A.sum(1), torch.ones(3, dtype=A.dtype)))
        self.assertTrue(torch.all(A > 0))  # All transitions possible
        
    def test_sample_A_left_to_right(self):
        """Test left-to-right transition matrix sampling."""
        A = constraints.sample_A(1.0, 3, Transitions.LEFT_TO_RIGHT)
        self.assertEqual(A.shape, (3, 3))
        self.assertTrue(torch.allclose(A.sum(1), torch.ones(3, dtype=A.dtype)))
        # Should be upper triangular
        self.assertTrue(torch.all(torch.tril(A, diagonal=-1) == 0))
        
    def test_sample_A_semi(self):
        """Test semi-Markov transition matrix sampling."""
        A = constraints.sample_A(1.0, 3, Transitions.SEMI)
        self.assertEqual(A.shape, (3, 3))
        self.assertTrue(torch.allclose(A.sum(1), torch.ones(3, dtype=A.dtype)))
        # Diagonal should be zero
        self.assertTrue(torch.all(torch.diag(A) == 0))
        
    def test_compute_information_criteria(self):
        """Test information criteria computation."""
        log_likelihood = torch.tensor([-100.0, -200.0])
        dof = 10
        n_samples = 100
        
        # Test AIC
        aic = constraints.compute_information_criteria(
            n_samples, log_likelihood, dof, InformCriteria.AIC
        )
        expected_aic = -2.0 * log_likelihood + 2.0 * dof
        self.assertTrue(torch.allclose(aic, expected_aic))
        
        # Test BIC
        bic = constraints.compute_information_criteria(
            n_samples, log_likelihood, dof, InformCriteria.BIC
        )
        expected_bic = -2.0 * log_likelihood + dof * np.log(n_samples)
        self.assertTrue(torch.allclose(bic, expected_bic))
        
        # Test HQC
        hqc = constraints.compute_information_criteria(
            n_samples, log_likelihood, dof, InformCriteria.HQC
        )
        expected_hqc = -2.0 * log_likelihood + 2.0 * dof * np.log(np.log(n_samples))
        self.assertTrue(torch.allclose(hqc, expected_hqc))
        
    def test_is_valid_A(self):
        """Test transition matrix validation."""
        # Valid ergodic matrix
        valid_A = torch.ones(3, 3) / 3
        self.assertTrue(constraints.is_valid_A(valid_A.log(), Transitions.ERGODIC))
        
        # Invalid ergodic matrix (has zeros)
        invalid_A = torch.eye(3)
        self.assertFalse(constraints.is_valid_A(invalid_A.log(), Transitions.ERGODIC))
        
        # Valid left-to-right matrix
        ltr_A = torch.triu(torch.ones(3, 3))
        ltr_A = ltr_A / ltr_A.sum(1, keepdim=True)
        self.assertTrue(constraints.is_valid_A(ltr_A.log(), Transitions.LEFT_TO_RIGHT))
        
        # Valid semi matrix
        semi_A = torch.ones(3, 3)
        semi_A.fill_diagonal_(0)
        semi_A = semi_A / semi_A.sum(1, keepdim=True)
        self.assertTrue(constraints.is_valid_A(semi_A.log(), Transitions.SEMI))
        
    def test_log_normalize(self):
        """Test log normalization."""
        # Test 1D normalization
        log_probs = torch.randn(3)
        normalized = constraints.log_normalize(log_probs, dim=0)
        self.assertTrue(torch.allclose(normalized.logsumexp(0), torch.zeros(1)))
        
        # Test 2D normalization along dim=1
        log_probs_2d = torch.randn(3, 4)
        normalized_2d = constraints.log_normalize(log_probs_2d, dim=1)
        self.assertTrue(torch.allclose(normalized_2d.logsumexp(1), torch.zeros(3)))
        
    def test_validate_lambdas(self):
        """Test lambda validation for Poisson parameters."""
        # Valid lambdas
        valid_lambdas = torch.ones(3, 2)
        result = constraints.validate_lambdas(valid_lambdas, 3, 2)
        self.assertTrue(torch.equal(result, valid_lambdas))
        
        # Invalid shape
        with self.assertRaises(ValueError):
            constraints.validate_lambdas(torch.ones(2, 3), 3, 2)
            
        # Invalid values (negative)
        with self.assertRaises(ValueError):
            constraints.validate_lambdas(torch.ones(3, 2) * -1, 3, 2)
            
        # Invalid values (NaN)
        with self.assertRaises(ValueError):
            constraints.validate_lambdas(torch.tensor([[1.0, float('nan')], [1.0, 1.0], [1.0, 1.0]]), 3, 2)
            
    def test_validate_covars(self):
        """Test covariance matrix validation."""
        n_states, n_features = 3, 2
        
        # Test spherical covariance
        spherical_covars = torch.ones(n_states)
        result = constraints.validate_covars(
            spherical_covars, CovarianceType.SPHERICAL, n_states, n_features
        )
        self.assertTrue(torch.equal(result, spherical_covars))
        
        # Test diagonal covariance
        diag_covars = torch.ones(n_states, n_features)
        result = constraints.validate_covars(
            diag_covars, CovarianceType.DIAG, n_states, n_features
        )
        self.assertTrue(torch.equal(result, diag_covars))
        
        # Test full covariance
        full_covars = torch.eye(n_features).unsqueeze(0).expand(n_states, -1, -1)
        result = constraints.validate_covars(
            full_covars, CovarianceType.FULL, n_states, n_features
        )
        self.assertTrue(torch.equal(result, full_covars))
        
    def test_init_covars(self):
        """Test covariance initialization."""
        n_states = 3
        tied_cv = torch.eye(2)
        
        # Test spherical
        spherical = constraints.init_covars(
            tied_cv, CovarianceType.SPHERICAL, n_states
        )
        self.assertEqual(spherical.shape, (n_states,))
        
        # Test tied
        tied = constraints.init_covars(
            tied_cv, CovarianceType.TIED, n_states
        )
        self.assertEqual(tied.shape, (2, 2))
        
        # Test diagonal
        diag = constraints.init_covars(
            tied_cv, CovarianceType.DIAG, n_states
        )
        self.assertEqual(diag.shape, (n_states, 2))
        
        # Test full
        full = constraints.init_covars(
            tied_cv, CovarianceType.FULL, n_states
        )
        self.assertEqual(full.shape, (n_states, 2, 2))


class TestConvergenceHandler(unittest.TestCase):
    """Test convergence handling."""
    
    def test_initialization(self):
        """Test convergence handler initialization."""
        conv = ConvergenceHandler(
            tol=0.01,
            max_iter=10,
            n_init=3,
            post_conv_iter=2,
            verbose=False
        )
        
        self.assertEqual(conv.tol, 0.01)
        self.assertEqual(conv.max_iter, 10)
        self.assertEqual(conv.n_init, 3)
        self.assertEqual(conv.post_conv_iter, 2)
        self.assertFalse(conv.verbose)
        
    def test_convergence_detection(self):
        """Test convergence detection."""
        conv = ConvergenceHandler(tol=0.01, max_iter=10, n_init=1, verbose=False)
        
        # First iteration
        converged = conv.push_pull(-100.0, 0, 0)
        self.assertFalse(converged)
        
        # Second iteration with small improvement
        converged = conv.push_pull(-100.01, 1, 0)
        self.assertFalse(converged)
        
        # Third iteration with improvement below tolerance
        converged = conv.push_pull(-100.005, 2, 0)
        self.assertTrue(converged)
        
    def test_max_iterations(self):
        """Test maximum iterations handling."""
        conv = ConvergenceHandler(tol=0.01, max_iter=3, n_init=1, verbose=False)
        
        # Should not converge before max_iter
        conv.push_pull(-100.0, 0, 0)
        conv.push_pull(-100.1, 1, 0)
        converged = conv.push_pull(-100.2, 2, 0)
        self.assertFalse(converged)
        
        # Should converge at max_iter
        converged = conv.push_pull(-100.3, 3, 0)
        self.assertTrue(converged)
        
    def test_multiple_initializations(self):
        """Test multiple initialization handling."""
        conv = ConvergenceHandler(tol=0.01, max_iter=5, n_init=2, verbose=False)
        
        # First initialization
        conv.push_pull(-100.0, 0, 0)
        conv.push_pull(-100.1, 1, 0)
        conv.push_pull(-100.2, 2, 0)
        
        # Second initialization
        conv.push_pull(-90.0, 0, 1)
        conv.push_pull(-90.1, 1, 1)
        converged = conv.push_pull(-90.2, 2, 1)
        self.assertTrue(converged)


class TestSeedGenerator(unittest.TestCase):
    """Test seed generation."""
    
    def test_initialization(self):
        """Test seed generator initialization."""
        # With seed
        gen = SeedGenerator(42)
        self.assertEqual(gen.seed, 42)
        
        # Without seed
        gen2 = SeedGenerator(None)
        self.assertIsNotNone(gen2.seed)
        
    def test_reproducibility(self):
        """Test seed reproducibility."""
        gen1 = SeedGenerator(42)
        gen2 = SeedGenerator(42)
        self.assertEqual(gen1.seed, gen2.seed)
        
    def test_different_seeds(self):
        """Test different seeds."""
        gen1 = SeedGenerator(42)
        gen2 = SeedGenerator(43)
        self.assertNotEqual(gen1.seed, gen2.seed)


class TestUtils(unittest.TestCase):
    """Test utility functions and classes."""
    
    def test_observations_initialization(self):
        """Test Observations class initialization."""
        sequences = [torch.randn(10, 2), torch.randn(15, 2)]
        log_probs = [torch.randn(10, 3), torch.randn(15, 3)]
        lengths = [10, 15]
        
        obs = Observations(sequences, log_probs, lengths)
        
        self.assertEqual(len(obs.sequence), 2)
        self.assertEqual(len(obs.log_probs), 2)
        self.assertEqual(obs.lengths, lengths)
        
    def test_contextual_variables_initialization(self):
        """Test ContextualVariables class initialization."""
        n_context = 2
        context_matrix = [torch.randn(3, 10), torch.randn(3, 15)]
        time_dependent = True
        
        ctx = ContextualVariables(n_context, context_matrix, time_dependent)
        
        self.assertEqual(ctx.n_context, n_context)
        self.assertEqual(len(ctx.context_matrix), 2)
        self.assertTrue(ctx.time_dependent)
        
    def test_contextual_variables_time_independent(self):
        """Test time-independent contextual variables."""
        n_context = 2
        context_matrix = [torch.randn(3, 1), torch.randn(3, 1)]
        time_dependent = False
        
        ctx = ContextualVariables(n_context, context_matrix, time_dependent)
        
        self.assertEqual(ctx.n_context, n_context)
        self.assertFalse(ctx.time_dependent)


class TestUtilityFunctions(unittest.TestCase):
    """Test additional utility functions."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        # Test Transitions enum
        self.assertEqual(Transitions.ERGODIC.value, "ergodic")
        self.assertEqual(Transitions.LEFT_TO_RIGHT.value, "left-to-right")
        self.assertEqual(Transitions.SEMI.value, "semi")
        
        # Test InformCriteria enum
        self.assertEqual(InformCriteria.AIC.value, "AIC")
        self.assertEqual(InformCriteria.BIC.value, "BIC")
        self.assertEqual(InformCriteria.HQC.value, "HQC")
        
        # Test CovarianceType enum
        self.assertEqual(CovarianceType.FULL.value, "full")
        self.assertEqual(CovarianceType.DIAG.value, "diag")
        self.assertEqual(CovarianceType.TIED.value, "tied")
        self.assertEqual(CovarianceType.SPHERICAL.value, "spherical")
        
    def test_edge_cases(self):
        """Test edge cases in utility functions."""
        # Test with very small tolerance
        conv = ConvergenceHandler(tol=1e-10, max_iter=5, n_init=1, verbose=False)
        conv.push_pull(-100.0, 0, 0)
        converged = conv.push_pull(-100.0 + 1e-11, 1, 0)
        self.assertTrue(converged)
        
        # Test with zero tolerance
        conv2 = ConvergenceHandler(tol=0.0, max_iter=5, n_init=1, verbose=False)
        conv2.push_pull(-100.0, 0, 0)
        converged = conv2.push_pull(-100.0, 1, 0)
        self.assertTrue(converged)
        
    def test_error_handling(self):
        """Test error handling in utility functions."""
        # Test invalid transition type
        with self.assertRaises(NotImplementedError):
            constraints.sample_A(1.0, 3, "invalid")
            
        # Test invalid information criteria
        with self.assertRaises(AttributeError):
            constraints.compute_information_criteria(
                100, torch.tensor([-100.0]), 10, "invalid"
            )
            
        # Test invalid covariance type
        with self.assertRaises(NotImplementedError):
            constraints.init_covars(torch.eye(2), "invalid", 3)


if __name__ == '__main__':
    unittest.main()
