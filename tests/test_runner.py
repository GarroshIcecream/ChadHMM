"""
Enhanced test runner for ChadHMM with comprehensive test discovery and reporting.
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def discover_and_run_tests():
    """Discover and run all tests with detailed reporting."""
    
    # Test discovery
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    
    # Discover all test modules
    suite = loader.discover(
        start_dir=str(start_dir),
        pattern='test_*.py',
        top_level_dir=str(project_root)
    )
    
    # Count tests
    test_count = suite.countTestCases()
    print(f"Discovered {test_count} test cases")
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("\n" + "="*70)
    print("RUNNING CHADHMM TEST SUITE")
    print("="*70)
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
            
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

def run_specific_test_module(module_name):
    """Run tests from a specific module."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0

def run_performance_tests():
    """Run performance-focused tests."""
    print("Running performance tests...")
    return run_specific_test_module('tests.test_integration.TestPerformance')

def run_unit_tests():
    """Run unit tests only."""
    print("Running unit tests...")
    modules = [
        'tests.test_hmm_comprehensive',
        'tests.test_hsmm_comprehensive', 
        'tests.test_utilities'
    ]
    
    all_passed = True
    for module in modules:
        print(f"\nRunning {module}...")
        passed = run_specific_test_module(module)
        all_passed = all_passed and passed
        
    return all_passed

def run_integration_tests():
    """Run integration tests only."""
    print("Running integration tests...")
    return run_specific_test_module('tests.test_integration')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ChadHMM Test Runner')
    parser.add_argument(
        '--type', 
        choices=['all', 'unit', 'integration', 'performance'],
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--module',
        help='Specific test module to run'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.module:
        success = run_specific_test_module(args.module)
    elif args.type == 'unit':
        success = run_unit_tests()
    elif args.type == 'integration':
        success = run_integration_tests()
    elif args.type == 'performance':
        success = run_performance_tests()
    else:  # all
        success = discover_and_run_tests()
    
    sys.exit(0 if success else 1)
