import unittest

import torch
from torch.distributions import Distribution

from chadhmm.hmm import MultinomialHMM  # type:ignore


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.hmm = MultinomialHMM(2, 3)

    def test_pdf_subclass(self):
        self.assertTrue(issubclass(type(self.hmm.pdf), Distribution))

    def test_dof_is_int(self):
        self.assertIsInstance(self.hmm.dof, int)

    def test_emission_pdf(self):
        emission_pdf = self.hmm.sample_emission_pdf()
        self.assertIsInstance(emission_pdf, Distribution)

    def test_emission_pdf_with_data(self):
        # Create some test data
        X = torch.tensor([0, 1, 2, 0, 1])
        emission_pdf = self.hmm.sample_emission_pdf(X)
        self.assertIsInstance(emission_pdf, Distribution)
