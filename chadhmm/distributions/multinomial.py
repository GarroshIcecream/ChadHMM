import torch
from torch.distributions import Multinomial

from chadhmm.utilities import constraints, utils


class MultinomialDist(Multinomial):
    def __init__(self, logits: torch.Tensor, trials: int = 1):
        super().__init__(total_count=trials, logits=logits)

    @property
    def dof(self):
        return self.batch_shape * (self.event_shape - 1)

    @classmethod
    def sample_emission_pdf(
        cls,
        trials: int,
        batch_shape: int,
        event_shape: int,
        alpha: float = 1.0,
        X: torch.Tensor | None = None,
    ):
        if X is not None:
            emission_freqs = torch.bincount(X) / X.shape[0]
            emission_matrix = torch.log(emission_freqs.expand(batch_shape, -1))
        else:
            emission_matrix = torch.log(
                constraints.sample_probs(alpha, (batch_shape, event_shape))
            )

        return cls(emission_matrix, trials)

    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: utils.ContextualVariables | None = None,
    ):
        self.logits = torch.log(self._compute_B(X, posterior, theta))

    def _compute_B(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: utils.ContextualVariables | None = None,
    ) -> torch.Tensor:
        """Compute the emission probabilities for each hidden state."""
        if theta is not None:
            # TODO: Implement contextualized emissions
            raise NotImplementedError(
                "Contextualized emissions not implemented for MultinomialEmissions"
            )
        else:
            new_B = posterior.T @ X
            new_B /= posterior.T.sum(1, keepdim=True)

        return new_B
