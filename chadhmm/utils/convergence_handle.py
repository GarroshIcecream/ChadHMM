import matplotlib.pyplot as plt
import torch


class ConvergenceHandler:
    """
    Convergence Monitor
    ----------
    Convergence monitor for HMM training. Stores the score at each iteration
    and checks for convergence.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.
    n_init : int
        Number of initializations.
    tol : float
        Convergence threshold.
    post_conv_iter : int
        Number of iterations to run after convergence.
    verbose : bool
        Print convergence information.
    """

    def __init__(
        self,
        max_iter: int,
        n_init: int,
        tol: float,
        post_conv_iter: int,
        verbose: bool = True,
    ):
        self.tol = tol
        self.verbose = verbose
        self.post_conv_iter = post_conv_iter
        self.max_iter = max_iter
        self.score = torch.full(
            size=(max_iter + 1, n_init), fill_value=float("nan"), dtype=torch.float64
        )
        self.delta = self.score.clone()

    def __repr__(self):
        return (f"ConvergenceHandler(tol={self.tol}, "
                f"n_iters={self.max_iter + 1}, "
                f"post_conv_iter={self.post_conv_iter}, "
                f"converged={self.is_converged}, "
                f"verbose={self.verbose})")

    def push_pull(self, new_score: torch.Tensor, iter: int, rank: int) -> bool:
        """Push a new score and check for convergence."""
        self.push(new_score, iter, rank)
        return self.converged(iter, rank)

    def push(self, new_score: torch.Tensor, iter: int, rank: int):
        """Update the iteration count."""
        self.score[iter, rank] = new_score
        self.delta[iter, rank] = new_score - self.score[iter - 1, rank]

    def converged(self, iter: int, rank: int) -> bool:
        """Check if the model has converged and update the convergence monitor."""
        conv_lag = iter - self.post_conv_iter

        if conv_lag < 0:
            self.is_converged = False
        elif torch.all(self.delta[conv_lag:iter, rank] < self.tol):
            self.is_converged = True
        else:
            self.is_converged = False

        if self.verbose:
            score = self.score[iter, rank].item()
            delta = self.delta[iter, rank].item()

            if self.is_converged:
                print(
                    f"Model converged after {iter} iterations with "
                    f"log-likelihood: {score:.2f}"
                )
            elif iter == 0:
                print(f"Run {rank + 1} | Initialization | Score: {score:.2f}")
            else:
                print(
                    f"Run {rank + 1} | "
                    + f"Iteration: {iter} | "
                    + f"Score: {score:.2f} | "
                    + f"Delta: {delta:.2f} | "
                    + f"Converged = {self.is_converged}"
                )

        return self.is_converged

    def plot_convergence(self):
        labels = [f"Log-likelihood - Run #{i + 1}" for i in range(self.score.shape[1])]
        plt.style.use("ggplot")
        _, ax = plt.subplots(figsize=(10, 7))
        ax.plot(
            torch.arange(self.max_iter + 1),
            self.score.cpu(),
            linewidth=2,
            marker="o",
            markersize=5,
            label=labels,
        )

        ax.set_title("HMM Model Log-Likelihood Convergence")
        ax.set_xlabel("# Iterations")
        ax.set_ylabel("Log-likelihood")
        ax.legend(loc="lower right")
        plt.show()
