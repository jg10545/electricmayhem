"""

Implementation of the BAxUS algorithm. Following the
tutorial here: https://botorch.org/tutorials/baxus

"""

import math
import os
from dataclasses import dataclass
import logging

import botorch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from botorch.acquisition.analytic import ExpectedImprovement
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf



@dataclass
class BaxusState:
    dim: int
    eval_budget: int
    new_bins_on_split: int = 3
    d_init: int = float("nan")  # Note: post-initialized
    target_dim: int = float("nan")  # Note: post-initialized
    n_splits: int = float("nan")  # Note: post-initialized
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        n_splits = round(math.log(self.dim, self.new_bins_on_split + 1))
        self.d_init = 1 + np.argmin(
            np.abs(
                (1 + np.arange(self.new_bins_on_split))
                * (1 + self.new_bins_on_split) ** n_splits
                - self.dim
            )
        )
        self.target_dim = self.d_init
        self.n_splits = n_splits

    @property
    def split_budget(self) -> int:
        return round(
            -1
            * (self.new_bins_on_split * self.eval_budget * self.target_dim)
            / (self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1)))
        )

    @property
    def failure_tolerance(self) -> int:
        if self.target_dim == self.dim:
            return self.target_dim
        k = math.floor(math.log(self.length_min / self.length_init, 0.5))
        split_budget = self.split_budget
        return min(self.target_dim, max(1, math.floor(split_budget / k)))


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state



def embedding_matrix(input_dim: int, target_dim: int, 
                     device: str = "cpu") -> torch.Tensor:
    """
    
    """
    if (
        target_dim >= input_dim
    ):  # return identity matrix if target size greater than input size
        return torch.eye(input_dim, device=device, dtype=torch.double)

    input_dims_perm = (
        torch.randperm(input_dim, device=device) + 1
    )  # add 1 to indices for padding column in matrix

    bins = torch.tensor_split(
        input_dims_perm, target_dim
    )  # split dims into almost equally-sized bins
    bins = torch.nn.utils.rnn.pad_sequence(
        bins, batch_first=True
    )  # zero pad bins, the index 0 will be cut off later

    mtrx = torch.zeros(
        (target_dim, input_dim + 1), dtype=torch.double, device=device
    )  # add one extra column for padding
    mtrx = mtrx.scatter_(
        1,
        bins,
        2 * torch.randint(2, (target_dim, input_dim), dtype=torch.double,
                          device=device) - 1,
    )  # fill mask with random +/- 1 at indices

    return mtrx[:, 1:]  # cut off index zero as this corresponds to zero padding

def increase_embedding_and_observations(
        S: torch.Tensor, 
        X: torch.Tensor, 
        n_new_bins: int,
        device: str = "cpu") -> torch.Tensor:
    """
    
    """
    assert X.size(1) == S.size(0), "Observations don't lie in row space of S"

    S_update = S.clone()
    X_update = X.clone()

    for row_idx in range(len(S)):
        row = S[row_idx]
        idxs_non_zero = torch.nonzero(row)
        idxs_non_zero = idxs_non_zero[torch.randperm(len(idxs_non_zero))].squeeze()

        non_zero_elements = row[idxs_non_zero].squeeze()

        n_row_bins = min(
            n_new_bins, len(idxs_non_zero)
        )  # number of new bins is always less or equal than the contributing input dims in the row minus one

        new_bins = torch.tensor_split(idxs_non_zero, n_row_bins)[
            1:
        ]  # the dims in the first bin won't be moved
        elements_to_move = torch.tensor_split(non_zero_elements, n_row_bins)[1:]

        new_bins_padded = torch.nn.utils.rnn.pad_sequence(
            new_bins, batch_first=True
        )  # pad the tuples of bins with zeros to apply _scatter
        els_to_move_padded = torch.nn.utils.rnn.pad_sequence(
            elements_to_move, batch_first=True
        )

        S_stack = torch.zeros(
            (n_row_bins - 1, len(row) + 1), device=device, dtype=torch.double
        )  # submatrix to stack on S_update

        S_stack = S_stack.scatter_(
            1, new_bins_padded + 1, els_to_move_padded
        )  # fill with old values (add 1 to indices for padding column)

        S_update[
            row_idx, torch.hstack(new_bins)
        ] = 0  # set values that were move to zero in current row

        X_update = torch.hstack(
            (X_update, X[:, row_idx].reshape(-1, 1).repeat(1, len(new_bins)))
        )  # repeat observations for row at the end of X (column-wise)
        S_update = torch.vstack(
            (S_update, S_stack[:, 1:])
        )  # stack onto S_update except for padding column

    return S_update, X_update


def get_initial_points(dim, n_pts, seed=0, device: str = "cpu"):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = (
        #2 * sobol.draw(n=n_pts).to(dtype=torch.double, device=device) - 1
        sobol.draw(n=n_pts).to(dtype=torch.double, device=device)
    )  # points have to be in [-1, 1]^d // that was for branin turtorial;
       # let's stick with unit hypercube here.
    
    return X_init


def create_candidate(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [-1, 1]^d
    Y,  # Function values
    train_Y,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    device: str = "cpu"
):
    """
    
    """
    assert acqf in ("ts", "ei")
    assert X.min() >= -1.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.detach().view(-1)
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    #tr_lb = torch.clamp(x_center - weights * state.length, -1.0, 1.0)
    #tr_ub = torch.clamp(x_center + weights * state.length, -1.0, 1.0)
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=torch.double, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=torch.double, 
                          device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=1)

    elif acqf == "ei":
        ei = ExpectedImprovement(model, train_Y.max(), maximize=True)
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next



class BAxUS():
    """
    Assume we're MAXIMIZING the function
    """
    
    def __init__(self, func, dim, n_init, eval_budget, device="cpu",
                 num_restarts=10,
                 raw_samples=512,
                 acqf="ts",
                 max_cholesky_size = float("inf") 
                 ):
        """
        

        Parameters
        ----------
        dim : TYPE
            DESCRIPTION.
        n_init : TYPE
            DESCRIPTION.
        eval_budget : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.func = func
        self.dim = dim
        self.n_init = n_init
        self.eval_budget = eval_budget
        self.device = device
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.acqf = acqf
        self.n_candidates = min(5000, max(2000, 200 * dim))
        self.max_cholesky_size = max_cholesky_size
        
        # BAxUS state
        self.state = BaxusState(dim, eval_budget-n_init)
        # initial low-dimension matrix
        self.S = embedding_matrix(input_dim=self.state.dim,
                                  target_dim=self.state.d_init)
        # Sobol-sample initial points
        self.X_baxus_target = get_initial_points(self.state.d_init, n_init)
        # map those low-dimensional starting points to the function's input space
        self.X_baxus_input = self.X_baxus_target @ self.S 
        # evaluate function for each initial point
        self.Y_baxus = torch.tensor([func(x) for x in self.X_baxus_input], 
                               dtype=torch.double, device=device
                               ).unsqueeze(-1)
        
    def fit_gp(self):
        """
        TODO: check settings of the kernel; do outputscale constraints
        make sense?
        """
        # Fit a GP model
        self.train_Y = (self.Y_baxus - self.Y_baxus.mean()) / self.Y_baxus.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = (
            ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.state.target_dim,
                    lengthscale_constraint=Interval(0.005, 10),
                ),
                outputscale_constraint=Interval(0.05, 10),
            )
        )
        model = SingleTaskGP(
            self.X_baxus_target, self.train_Y, 
            covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        # Do the fitting and acquisition function optimization inside the Cholesky context
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            # Fit the model
            try:
                fit_gpytorch_mll(mll)
            except ModelFittingError:
                logging.warn("ModelFittingError!")
                # Right after increasing the target dimensionality, the 
                # covariance matrix becomes indefinite.
                # In this case, the Cholesky decomposition might fail due to 
                # numerical instabilities
                # In this case, we revert to Adam-based optimization
                optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

                for _ in range(100):
                    optimizer.zero_grad()
                    output = model(self.X_baxus_target)
                    loss = -mll(output, self.train_Y.flatten())
                    loss.backward()
                    optimizer.step()
        return model
    
    def _create_batch(self, model):
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_next_target = create_candidate(
                state=self.state,
                model=model,
                X=self.X_baxus_target,
                Y=self.train_Y,
                train_Y=self.train_Y,
                n_candidates=self.n_candidates,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                acqf=self.acqf,
            )
            return X_next_target
        
    def run_one_step(self):
        # fit a GP
        model = self.fit_gp()
        # sample the next batch
        X_next_target = self._create_batch(model)
        # map low-dimensional space to input space
        X_next_input = X_next_target @ self.S 

        Y_next = torch.tensor(
            [self.func(x) for x in X_next_input], dtype=torch.double, 
            device=self.device).unsqueeze(-1)

        # Update state
        self.state = update_state(state=self.state, Y_next=Y_next)

        # Append data
        self.X_baxus_input = torch.cat((self.X_baxus_input, X_next_input), dim=0)
        self.X_baxus_target = torch.cat((self.X_baxus_target, X_next_target), dim=0)
        self.Y_baxus = torch.cat((self.Y_baxus, Y_next), dim=0)

        # Print current status
        logging.info(
            f"iteration {len(self.X_baxus_input)}, d={len(self.X_baxus_target.T)})  Best value: {self.state.best_value:.3}, TR length: {self.state.length:.3}"
        )

        if self.state.restart_triggered:
            self.state.restart_triggered = False
            logging.info("increasing target space")
            self.S, self.X_baxus_target = increase_embedding_and_observations(
                self.S, self.X_baxus_target, self.state.new_bins_on_split
            )
            logging.info(f"new dimensionality: {len(self.S)}")
            self.state.target_dim = len(self.S)
            self.state.length = self.state.length_init
            self.state.failure_counter = 0
            self.state.success_counter = 0
            
    def get_best_X(self):
        best_index = self.Y_baxus.numpy().ravel().argmax()
        return self.X_baxus_input[best_index]
    
    def get_best_Y(self):
        return torch.max(self.Y_baxus)
    
    def get_last_X(self):
        return self.X_baxus_input[-1]
        
        
