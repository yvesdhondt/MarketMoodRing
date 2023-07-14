from marketmoodring.portfolio_optimization.base import (
    PortfolioOptimizationBase,
    PortfolioOptimizationError,
)
import numpy as np
import numba
from scipy.optimize import minimize
from typing import Union
import pandas as pd
from scipy.stats import norm


class JointStochasticProgOptimization(PortfolioOptimizationBase):
    def __init__(self, n_regimes: int, objective: str = "max_avg_sharpe", *args, **kwargs):
        """
        Stochastic Programming Portfolio Optimization where each simulated regime sequence is jointly optimized for.

        Parameters
        ----------
        n_regimes : int
            The number of regimes.
        objective : str, optional
            The objective function to optimize for. The options are "max_avg_sharpe" and "min_avg_VaR".
        """
        super().__init__(n_regimes, *args, **kwargs)
        if objective not in ("max_avg_sharpe", "min_avg_VaR"):
            raise ValueError("objective is note one of max_avg_sharpe, min_avg_VaR")
        self._objective = objective

    def calculate_weights(self, fitted_states: np.ndarray, trans_mat: np.ndarray,
                          index_data: Union[np.ndarray, pd.DataFrame],
                          seq_length: int = 22, n_sequences: int = 1000, *args, **kwargs):
        """
        Calculate the weights for the given fitted states and asset returns.

        Parameters
        ----------
        fitted_states : np.ndarray
            The fitted states.
        trans_mat : np.ndarray
            The transition matrix.
        index_data : Union[np.ndarray, pd.DataFrame]
            The index or asset returns data.
        seq_length : int, optional
            The sequence length. The default is 22.
        n_sequences : int, optional
            The number of sequences. The default is 1000.

        Returns
        -------
        weights : np.ndarray
            The weights that maximize the average expected Sharpe Ratio over all possible state sequences
        """
        fitted_states = fitted_states.astype(int)
        states = np.unique(fitted_states)
        n_states = len(states)

        regime_groups = index_data.groupby(fitted_states)
        mu = regime_groups.mean().to_numpy()
        n_assets = index_data.shape[1]
        sigma = regime_groups.cov().to_numpy().reshape(n_states, n_assets, n_assets)

        sequences = _calc_n_sequences(n_sequences, fitted_states[-1], trans_mat, states, seq_length)
        # np.unique returns sorted unique values, we only care about counts
        state_counts = _count_states(sequences, n_states)

        if self._objective == "max_avg_sharpe":
            result = minimize(
                fun=lambda weights: _max_avg_sharpe(weights, mu, sigma, state_counts, seq_length),
                x0=np.ones(shape=n_assets) / n_assets,
                bounds=[(0, 1)] * n_assets,
                method="SLSQP",
                constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1},
                tol=1e-5
            )
        elif self._objective == "min_avg_VaR":
            result = minimize(
                fun=lambda weights: _min_avg_VaR(weights, mu, sigma, state_counts),
                x0=np.ones(shape=n_assets) / n_assets,
                bounds=[(0, 1)] * n_assets,
                method="SLSQP",
                constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1},
                tol=1e-5
            )

        return result.x

    def __str__(self):
        return "JointStochasticProgOpt"


@numba.jit("int64[:](int64,float64[:,:],int64[:],int64)", nopython=True, nogil=True)
def _calc_sequence(start_state, trans_mat, states, seq_length):
    """
    Calculate a random sequence of states.

    Parameters
    ----------
    start_state : int64
        The initial state.
    trans_mat : float64[:,:]
        The transition matrix.
    states : int64[:]
        The possible states.
    seq_length : int64
        The sequence length.

    Returns
    -------
    sequence : int64[:]
    """
    sequence = np.empty(seq_length, dtype=np.int64)
    sequence[0] = start_state
    for i in range(1, seq_length):
        sequence[i] = states[np.searchsorted(np.cumsum(trans_mat[sequence[i-1]]), np.random.random(), side="right")]
    return sequence


def _calc_n_sequences(n_sequences, start_state, trans_mat, states, seq_length):
    """
    Calculate a random sequence of states.

    Parameters
    ----------
    n_sequences : int64
        The number of sequences.
    start_state : int64
        The initial state.
    trans_mat : float64[:,:]
        The transition matrix.
    states : int64[:]
        The possible states.
    seq_length : int64
        The sequence length.

    Returns
    -------
    sequences : int64[:,:]
    """
    sequences = np.empty((n_sequences, seq_length))
    return np.apply_along_axis(
        lambda _: _calc_sequence(start_state, trans_mat, states, seq_length), axis=1, arr=sequences
    )


def _calc_sharpes(weights, mu, sigma, state_counts, seq_length, rf=0):
    """
    Calculate Sharpe Ratios.

    Parameters
    ----------
    weights : float64[:]
        The weights.
    mu : float64[:,:]
        The expected returns.
    sigma : float64[:,:,:]
        The covariance matrix.
    state_counts : int64[:,:]
        The state counts for each sequence.
    seq_length : int64
        The sequence length.
    rf : float64
        The risk free rate.

    Returns
    -------
    srs : float64[:,:]
        Sharpe ratios.
    """
    w = weights.reshape(-1, 1)

    exp_r = state_counts @ mu @ w / seq_length
    exp_sigma = (w.T @ np.einsum('ij,jkl->ikl', state_counts, sigma) @ w).reshape(-1, 1) / np.sqrt(seq_length)

    srs = (exp_r - rf) / exp_sigma

    return srs


def _max_avg_sharpe(weights, mu, sigma, state_counts, seq_length):
    srs = _calc_sharpes(weights, mu, sigma, state_counts, seq_length, rf=0)
    return -np.mean(srs)


def _calc_value_at_risk(weights, mu, sigma, state_counts):
    """
    Calculate the value at risk.

    Parameters
    ----------
    weights : float64[:]
        The weights.
    mu : float64[:,:]
        The expected returns.
    sigma : float64[:,:,:]
        The covariance matrix.
    state_counts : int64[:,:]
        The state counts for each sequence.

    Returns
    -------
    vars : float64[:,:]
        5% Value at Risk for each state sequence
    """
    w = weights.reshape(-1, 1)

    exp_r = state_counts @ mu @ w
    exp_sigma = (w.T @ np.einsum('ij,jkl->ikl', state_counts, sigma) @ w).reshape(-1, 1)
    exp_r_sigma = np.concatenate([exp_r, exp_sigma], axis=1)
    return np.apply_along_axis(
        lambda x: norm.ppf(0.05, x[0], x[1]), axis=1, arr=exp_r_sigma
    )


def _min_avg_VaR(weights, mu, sigma, state_counts):
    VaRs = _calc_value_at_risk(weights, mu, sigma, state_counts)
    return -np.mean(VaRs)


@numba.jit("int64[:,:](int64[:,:],int64)", nopython=True, nogil=True)
def _count_states(sequences, n_states):
    """
    Count the number of each state in each sequence.

    Parameters
    ----------
    sequences : int64[:,:]
        The sequences.
    n_states : int64
        The number of states.

    Returns
    -------
    state_counts : int64[:,:]
    """
    result = np.empty((len(sequences), n_states), dtype=np.int64)

    # np.unique returns sorted unique values, we only care about counts
    for i in range(len(sequences)):
        for s in range(n_states):
            result[i, s] = np.sum(sequences[i] == s)

    return result
