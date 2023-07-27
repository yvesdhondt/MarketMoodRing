from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import pandas as pd
from marketmoodring.portfolio_optimization.base import PortfolioOptimizationBase, PortfolioOptimizationError
from marketmoodring.tools.portfolio_opt import ERC, MVO


class FactorPortfolioOptimization(PortfolioOptimizationBase, ABC):
    def __init__(self, n_regimes, optimizer="ERC", *args, **kwargs):
        """
        Initialize a factor portfolio optimizer.

        Parameters
        ----------
        n_regimes : int
            The number of regimes used in the portfolio optimization.
        optimizer : str
            The portfolio optimization algorithm to use. Options are "ERC" and "MVO".
        *args : tuple
            Any additional positional arguments from the superclass, if any.
        **kwargs : dict
            Any additional keyword arguments from the superclass, if any.

        Raises
        ------
        ValueError
            If the given optimizer is not one of "ERC" or "MVO".
        """
        super().__init__(n_regimes, *args, **kwargs)
        if optimizer == "ERC":
            self._optimizer = ERC()
        elif optimizer == "MVO":
            self._optimizer = MVO()
        else:
            raise ValueError("Given optimizer is not one of ERC or MVO")

    def calculate_weights(self, fitted_states: np.ndarray, trans_mat: np.ndarray,
                          index_data: Union[np.ndarray, pd.DataFrame],
                          factor_data: Union[np.ndarray, pd.DataFrame] = None, *args, **kwargs):
        """
        Calculate equal risk contribution (ERC) portfolio weights using a regime-dependent factor model with static
        idiosyncratic risk.

        Parameters
        ----------
        fitted_states : numpy.ndarray
            An array of fitted state labels.
        trans_mat : numpy.ndarray
            A matrix of regime transition probabilities.
        index_data : numpy.ndarray or pandas.DataFrame
            A matrix of time series returns of the assets to be included in the portfolio. This time-series has to
            match 1:1 with the fitted_states labels.
        factor_data : numpy.ndarray or pandas.DataFrame, optional
            A matrix of time series returns of the factors to be used for the regime-dependent factor model. This
            time-series has to match 1:1 with the fitted_states labels.
        *args : tuple
            Any additional positional arguments from the superclass, if any.
        **kwargs : dict
            Any additional keyword arguments from the superclass, if any.

        Returns
        -------
        numpy.ndarray
            An array of equal risk contribution (ERC) portfolio weights for the given assets using the simple factor
            regime model.
        """
        mean, cov = self._get_regime_based_mean_cov(index_data, factor_data, trans_mat, fitted_states)

        return self._optimizer.calculate_weights(mean, cov)

    @abstractmethod
    def _get_regime_based_mean_cov(self, index_data, factor_data, trans_mat: Union[np.ndarray, pd.DataFrame],
                                   fitted_states):
        """
        Get the regime-dependent expected return vector and variance covariance matrix of the given assets according
        to the factor-based model proposed by Costa & Kwon (2020).

        Parameters
        ----------
        index_data : numpy.ndarray or pandas.DataFrame
            A matrix of time series asset returns.
        factor_data : numpy.ndarray or pandas.DataFrame
            A matrix of time series factor returns.
        trans_mat : numpy.ndarray or pandas.DataFrame
            The transition probability matrix of the regime switching model.
        fitted_states : numpy.ndarray
            The fitted states of the regime switching model.

        Returns
        -------
        tuple
            A tuple (mu, sigma) containing the expected return vector and covariance matrix of the assets.

            mu : numpy.ndarray
                The regime-dependent expected return vector of the assets.
            sigma : numpy.ndarray
                The regime-dependent covariance matrix of the assets.
        """
        pass
