from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import pandas as pd
from marketmoodring.portfolio_optimization.base import PortfolioOptimizationBase, PortfolioOptimizationError
from marketmoodring.tools.portfolio_opt import ERC, MVO


class FactorPortfolioOptimization(PortfolioOptimizationBase, ABC):
    def __init__(self, n_regimes, optimizer="ERC", *args, **kwargs):
        """
        Initialize a factor portfolio optimizer

        Parameters
        ----------
        n_regimes : int
            The number of regimes used in the portfolio optimization
        optimizer : str
            The portfolio optimization algorithm to use. Options are "ERC" and "MVO"
        args : any
        kwargs : dict
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
        Calculate ERC portfolio weights using a regime-dependent factor model with static idiosyncratic risk

        Parameters
        ----------
        fitted_states : np.ndarray
            array with fitted state labels
        trans_mat : np.ndarray
            matrix with regime transition probabilities
        index_data : np.ndarray
            array with the time series returns of the assets to be included in the portfolio. This time-series has
            to match 1:1 with the fitted_states labels
        factor_data : np.ndarray
            array with the time series returns of the factors to be used for the regime-dependent factor model. This
            time-series has to match 1:1 with the fitted_states labels
        args : any
        kwargs : any

        Returns
        -------
        np.ndarray
            array with equal risk contribution (ERC) weights for the given assets using the simple factor regime model
        """
        mean, cov = self._get_regime_based_mean_cov(index_data, factor_data, trans_mat, fitted_states)

        return self._optimizer.calculate_weights(mean, cov)

    @abstractmethod
    def _get_regime_based_mean_cov(self, index_data, factor_data, trans_mat: Union[np.ndarray, pd.DataFrame],
                                   fitted_states):
        """
        Get the regime-dependent expected return vector and variance covariance matrix of the given assets.

        Parameters
        ----------
        index_data : A data frame with the time series asset returns
        factor_data : A data frame with the time series factor returns
        trans_mat :  The transition probability matrix of the regime switching model.
        fitted_states : The fitted states of the regime switching model.

        Returns
        -------
        (np.ndarray, np.ndarray)
            A tuple (mu, sigma) containing the expected return vector and covariance matrix of the assets.
        """
        pass
