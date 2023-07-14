from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import pandas as pd


class PortfolioOptimizationError(Exception):
    pass


class PortfolioOptimizationBase(ABC):
    """
    A base class for portfolio optimization.
    This class is not meant to be used directly, but rather to be inherited by other classes.
    The purpose of this class is to provide a common interface for all regime dependent optimization classes.

    The following methods must be implemented by any class that inherits this class:
    - calculate_weights
    """

    def __init__(self, n_regimes, *args, **kwargs):
        self.n_regimes = n_regimes

    @abstractmethod
    def calculate_weights(self, fitted_states: np.ndarray, trans_mat: np.ndarray,
                          index_data: Union[np.ndarray, pd.DataFrame], *args, **kwargs):
        """
        Abstract method to calculate weights.

        Parameters:
        ----------
        fitted_states : np.ndarray
            Array of fitted states.

        trans_mat : np.ndarray
            Transition matrix.

        index_data : Union[np.ndarray, pd.DataFrame]
            Index data used for weight calculation.

        *args, **kwargs :
            Additional arguments and keyword arguments.

        Returns:
        -------
        None
        """
        pass
