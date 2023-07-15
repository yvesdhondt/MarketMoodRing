"""
API reference documentation for the base regime detection class
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from typing import Union

from marketmoodring.tools.data_checks import reconcile_regime_data_arg


class RegimeDetectionError(Exception):
    pass


class RegimeDetectionBase(ABC):
    """
    A base class for regime detection.
    This class is not meant to be used directly, but rather to be inherited by other classes.
    The purpose of this class is to provide a common interface for all regime detection classes.

    The following methods must be implemented by any class that inherits this class:
    - fit
    - transform

    The following method is optional to implement:
    - fit_transform
    """

    def __init__(self, n_regimes: Union[int, None], *args, **kwargs) -> None:
        """
        A class to handle regime detection.

        Parameters
        ----------
        n_regimes : int
            The number of regimes to transform the data to
        """
        if (n_regimes is not None) and n_regimes <= 0:
            raise ValueError("Number of regimes has to be larger than 0 or ")
        self.n_regimes = int(n_regimes) if n_regimes is not None else None
        self._fit_called = False
        self._fitted_states = None
        self._fitted_states_proba = None
        self._trans_mat = None
        self._model = None

    @abstractmethod
    def _fit(self, data: np.ndarray, index: np.ndarray = None, *args, **kwargs):
        """
        Abstract method to be implemented by any class that inherits this class.
        Checking arguments is done by the `fit` method that wraps this method.
        """
        pass

    def fit(self, data: Union[np.ndarray, pd.DataFrame], *args, **kwargs):
        """
        Fits the model to the data.
        Wraps the _fit method which is implemented by any class that inherits this class.

        Parameters
        ----------
        data : numpy Array or pandas DataFrame
            The data to fit the regimes on
        *args, **kwargs
            Any additional arguments to be passed to the _fit method

        Returns
        -------
        None
        """
        fit_data, fit_index = reconcile_regime_data_arg(data)
        self._fit(fit_data, fit_index, *args, **kwargs)
        self._fit_called = True

    @abstractmethod
    def _transform(self, data: np.ndarray, index: np.ndarray = None, *args, **kwargs):
        """
        Abstract method to be implemented by any class that inherits this class.
        Checking arguments is done by the `transform` method that wraps this method.
        """
        pass

    def transform(self, data: Union[np.ndarray, pd.DataFrame], *args, **kwargs):
        """
        Transforms the data using the fitted model.
        Wraps the _transform method which is implemented by any class that inherits this class.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            The data to transform using the fitted model
        *args, **kwargs
            Any additional arguments to be passed to the _transform method

        Returns
        -------
        tuple
            Two numpy arrays containing the (1) predicted states and (2) predicted
            state probabilities
        """

        transform_data, transform_index = reconcile_regime_data_arg(data)

        if not self._fit_called:
            raise RegimeDetectionError("fit must be called before transform")
        return self._transform(transform_data, transform_index, *args, **kwargs)

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame], *args, **kwargs):
        """
        Fit and transform the given data with a single function call. This function behaves the same
        as sequentially calling fit and transform.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            The data to fit and transform
        *args, **kwargs
            Any additional arguments to be passed to the fit_transform method

        Returns
        -------
        tuple
            Two numpy arrays containing the (1) predicted states and (2) predicted
            state probabilities
        """
        self.fit(data, *args, **kwargs)
        return self.transform(data, *args, **kwargs)

    def get_fitted_states(self):
        if not self._fit_called:
            raise RegimeDetectionError(
                "fit must be called before retrieving fitted states"
            )
        return self._fitted_states

    def get_fitted_states_proba(self):
        if not self._fit_called:
            raise RegimeDetectionError(
                "fit must be called before retrieving fitted states"
            )
        return self._fitted_states_proba

    def get_trans_mat(self):
        if not self._fit_called:
            raise RegimeDetectionError(
                "fit must be called before retrieving fitted transition matrix"
            )
        return self._trans_mat


class NonParametricRegimeDetection(RegimeDetectionBase, ABC):
    """
    A class to handle non-parametric regime detection
    i.e. clustering methods, and not Hidden Markov Models
    """

    def fit(self, data: Union[np.ndarray, pd.DataFrame], *args, **kwargs):
        super().fit(data, *args, **kwargs)
        self._fit_empirical_trans_matrix()

    def _fit_empirical_trans_matrix(self):
        """
        Fit an empirical transition matrix using the fitted labeled data.

        Returns
        -------
        trans_mat : np.ndarray
            An NxN matrix containing the transition matrix from each state to all other states
        """
        if self._fitted_states is None:
            raise RegimeDetectionError("Model has not been fitted yet or did not produce fitted states")

        transitions = pd.Series(self._fitted_states).to_frame("start_state")
        transitions["end_state"] = transitions["start_state"].shift(-1)

        # Get the transition matrix
        trans_mat = (
            transitions.value_counts(normalize=False)
            .reset_index()
            .pivot(index="start_state", columns="end_state")
            .fillna(0)
        )
        trans_mat += 1
        trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0)

        self._trans_mat = trans_mat.to_numpy()
