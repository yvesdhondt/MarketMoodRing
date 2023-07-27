import numpy as np
import pandas as pd
from typing import Union
import importlib

from marketmoodring.regime_detection.base import (RegimeDetectionBase,
                                                  RegimeDetectionError)


class HiddenMarkovRegimeDetection(RegimeDetectionBase):
    hmm = None

    def __init__(self, n_regimes: int, hmm_type: str = "GaussianHMM", n_iter: int = 50, covar_type: str = "diag",
                 *args, **kwargs) -> None:
        """
        A class to handle regime detection through Gaussian HMM.

        Parameters
        ----------
        n_regimes : int
            The number of regimes to transform the data to
        hmm_type : str
            The type of HMM model to use, options are "GaussianHMM" or "GMMHMM".
        n_iter : int
            The number of iterations to run the HMM model
        covar_type : str
            The type of covariance matrix to use, options are "spherical", "full", "diag" or "tied"
        """
        super().__init__(n_regimes, *args, **kwargs)

        if hmm_type not in ("GaussianHMM", "GMMHMM"):
            raise RegimeDetectionError("given type is not one of the available options")
        if n_iter < 0:
            raise RegimeDetectionError("n_iter has to be larger than 0")

        self._type = hmm_type
        self._n_iter = n_iter
        self._covar_type = covar_type
        # Conditional import, only import if class is ever created
        if HiddenMarkovRegimeDetection.hmm is None:
            HiddenMarkovRegimeDetection.hmm = importlib.import_module(
                "hmmlearn.hmm"
            )

    def _fit(self, data: np.ndarray, index: np.ndarray = None, *args, **kwargs):
        """
        Fit a Gaussian HMM or GMM HMM on the given data.

        Parameters
        ----------
        data : numpy.ndarray
            A matrix of time series data to fit the HMM on.
        index : numpy.ndarray, optional
            The time index of the time series data.
        *args : tuple
            Any additional positional arguments from the superclass, if any.
        **kwargs : dict
            Any additional keyword arguments from the superclass, if any.

        Returns
        -------
        None
            This method does not return anything, but it sets the following instance variables:

            _model : hmmlearn.hmm.GaussianHMM or hmmlearn.hmm.GMMHMM
                The fitted HMM model.
            _fitted_states : numpy.ndarray
                The fitted states of the HMM model.
            _fitted_states_proba : numpy.ndarray
                The fitted state probabilities of the HMM model.
            _trans_mat : numpy.ndarray
                The transition probability matrix of the HMM model.
        """
        # Create HMM Model
        if self._type == "GaussianHMM":
            self._model = self.hmm.GaussianHMM(
                n_components=self.n_regimes, n_iter=self._n_iter, covariance_type=self._covar_type, random_state=None
            )
        elif self._type == "GMMHMM":
            self._model = self.hmm.GMMHMM(
                n_components=self.n_regimes, n_iter=self._n_iter, covariance_type=self._covar_type, random_state=None
            )

        # Fit model
        self._model.fit(data)

        # Predict and store output of training data
        self._fitted_states = self._model.predict(data)
        self._fitted_states_proba = self._model.predict_proba(data)
        self._trans_mat = self._model.transmat_

    def _transform(self, data: np.ndarray, *args, **kwargs):
        if not self._fit_called:
            raise RegimeDetectionError("fit must be called before transforming")

        transformed_states = self._model.predict(data)
        transformed_states_proba = self._model.predict_proba(data)

        return transformed_states, transformed_states_proba

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame], *args, **kwargs):
        self.fit(data, *args, **kwargs)

        return self._fitted_states, self._fitted_states_proba

    def __str__(self):
        return self._type
