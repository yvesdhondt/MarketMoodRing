from typing import Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

from marketmoodring.portfolio_optimization.factor_model import FactorPortfolioOptimization


class IdiosyncraticFactorPortfolioOptimization(FactorPortfolioOptimization):
    def _get_regime_based_mean_cov(self, index_data, factor_data, trans_mat: Union[np.ndarray, pd.DataFrame],
                                   fitted_states):
        """
        Get the regime-dependent expected return vector and variance covariance matrix of the given assets according
        to the factor-based model proposed by Costa & Kwon (2020). Here idiosyncratic risk is assumed to be regime-
        dependant and factors are centered before being used.

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
        factor_names = factor_data.columns
        n_factors = len(factor_names)
        current_state = int(fitted_states[-1])

        Y = index_data.copy()
        X = factor_data.copy()
        # De-mean factors
        X = X - np.mean(X, axis=0)

        ols, state_factors = self._build_factor_model(X, Y, factor_names, fitted_states)

        for state in range(self.n_regimes):
            state_factors[state]["V"] = ols.params.values[state * (n_factors + 1): (1 + state) * (n_factors + 1) - 1, :]
            state_factors[state]["F"] = factor_data[fitted_states == state].cov().values
            state_factors[state]["mu"] = ols.params.values[(1 + state) * (n_factors + 1) - 1, :]
            state_factors[state]["D"] = ols.resid[fitted_states == state].cov().values

        # Construct regime-dependent expected return and variance-covariance matrices
        mu = 0
        sigma = 0
        for state in range(self.n_regimes):
            # update mu
            mu += trans_mat[current_state][state] * state_factors[state]["mu"]

            # update sigma
            sigma += trans_mat[current_state][state] \
                * (
                    state_factors[state]["V"].T @ state_factors[state]["F"] @ state_factors[state]["V"]
                    + state_factors[state]["D"]
                ) + trans_mat[current_state][state] * (1 - trans_mat[current_state][state]) \
                * state_factors[state]["mu"] @ state_factors[state]["mu"].T

            for other_state in range(self.n_regimes):
                if other_state == state:
                    continue
                sigma -= trans_mat[current_state][state] * trans_mat[current_state][other_state] \
                    * state_factors[state]["mu"] @ state_factors[state]["mu"].T

        return mu.reshape(-1, 1), sigma

    def _build_factor_model(self, X, Y, factor_names, fitted_states):
        """
        Build a regime-factor model of target asset returns (Y) to factors (X) for the given factor_names and
        fitted_states. Alpha is regime-dependent in this model.

        Parameters
        ----------
        X : np.ndarray
            matrix of factor returns
        Y : np.ndarray
            matrix of target asset returns
        factor_names : [string]
            names of the factors in the factor model
        fitted_states : np.ndarray
            array of regime labels

        Returns
        -------
        (ols, state_factors)
            A tuple consisting of an OLS model and a dictionary of state-factor information
        """
        # Transform factors by indicator function to allow for OLS estimation of regime-dependent FF3 model
        X["state"] = fitted_states
        state_factors = {}
        for state in range(self.n_regimes):
            state_factors[state] = {"names": []}
            for fn in factor_names:
                X[fn + "_" + str(state)] = X[[fn, "state"]].apply(lambda x: x[0] if x[1] == state else 0, axis=1)
                state_factors[state]["names"].append(fn + "_" + str(state))
            # Add regime-dependent constant
            X["mu_" + str(state)] = X["state"].apply(lambda x: 1 if x == state else 0)
            state_factors[state]["names"].append("mu_" + str(state))
        x_names = []
        for state in range(self.n_regimes):
            x_names += state_factors[state]["names"]

        X = X[x_names]
        # Fit regime-dependent Factor model
        ols = sm.OLS(Y, X).fit()
        return ols, state_factors

    def __str__(self):
        return "IdiosyncraticFactorOpt"
