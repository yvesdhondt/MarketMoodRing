from marketmoodring.portfolio_optimization.factor_model import FactorPortfolioOptimization
import pandas as pd
from statsmodels import api as sm
from typing import Union
import numpy as np


class SimpleFactorPortfolioOptimization(FactorPortfolioOptimization):

    def _get_regime_based_mean_cov(self, index_data, factor_data, trans_mat: Union[np.ndarray, pd.DataFrame],
                                   fitted_states):
        """
        Get the regime-dependent expected return vector and variance covariance matrix of the given assets according
        to the factor-based model proposed by Costa & Kwon (2019). Here idiosyncratic risk is assumed to be independent
        of regime shifts.

        Parameters
        ----------
        index_data : pandas.DataFrame
            A DataFrame with the time series asset returns.
        factor_data : pandas.DataFrame
            A DataFrame with the time series factor returns.
        trans_mat : Union[numpy.ndarray, pandas.DataFrame]
            The transition probability matrix of the regime switching model.
        fitted_states : numpy.ndarray[numpy.int64]
            The fitted states of the regime switching model.

        Returns
        -------
        Tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]
            A tuple (mu, sigma) containing the expected return vector and covariance matrix of the assets.
        """
        factor_names = factor_data.columns
        n_factors = len(factor_names)
        current_state = int(fitted_states[-1])

        Y = index_data.copy()
        X = factor_data.copy()

        # Transform factors by indicator function to allow for OLS estimation of regime-dependent FF3 model
        X["state"] = fitted_states
        state_factors = {}
        for state in range(self.n_regimes):
            state_factors[state] = {"names": []}
            for fn in factor_names:
                X[fn + "_" + str(state)] = X[[fn, "state"]].apply(lambda x: x[0] if x[1] == state else 0, axis=1)
                state_factors[state]["names"].append(fn + "_" + str(state))

        x_names = []
        for state in range(self.n_regimes):
            x_names += state_factors[state]["names"]

        X = X[x_names]

        # Fit regime-dependent Factor model
        ols = sm.OLS(Y, sm.add_constant(X)).fit()

        # Define parameters
        alpha = ols.params.values[0:1, :].reshape(-1, 1)

        for state in range(self.n_regimes):
            state_factors[state]["V"] = ols.params.values[1 + state * n_factors: 1 + (1 + state) * n_factors, :]
            state_factors[state]["F"] = factor_data[fitted_states == state].cov().values
            state_factors[state]["f_bar"] = factor_data[fitted_states == state].mean().values.reshape(-1, 1)

        D = ols.resid.cov().values

        # Construct regime-dependent expected return and variance-covariance matrices
        mu = alpha
        sigma = D
        for state in range(self.n_regimes):
            # update mu
            mu += trans_mat[current_state][state] * state_factors[state]["V"].T @ state_factors[state]["f_bar"]

            # update sigma
            sigma += trans_mat[current_state][state] \
                     * state_factors[state]["V"].T @ state_factors[state]["F"] @ state_factors[state]["V"] \
                     + trans_mat[current_state][state] * (1 - trans_mat[current_state][state]) \
                     * state_factors[state]["V"].T @ state_factors[state]["f_bar"] @ state_factors[state]["f_bar"].T \
                     @ state_factors[state]["V"]
            for other_state in range(self.n_regimes):
                if other_state == state:
                    continue
                sigma -= trans_mat[current_state][state] * trans_mat[current_state][other_state] \
                         * state_factors[state]["V"].T @ state_factors[state]["f_bar"] \
                         @ state_factors[other_state]["f_bar"].T @ state_factors[other_state]["V"]

        return mu.reshape(-1, 1), sigma

    def __str__(self):
        return "SimpleFactorOpt"

