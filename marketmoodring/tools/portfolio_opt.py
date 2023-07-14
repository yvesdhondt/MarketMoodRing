import numpy as np
from scipy.optimize import minimize
from abc import ABC, abstractmethod


class PortfolioOptimizer(ABC):
    @abstractmethod
    def calculate_weights(self, mu, cov):
        """
        Calculates the weights for the optimization.

        Parameters
        ----------
        mu : np.ndarray
            The mean of the distribution.
        cov : np.ndarray
            The covariance of the distribution.

        Returns
        -------
        weights : np.ndarray
            The weights for the optimization.
        """
        pass


class ERC(PortfolioOptimizer):
    def __init__(self):
        pass

    def calculate_weights(self, mu, cov):
        """
        Calculate ERC portfolio weights

        Parameters
        ----------
        mu : np.ndarray
            Mean vector
        sigma : np.ndarray
            Covariance matrix

        Returns
        -------
        np.ndarray
            array with equal risk contribution (ERC) weights for the given assets
        """
        # Set up optimization
        num_assets = cov.shape[0]
        w_guess = np.repeat(1 / num_assets, num_assets)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0, 1) for asset in range(num_assets))

        # Optimize
        result = minimize(self._erc_objective, w_guess, args=(cov), method='SLSQP', constraints=constraints,
                          bounds=bounds)

        # Return weights
        return result.x

    def _calculate_var(self, w: np.ndarray, sigma: np.ndarray) -> float:
        """
        Calculate the variance of the portfolio with the given weights, w, and asset covariances, sigma

        Parameters
        ----------
        w : np.ndarray
            array of asset weights
        sigma : np.ndarray
            matrix of asset covariances

        Returns
        -------
        float
            estimated variance of the portoflio
        """
        # Calculate portfolio variance
        return w.T @ sigma @ w

    def _calculate_mctr(self, w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate the marginal contribution to risk (MCTR) of each portfolio asset for the given weights, w, and
        asset covariances, sigma

        Parameters
        ----------
        w : np.ndarray
            array of asset weights
        sigma : np.ndarray
            matrix of asset covariances

        Returns
        -------
        np.ndarray
            array of MCTR for each asset
        """
        # Calculate risk contribution of each asset
        portfolio_var = self._calculate_var(w, sigma)
        return w * (sigma @ w) / portfolio_var

    def _erc_objective(self, w: np.ndarray, sigma: np.ndarray) -> float:
        """
        Calculate the objective function for an equal risk contribution portfolio (ERC)

        Parameters
        ----------
        w : np.ndarray
            array of asset weights
        sigma : np.ndarray
            matrix of asset covariances

        Returns
        -------
        float
            MSE of the asset MCTRs vs the average MCTR
        """
        # Objective function to minimize
        mctr = self._calculate_mctr(w, sigma)
        return np.sum((mctr - mctr.mean()) ** 2)


class MVO(PortfolioOptimizer):
    def __init__(self):
        pass

    def calculate_weights(self, mu, sigma, rf=0):
        """
        Calculate ERC portfolio weights

        Parameters
        ----------
        mu : np.ndarray
            Mean vector
        sigma : np.ndarray
            Covariance matrix

        Returns
        -------
        np.ndarray
            array with equal risk contribution (ERC) weights for the given assets
        """
        """
        ones = np.ones(sigma.shape[0]).reshape(-1, 1)

        return (inv(sigma) @ (mu - rf * ones) / (
            ones.T @ inv(sigma) @ (mu - rf * ones)
        )).reshape(-1,)
        """
        # Set up optimization
        num_assets = sigma.shape[0]
        w_guess = np.repeat(1 / num_assets, num_assets)

        args = (mu, sigma, rf)
        constraints = ({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1
        })
        result = minimize(
            fun=self.mvo_objective,
            x0=w_guess,
            args=args,
            method='SLSQP',
            bounds=tuple((0.0, 1.0) for _ in range(num_assets)),
            constraints=constraints,
            tol=1e-5
        )

        if result['success']:
            return (result['x'] / np.sum(result['x'])).reshape(-1,)
        else:
            return None

    def mvo_objective(self, weights, mu, sigma, rf=0):
        """
        Objective function to minimize to find the MVO portfolio weights

        Parameters
        ----------
        weights : np.ndarray
            array of asset weights
        mu : np.ndarray
            Mean vector
        sigma : np.ndarray
            Covariance matrix
        rf : float
            Risk free rate

        Returns
        -------
        negative Sharpe Ratio for the given weights
        """
        # Return negative Sharpe ratio
        weights = weights.reshape(-1, 1)
        return - (weights.T @ mu - rf) / np.sqrt(weights.T @ sigma @ weights)
