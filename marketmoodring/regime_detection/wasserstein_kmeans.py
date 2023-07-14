import numpy as np
import scipy.stats as sts
from typing import Union
import pandas as pd

from marketmoodring.regime_detection.base import (NonParametricRegimeDetection)

MAX_DRAW_KMEANS_INIT = 100


class WassersteinKMeansRegimeDetection(NonParametricRegimeDetection):
    ot = None

    def __init__(self, n_regimes: int, frequency: str = "D", window_size: int = 5,
                 n_bins: int = 21, min_range: float = -0.15, max_range: float = 0.15,
                 p: int = 3, reg: float = 1e-3, max_iter: int = 200, kde_smoothing: float = 0.5) -> None:
        """
        A class to handle regime detection through Wasserstein K-Means on rolling return distributions.
        Currently only accepts 1D time series.

        Parameters
        ----------
        n_regimes : int
            The number of regimes to transform the data to
        frequency : str
            The frequency or unit on which to sample windows. For instance if frequency="D" and window_size=5, then
            windows will have a size of 5 days. Current options: "D" for daily
        window_size : int
            The window size of rolling return distributions
        n_bins : int
            The number of bins used for histograms in the WKM algorithm. A larger value results in more fine-grained
            but more jagged histograms
        min_range : float
            The minimum value of the histograms
        max_range : float
            The maximum value of the histograms
        p : int
            The p-Wasserstein distance to be used. p corresponds to the number of distribution moments that should
            be considered in the distance metric
        reg : float
            A regularization coefficient
        max_iter : int
            The maximum number of iterations of the WKM algorithm
        kde_smoothing : float
            A number between 0 (none) and 1 (full) indication the level of smoothing to be applied on the window
            distributions
        """
        super().__init__(n_regimes)
        if frequency not in ("D",):
            raise ValueError("Sampling frequency is invalid. Must be any of (\"D\",)")
        if window_size <= 0:
            raise ValueError("Window Size must be at least 1")
        if kde_smoothing < 0 or kde_smoothing > 1:
            raise ValueError("KDE smoothing must lie between 0 and 1")
        # Model hyperparameters
        self._frequency = frequency
        self._window_size = window_size
        self._n_bins = n_bins
        self._min_range = min_range
        self._max_range = max_range
        self._p = p
        self._reg = reg
        self._max_iter = max_iter
        self._kde_smoothing = kde_smoothing
        # Model fitting parameters
        self._barycenters = None
        self._m = None
        # Conditional import, only import if class is ever created
        if WassersteinKMeansRegimeDetection.ot is None:
            WassersteinKMeansRegimeDetection.ot = __import__('ot')

    def fit(self, data: Union[np.ndarray, pd.DataFrame], *args, **kwargs):
        """
        Fits the regime detection model to the given data. If data is provided as a pandas DataFrame with
        a resolution higher than daily, the data will be resampled to a daily resolution and the corresponding
        labels will apply to the days in the data.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            The data to fit the regime detection model to
        args : tuple, optional
        kwargs : dict, optional

        Returns
        -------
        None
        """
        super().fit(data, *args, **kwargs)

    def _fit(self, data: np.ndarray, index: np.ndarray = None, reuse_last_run: bool = True, *args, **kwargs):
        # Dependency only necessary if _fit of WKM is ever called
        sample_distributions = self._sample_distributions(data, index)

        # Fit and store WKM model
        self._barycenters, _ = self._wkmeans(sample_distributions, reuse_last_run)

        # Prepend fitted states with 0, due to the window size the first few observations
        # cannot be clustered
        self._fitted_states, _ = self._transform(data, index)

    def _transform(self, data: np.ndarray, index: np.ndarray = None, *args, **kwargs):
        sample_distributions = self._sample_distributions(data, index, jump=False)

        if self._m is None:
            x = np.arange(self._n_bins, dtype=np.float64)
            self._m = self._p_wass_dist(x)
        m = self._m

        distances = self.update_distances(self._barycenters, sample_distributions, m)
        labels = np.argmin(distances, axis=1)

        # Prepend fitted states with 0, due to the window size the first few observations
        # cannot be clustered
        if index is None:
            labels = np.concatenate((
                [np.nan] * (len(data) - len(labels)),
                labels
            ))
        if index is not None:
            index = index.astype('datetime64[D]')
            labels = np.concatenate((
                [np.nan] * (len(np.unique(index)) - len(labels)),
                labels
            ))

        return labels, None

    def _sample_distributions(self, data: np.ndarray, index: np.ndarray = None, jump=True):
        """
        Create sample distributions from the given 1D time series data and associated index

        Parameters
        ----------
        data : np.ndarray
            A 1D time series
        index : np.ndarray
            An optional 1D date time index corresponding to the time series
        jump : bool
            True if and only if a jump should be applied in between rolling windows

        Returns
        -------
        distributions : np.ndarray
            An array of sampled distributions
        """
        # Set up linspace of histogram bucket midpoints
        x = (
                    np.linspace(self._min_range, self._max_range, self._n_bins + 1, dtype=np.float64)[1:]
                    + np.linspace(self._min_range, self._max_range, self._n_bins + 1, dtype=np.float64)[:-1]
            ) / 2

        if index is None:
            # Sample distributions
            if jump:
                sample = np.array([
                    data[i:i + self._window_size]
                    # TODO: self._window_size // 4 replace by variable
                    for i in range(0, len(data) - self._window_size, self._window_size // 4)
                ])
            else:
                sample = np.lib.stride_tricks.sliding_window_view(data.flatten(), (self._window_size,))
        else:
            index = index.astype('datetime64[D]')
            sample = np.array([
                data[
                    (index >= np.busday_offset(i, -self._window_size, roll='backward'))
                    & (index <= i)
                ] for i in np.unique(index)[self._window_size:]
            ], dtype=object)

        # Create sample distributions
        dist_sample = np.zeros((len(sample), self._n_bins))
        for i in range(len(sample)):
            s = sample[i]
            kde = sts.gaussian_kde(s.reshape(1, -1)).pdf(x)
            hist = np.histogram(
                    s, bins=self._n_bins,
                    range=(self._min_range, self._max_range), density=False
                )[0]
            hist = hist / np.sum(hist)
            dist_sample[i, :] = self._kde_smoothing * kde + (1 - self._kde_smoothing) * hist
            dist_sample[i, :] = dist_sample[i, :] / np.sum(dist_sample[i, :])

        return dist_sample

    def _wkmeans(self, distributions, reuse_last_run):
        """
        Fit a WKM model on the given distributions

        Parameters
        ----------
        distributions : np.ndarray
            Historical return distributions
        reuse_last_run : bool
            True if and only if the last fitted model should be reused

        Returns
        -------
        tuple : (np.ndarray, np.ndarray)
            A tuple with the cluster barycenters and predicted labels for each distribution
        """
        if self._m is None:
            x = np.arange(self._n_bins, dtype=np.float64)
            self._m = self._p_wass_dist(x)
        m = self._m

        # Find appropriate initial points, try up to 100 random draws until one is found
        # where the distance is large enough to result in two distinct sets of distributions
        barycenters = None

        if self._barycenters is None or not reuse_last_run:
            for _ in range(MAX_DRAW_KMEANS_INIT):
                barycenters = distributions[np.random.choice(len(distributions), self.n_regimes, replace=False)]

                distances = self.update_distances(barycenters, distributions, m)
                labels = np.argmin(distances, axis=1)

                try:
                    self._update_barycenters(barycenters, distributions, labels, m)

                    break
                except ZeroDivisionError:
                    continue
        else:
            barycenters = self._barycenters

        # Iterate and update the barycenters until convergence or the maximum amount of iterations is reached
        it = 0
        labels = None

        while it < self._max_iter:
            distances = self.update_distances(barycenters, distributions, m)
            old_labels = labels
            labels = np.argmin(distances, axis=1)

            if np.array_equal(old_labels, labels):
                break

            self._update_barycenters(barycenters, distributions, labels, m)

            it += 1

        return barycenters, labels

    def update_distances(self, barycenters, distributions, m_b):
        """
        Calculate the distances from all given distributions to all given barycenters using the given loss matrix

        Parameters:
        ----------
        barycenters : numpy.ndarray
            The barycenters to calculate distances from.

        distributions : numpy.ndarray
            The distributions to calculate distances to.

        m_b : numpy.ndarray
            The loss matrix for the distances.

        Returns:
        -------
        numpy.ndarray
            A NxK numpy array of distances where N is the number of distributions and K is the number of regimes.
        """
        distances = np.zeros((len(distributions), self.n_regimes))

        for k in range(self.n_regimes):
            for d in range(len(distributions)):
                distances[d, k] = self.ot.emd2(distributions[d], barycenters[k], m_b)
        distances = np.power(distances, 1 / self._p)
        return distances

    def _update_barycenters(self, barycenters, distributions, labels, m_b):
        """
        Update the barycenters in place after a single WKM iteration.

        Parameters:
        ----------
        barycenters : numpy.ndarray
            The barycenters to update.

        distributions : numpy.ndarray
            The input distributions.

        labels : numpy.ndarray
            The classification labels of the current iteration.

        m_b : numpy.ndarray
            Loss matrix.
        """
        for k in range(self.n_regimes):
            k_dists = distributions[labels == k]
            weights = np.array([1 / len(k_dists)] * len(k_dists))

            barycenters[k] = self.ot.bregman.barycenter(
                np.vstack(k_dists).T,
                m_b,
                self._reg,
                weights,
                numItermax=20000
            )

            barycenters[k] /= np.sum(barycenters[k])

    def _p_wass_dist(self, x):
        """
        Construct the p-Wasserstein distance matrix for the given linspace x.

        Parameters:
        ----------
        x : numpy.ndarray
            A 1D linspace to construct a distance matrix on.

        Returns:
        -------
        numpy.ndarray
            An NxN distance matrix where N is the length of x.
        """
        m = self.ot.dist(x.reshape((self._n_bins, 1)), x.reshape((self._n_bins, 1)), metric="minkowski", p=self._p)
        m = m ** self._p
        m /= m.max()
        # m_b *= multiplier
        return m

    def __str__(self):
        return "WassersteinKMeans"
