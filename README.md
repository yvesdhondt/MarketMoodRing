# MarketMoodRing

## Description

MarketMoodRing is a Python package designed for testing different regime detection models and portfolio optimizers. This tool is a product of research conducted by the UC Berkeley, Haas School of Business, Master of Financial Engineering, 2023. It aims to provide a framework for financial market regime analysis and portfolio management testing.

## Collaborators
- [Yves D'hondt](https://github.com/yvesdhondt)
- [Matteo Di Venti](https://github.com/MatteoMarioDiVenti)
- [Rohan Rishi](https://github.com/RohanRishi)
- [Jackson Walker](https://github.com/jacksonrgwalker/])


## Features

- **Regime Detection Models**: Includes various models such as Hidden Markov Models (HMM), Wasserstein K-Means, and others.
- **Portfolio Optimizers**: Implements different portfolio optimization strategies, including stochastic programming and factor-based optimization.


## Installation

As of now, the MarketMoodRing package is not available on PyPI. To install and use this package, you need to clone the repository and reference it locally. Here are the steps to do so:

1. Clone the repository:

```bash
git clone https://github.com/yvesdhondt/MarketMoodRing.git
```

2. Navigate to the cloned directory:

```bash
cd MarketMoodRing
```

3. Now, you can import and use the package in your Python scripts. Make sure your script is in the same directory as the cloned repository or adjust the Python path accordingly.

```python
import sys
sys.path.insert(0, '/path/to/MarketMoodRing')

from marketmoodring import regime_detection, portfolio_optimization
```

Please replace `/path/to/MarketMoodRing` with the actual path to the cloned repository on your system.

Remember to keep the repository updated with:

```bash
git pull origin main
```

## Usage

```python
from marketmoodring.regime_detection import HiddenMarkovRegimeDetection
from marketmoodring.portfolio_optimization import JointStochasticProgOptimization

# Read in your data
index_data = pd.read_csv('path/to/index_data.csv')

# Initialize regime detection model
hmm_model = HiddenMarkovRegimeDetection(n_regimes=2, hmm_type='GMMHMM', covar_type="diag", n_iter=100)

# Fit the model to your data and predict regimes
fitted_states, fitted_states_proba = regime_model.fit_transform(index_data)

# Initialize portfolio optimizer
opt_model = JointStochasticProgOptimization(n_regimes=2, objective="max_avg_sharpe")

# Fit the optimizer to your data and regime predictions and calculate portfolio weights
weights = opt_model.calculate_weights(
                    fitted_states = fitted_states,
                    trans_mat = hmm_model.get_trans_mat()
                    index_data = index_data,
                )
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.