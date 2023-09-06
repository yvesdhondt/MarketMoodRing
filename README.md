# MarketMoodRing🎭

## Description

MarketMoodRing🎭 is a Python package designed for testing different regime detection models and portfolio optimizers. This tool is a product of research conducted by the UC Berkeley, Haas School of Business, Master of Financial Engineering, 2023. It aims to provide a framework for financial market regime analysis and portfolio management testing.

## Collaborators
- [Yves D'hondt](https://github.com/yvesdhondt)
- [Matteo Di Venti](https://github.com/MatteoMarioDiVenti)
- [Rohan Rishi](https://github.com/RohanRishi)
- [Jackson Walker](https://github.com/jacksonrgwalker/])

## Reference

The `\reference` folder contains the [project paper](reference/project_paper.pdf) from where this library originated.

## Features

- **Regime Detection Models**: Currently includes Hidden Markov Models (HMM) and Wasserstein K-Means clustering (WKM).
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

from marketmoodring.regime_detection import HiddenMarkovRegimeDetection
```

Please replace `/path/to/MarketMoodRing` with the actual path to the cloned repository on your system.

Remember to keep the repository updated with:

```bash
git pull origin main
```

## Dependencies

The MarketMoodRing package requires several dependencies to function properly. These dependencies can be installed using either conda (recommended) or pip.

### Using Conda

If you're using Conda, you can create a new environment and install all dependencies using the `environment.yml` file located in the root directory. Run the following command in your terminal / Anaconda Prompt once you've navigated to the cloned repository:

```bash
conda env create -f environment.yml
```

This will create a new Conda environment called `marketmoodring-env` and install all necessary packages. To activate the environment, use:

```bash
conda activate marketmoodring-env
```

### Using pip

If you prefer using pip, you can install all dependencies using the `requirements.txt` file also located in the root directory. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages listed in the `requirements.txt` file. Please note that this will install the packages globally on your system, which can result in unexpected behavior. If you want to install the packages in a virtual environment, please refer to the [Python documentation](https://docs.python.org/3/tutorial/venv.html), or use the Conda environment as described above.

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
