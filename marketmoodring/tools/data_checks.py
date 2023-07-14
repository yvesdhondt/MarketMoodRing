import numpy as np
import pandas as pd
from typing import Union


def reconcile_regime_data_arg(data: Union[np.ndarray, pd.DataFrame]) -> np.array:
    """
    Internal function to verify data types and transform into numpy arrays

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Data to check

    Returns
    -------
    (data, index) : (np.ndarray, np.ndarray)
        Reconciled data
    """
    if isinstance(data, np.ndarray):
        return data, None
    elif isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.DatetimeIndex):
            return data.to_numpy(), data.index.to_numpy()
        else:
            return data.to_numpy(), None
    else:
        raise ValueError("data must be a numpy Array or a pandas DataFrame")
