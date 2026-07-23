"""Verify that the initial machine-learning environment works."""

import matplotlib
import numpy as np
import pandas as pd
import sklearn


def main() -> None:
    """Print installed package versions and run a small array test."""
    values = np.array([1.0, 2.0, 3.0, 4.0])

    print("Environment setup successful.")
    print(f"NumPy version: {np.__version__}")
    print(f"pandas version: {pd.__version__}")
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Test array mean: {values.mean()}")


if __name__ == "__main__":
    main()