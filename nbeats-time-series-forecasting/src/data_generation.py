
import numpy as np

def generate_time_series(n_steps=10000, seed=42):
    np.random.seed(seed)
    t = np.arange(n_steps)

    trend = 0.0005 * t
    seasonality_1 = 2 * np.sin(2 * np.pi * t / 24)
    seasonality_2 = 5 * np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(scale=0.5, size=n_steps)

    return trend + seasonality_1 + seasonality_2 + noise
