# Advanced Time Series Forecasting with N-BEATS

This repository presents a complete, from-scratch implementation of the  
**N-BEATS (Neural Basis Expansion Analysis for Time Series)** architecture for
**univariate time series forecasting**, with a strong emphasis on:

- Predictive accuracy
- Proper time-series evaluation
- Model interpretability

The project is designed for **advanced students** and follows best practices
expected in academic and research-oriented machine learning work.

---

## Project Objectives

- Generate a complex, multi-seasonal univariate time series
- Implement the N-BEATS architecture using PyTorch
- Perform rigorous time-series cross-validation
- Compare against a strong classical baseline (ARIMA)
- Apply interpretability techniques (basis inspection and SHAP)
- Provide a fully reproducible, self-contained codebase

---

## Dataset

The dataset is **programmatically generated** to resemble real-world
high-frequency data (e.g., energy load or financial volatility).

### Characteristics
- Linear trend
- Short-term seasonality (e.g., daily)
- Long-term seasonality (e.g., yearly)
- Gaussian noise
- Non-stationary behavior

The dataset is generated programmatically during training and automatically saved as:

data/generated_series.npy

