import numpy as np
import torch
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

LOOKBACK = 168
HORIZON = 24

# Ensure the 'data' directory exists and save the generated series
os.makedirs('data', exist_ok=True)
np.save("data/generated_series.npy", series)

series = np.load("data/generated_series.npy")

train = series[:-HORIZON]
test = series[-HORIZON:]

# ARIMA baseline
arima = ARIMA(train, order=(5,1,0))
arima_fit = arima.fit()
arima_forecast = arima_fit.forecast(steps=HORIZON)

rmse_arima = np.sqrt(mean_squared_error(test, arima_forecast))

# N-BEATS evaluation
# Ensure the 'results' directory exists and save the trained NBeats model
os.makedirs('results', exist_ok=True)
torch.save(model.state_dict(), "results/nbeats_model.pth") # 'model' is from the trained state

# Load the model for evaluation
eval_model = NBeats(LOOKBACK, HORIZON) # Create a new instance
eval_model.load_state_dict(torch.load("results/nbeats_model.pth"))
eval_model.eval()

input_window = torch.tensor(
    series[-(LOOKBACK+HORIZON):-HORIZON],
    dtype=torch.float32
).unsqueeze(0)

with torch.no_grad():
    nbeats_forecast = eval_model(input_window).numpy().flatten()

rmse_nbeats = np.sqrt(mean_squared_error(test, nbeats_forecast))

with open("results/metrics.txt", "w") as f:
    f.write(f"ARIMA RMSE: {rmse_arima:.4f}\n")
    f.write(f"N-BEATS RMSE: {rmse_nbeats:.4f}\n")

print("Evaluation complete. Metrics saved.")