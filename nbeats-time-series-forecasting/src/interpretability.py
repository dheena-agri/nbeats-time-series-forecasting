import numpy as np
import torch
import shap

LOOKBACK = 168
HORIZON = 24

series = np.load("data/generated_series.npy")

model = NBeats(LOOKBACK, HORIZON)
model.load_state_dict(torch.load("results/nbeats_model.pth"))
model.eval()

def model_wrapper(x):
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return model(x).numpy()

background = series[:1000]
X_bg = np.array([
    background[i:i+LOOKBACK]
    for i in range(100)
])

explainer = shap.KernelExplainer(model_wrapper, X_bg)
sample = X_bg[0:1]
shap_values = explainer.shap_values(sample)

print("SHAP analysis completed.")