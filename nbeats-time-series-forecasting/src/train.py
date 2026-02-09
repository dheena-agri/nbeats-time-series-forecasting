import torch
import numpy as np

LOOKBACK = 168
HORIZON = 24

def create_windows(series):
    X, y = [], []
    for i in range(len(series) - LOOKBACK - HORIZON):
        X.append(series[i:i+LOOKBACK])
        y.append(series[i+LOOKBACK:i+LOOKBACK+HORIZON])
    # Convert lists to numpy arrays first, then to torch tensors
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

series = generate_time_series()
X, y = create_windows(series)

model = NBeats(LOOKBACK, HORIZON)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(20):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")