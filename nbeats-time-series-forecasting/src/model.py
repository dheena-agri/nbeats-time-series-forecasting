import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size, n_layers):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(input_size, hidden_size), nn.ReLU()]
            input_size = hidden_size
        self.fc = nn.Sequential(*layers)
        self.theta = nn.Linear(hidden_size, theta_size)

    def forward(self, x):
        return self.theta(self.fc(x))


class NBeats(nn.Module):
    def __init__(self, input_size, forecast_size,
                 hidden_size=256, n_layers=4, n_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size,
                        input_size + forecast_size,
                        hidden_size,
                        n_layers)
            for _ in range(n_blocks)
        ])
        self.input_size = input_size
        self.forecast_size = forecast_size

    def forward(self, x):
        residual = x
        forecast = torch.zeros(x.size(0), self.forecast_size).to(x.device)
        for block in self.blocks:
            theta = block(residual)
            backcast = theta[:, :self.input_size]
            forecast_part = theta[:, self.input_size:]
            residual = residual - backcast
            forecast += forecast_part
        return forecast
