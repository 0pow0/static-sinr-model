import torch
from torchinfo import summary
import math

def sinr_formula(d, num_RB, txPower):
    noise = -174.0 + 10.0 * math.log10(num_RB * 180000)
    # urban
    # path_loss = 28.0 + 22.0 * math.log10(d) + 20.0 * math.log10(2110e6 / 1e9)
    # rural
    path_loss = max(23.9, 1.8 * math.log10(0.01), 20.0) * math.log10(d) + 20.0 * math.log10(40.0 * math.pi * (2110e6 / 1e9) / 3.0)
    rx = txPower - path_loss - 9.0
    return rx - noise

class StaticSINRModel(torch.nn.Module):
    def __init__(self) -> None:
        super(StaticSINRModel, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.ff1 = torch.nn.Linear(5, 16)
        self.ff2 = torch.nn.Linear(16, 32)
        self.ff3 = torch.nn.Linear(32, 16)
        self.ff4 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu(self.ff1(x))
        x = self.relu(self.ff2(x))
        x = self.relu(self.ff3(x))
        x = self.ff4(x)
        return x

if __name__ == '__main__':
    model = StaticSINRModel()
    summary(model, input_size=(10, 5), device='cpu')
    # x = torch.tensor([[10.0, 1.0], [2.0, 1.0]])
    # y = model(x)

