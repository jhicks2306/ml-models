import numpy as np
import torch.nn as nn

class neuralnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.Layer1 = nn.Linear(input_size, hidden_size)
        self.Layer2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.Layer1(x)
        x = nn.Sigmoid()(x)
        x = self.Layer2(x)
        return x




