import torch

A = torch.tensor([[1.0, 0.0],
                  [1.0, 0.0]])

B = torch.tensor([[1.0],
                  [0.0]])

x0     = torch.tensor([1.0, 0.5])   # initial state
x_star = torch.tensor([0.0, 0.0])   # target state
T      = 1.0                         # control horizon
N_NODES = 2
