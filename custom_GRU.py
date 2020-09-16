import torch
from torch import nn


class GRU_cell_AI_SUMMER(torch.nn.Module):
    """
    A simple GRU cell network for educational purposes
    """

    def __init__(self, input_length=10, hidden_length=20):
        super(GRU_cell_AI_SUMMER, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # reset gate components
        self.linear_reset_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_reset_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)

        self.linear_reset_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_reset_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.activation_1 = nn.Sigmoid()

        # update gate components
        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.activation_2 = nn.Sigmoid()

        self.activation_3 = nn.Tanh()

    def reset_gate(self, x, h):
        x_1 = self.linear_reset_w1(x)
        h_1 = self.linear_reset_r1(h)
        # gate update
        reset = self.activation_1(x_1 + h_1)
        return reset

    def update_gate(self, x, h):
        x_2 = self.linear_reset_w2(x)
        h_2 = self.linear_reset_r2(h)
        z = self.activation_2(h_2 + x_2)
        return z

    def update_component(self, x, h, r):
        x_3 = self.linear_gate_w3(x)
        h_3 = r * self.linear_gate_r3(h)
        gate_update = self.activation_3(x_3 + h_3)
        return gate_update

    def forward(self, x, h):
        # Equation 1. reset gate vector
        r = self.reset_gate(x, h)

        # Equation 2: the update gate - the shared update gate vector z
        z = self.update_gate(x, h)

        # Equation 3: The almost output component
        n = self.update_component(x, h, r)

        # Equation 4: the new hidden state
        h_new = (1 - z) * n + z * h

        return h_new
