import torch
import torch.nn as nn

from custom_GRU import GRU_cell_AI_SUMMER
from cutom_LSTM import LSTM_cell_AI_SUMMER


class Sequence(nn.Module):
    def __init__(self, LSTM=True, custom=True):
        super(Sequence, self).__init__()
        self.LSTM = LSTM

        if LSTM:
            if custom:
                print("AI summer LSTM cell implementation...")
                self.rnn1 = LSTM_cell_AI_SUMMER(1, 51)
                self.rnn2 = LSTM_cell_AI_SUMMER(51, 51)
            else:
                print("Official PyTorch LSTM cell implementation...")
                self.rnn1 = nn.LSTMCell(1, 51)
                self.rnn2 = nn.LSTMCell(51, 51)
        # GRU
        else:
            if custom:
                print("AI summer GRU cell implementation...")
                self.rnn1 = GRU_cell_AI_SUMMER(1, 51)
                self.rnn2 = GRU_cell_AI_SUMMER(51, 51)
            else:
                print("Official PyTorch GRU cell implementation...")
                self.rnn1 = nn.GRUCell(1, 51)
                self.rnn2 = nn.GRUCell(51, 51)

        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):

            if self.LSTM:
                h_t, c_t = self.rnn1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))
            else:
                h_t = self.rnn1(input_t, h_t)
                h_t2 = self.rnn2(h_t, h_t2)

            output = self.linear(h_t2)
            outputs += [output]

        # if we should predict the future
        for i in range(future):
            if self.LSTM:
                h_t, c_t = self.rnn1(input_t, (h_t, c_t))
                h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))
            else:
                h_t = self.rnn1(input_t, h_t)
                h_t2 = self.rnn2(h_t, h_t2)

            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
