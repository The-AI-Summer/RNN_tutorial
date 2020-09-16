# -*- coding: utf-8 -*-
'''
custom LSTM/RNN AI summer
Original file is located at
    https://colab.research.google.com/drive/1Rb8OiF-AZ_Y3uFj1O2S0IyocFMhHoTCV

AI Summer tutorial:Intuitive understanding of recurrent neural networks
This eductional LSTM tutorial heavily borrows from the Pytorch
example for time sequence prediction that can be found here:
https://github.com/pytorch/examples/tree/master/time_sequence_prediction
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn

from Sequence_model import Sequence
from generate_data import generate_data

if __name__ == '__main__':
    generate_data()
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    print(input.shape)
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    '''build the model. 
    LSTM=False means GRU cell
    custom=False uses the official Pytorch modules
    '''
    seq = Sequence(LSTM=True, custom=True)

    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # begin to train
    for i in range(20):
        print('STEP: ', i)


        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss


        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(14, 8))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)


        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)


        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.png' % i)
        plt.close()
