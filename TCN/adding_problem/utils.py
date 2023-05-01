import torch
import numpy as np
from torch.autograd import Variable


def data_generator(N, seq_length):
    """
    Args:
        seq_length: Length of the adding problem data
        N: # of data in the set
    """
    X_num = torch.rand([N, 1, seq_length])
    X_mask = torch.zeros([N, 1, seq_length])
    Y = torch.zeros([N, 1])
    for i in range(N):
        positions = np.random.choice(seq_length, size=2, replace=False)
        X_mask[i, 0, positions[0]] = 1
        X_mask[i, 0, positions[1]] = 1
        Y[i, 0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
    X = torch.cat((X_num, X_mask), dim=1)
    return Variable(X), Variable(Y)


def time_weighted_power_loss(y, y_hat, loss=torch.nn.MSELoss(), p=2):
    """_summary_

    Args:
        y (_type_): true label
        y_hat (_type_): prediction
        loss (_type_, optional): loss function. Defaults to torch.nn.MSELoss().
        p (int, optional): supported [-1, 0, 1, 2, np.inf]. Defaults to 2.

    Returns:
        _type_: time weighted loss value,
                if p == 0, return the average value of original loss value
    """
    if not np.isinf(p):
        weighted_loss = 0
        length = y.shape[-2]

        scale = sum((i + 1) ** p for i in range(length))

        for i in range(length):
            weighted_loss += loss(y[:, i, :], y_hat[:, i, :]) * (i + 1) ** p / scale

        return weighted_loss
    else:
        weighted_loss = loss(y[:, -1, :], y_hat[:, -1, :])
        return weighted_loss
