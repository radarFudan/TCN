import numpy as np
import torch
from torch.autograd import Variable


def data_generator(T, mem_length, b_size):
    """
    Generate data for the copying memory task

    :param T: The total blank time length
    :param mem_length: The length of the memory to be recalled
    :param b_size: The batch size
    :return: Input and target data tensor
    """
    seq = torch.from_numpy(np.random.randint(1, 9, size=(b_size, mem_length))).float()
    zeros = torch.zeros((b_size, T))
    marker = 9 * torch.ones((b_size, mem_length + 1))
    placeholders = torch.zeros((b_size, mem_length))

    x = torch.cat((seq, zeros[:, :-1], marker), 1)
    y = torch.cat((placeholders, zeros, seq), 1).long()

    x, y = Variable(x), Variable(y)
    return x, y


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
