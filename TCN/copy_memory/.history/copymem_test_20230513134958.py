import pandas as pd

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import sys

sys.path.append("../../")
from TCN.copy_memory.utils import data_generator, time_weighted_power_loss
from TCN.copy_memory.model import TCN
import time


parser = argparse.ArgumentParser(description="Sequence Modeling - Copying Memory Task")
parser.add_argument(
    "--batch_size", type=int, default=32, metavar="N", help="batch size (default: 32)"
)
parser.add_argument("--cuda", action="store_false", help="use CUDA (default: True)")
parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="dropout applied to layers (default: 0.0)",
)
parser.add_argument(
    "--clip",
    type=float,
    default=1.0,
    help="gradient clip, -1 means no clip (default: 1.0)",
)
parser.add_argument(
    "--epochs", type=int, default=50, help="upper epoch limit (default: 50)"
)
parser.add_argument("--ksize", type=int, default=8, help="kernel size (default: 8)")
parser.add_argument(
    "--iters", type=int, default=100, help="number of iters per epoch (default: 100)"
)
parser.add_argument("--levels", type=int, default=8, help="# of levels (default: 8)")
parser.add_argument(
    "--blank_len",
    type=int,
    default=1000,
    metavar="N",
    help="The size of the blank (i.e. T) (default: 1000)",
)
parser.add_argument(
    "--seq_len", type=int, default=10, help="initial history size (default: 10)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=50,
    metavar="N",
    help="report interval (default: 50",
)
parser.add_argument(
    "--lr", type=float, default=5e-4, help="initial learning rate (default: 5e-4)"
)
parser.add_argument(
    "--optim", type=str, default="RMSprop", help="optimizer to use (default: RMSprop)"
)
parser.add_argument(
    "--nhid",
    type=int,
    default=10,
    help="number of hidden units per layer (default: 10)",
)
parser.add_argument(
    "--seed", type=int, default=1111, help="random seed (default: 1111)"
)
parser.add_argument("--power", type=int, default=2, help="random seed (default: 2)")
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


batch_size = args.batch_size
seq_len = args.seq_len  # The size to memorize
epochs = args.epochs
iters = args.iters
T = args.blank_len
n_steps = T + (2 * seq_len)
n_classes = 10  # Digits 0 - 9
n_train = 10000
n_test = 1000

print(args)
print("Preparing data...")
train_x, train_y = data_generator(T, seq_len, n_train)
test_x, test_y = data_generator(T, seq_len, n_test)


channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout
model = TCN(1, n_classes, channel_sizes, kernel_size, dropout=dropout)

if args.cuda:
    model.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
weight = torch.ones((1, n_steps, 1))
if args.power <= 99:
    for i in range(1, n_steps):
        weight[0, i, 0] = (i / n_steps) ** args.power
else:
    for i in range(1, n_steps):
        weight[0, i, 0] = 1
weight /= torch.sum(weight)
weight = weight.cuda()


df = pd.DataFrame(
    np.zeros((epochs, 4), dtype=float),
    index=range(epochs),
    columns=["epoch", "loss", "accuracy", "grad_norm"],
)


def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(test_x.unsqueeze(1).contiguous())

        weighted_out = out * weight[:, : out.shape[1]]
        # loss = criterion(out.view(-1, n_classes), test_y.view(-1))
        loss = criterion(weighted_out.view(-1, n_classes), test_y.view(-1))

        pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
        correct = pred.eq(test_y.data.view_as(pred)).cpu().sum()
        counter = out.view(-1, n_classes).size(0)
        print(
            "\nTest set: Average loss: {:.8f}  |  Accuracy: {:.4f}\n".format(
                loss.item(), 100.0 * correct / counter
            )
        )

        return loss.item(), 100.0 * correct / counter


def train(ep):
    global batch_size, seq_len, iters, epochs
    model.train()
    total_loss = 0
    start_time = time.time()
    correct = 0
    counter = 0
    for batch_idx, batch in enumerate(range(0, n_train, batch_size)):
        start_ind = batch
        end_ind = start_ind + batch_size

        x = train_x[start_ind:end_ind]
        y = train_y[start_ind:end_ind]

        optimizer.zero_grad()
        out = model(x.unsqueeze(1).contiguous())

        # print(out.shape, y.shape)
        # exit()
        weighted_out = out * weight[:, : out.shape[1]]
        # loss = criterion(out.view(-1, n_classes), y.view(-1))
        loss = criterion(weighted_out.view(-1, n_classes), y.view(-1))

        pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        counter += out.view(-1, n_classes).size(0)
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time

            grad_norm = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** (1.0 / 2)

            print(
                "| Epoch {:3d} | {:4d}/{:4d} batches | lr {:2.5f} | ms/batch {:5.2f} | "
                "loss {:5.7f} | accuracy {:5.4f} | gradnorm {:5.4f}".format(
                    ep,
                    batch_idx,
                    n_train // batch_size + 1,
                    args.lr,
                    elapsed * 1000 / args.log_interval,
                    avg_loss,
                    100.0 * correct / counter,
                    grad_norm,
                )
            )

            start_time = time.time()
            total_loss = 0
            correct = 0
            counter = 0

            df["grad_norm"][ep - 1] = grad_norm


for ep in range(1, epochs + 1):
    train(ep)
    loss, accuracy = evaluate()
    df["epoch"][ep - 1] = ep
    df["loss"][ep - 1] = loss
    df["accuracy"][ep - 1] = accuracy.detach().cpu().numpy()
df.to_csv(f"./reproduce/results_{args.seq_len//10}_{args.power}_g.csv")
