###按照正常的转换方法去实现

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use('agg')
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse, logging, datetime, time

from src.utils import seed_all, accuracy, AverageMeter, result2csv, setup_default_logging, mergeConvBN
from src.clipquantization import replace_relu_by_cqrelu
from src.convertor import *
from src.dataset import *
from src.model import *
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import pandas as pd
from torchtext.data import Iterator, BucketIterator, TabularDataset
from torchtext import data
from torchtext.vocab import Vectors
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from src.clipquantization import replace_relu_by_cqrelu

from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from train_rnn import Model


parser = argparse.ArgumentParser(description='RNN')
parser.add_argument('--device', default='6', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--max_len', type=int,  default=200)
parser.add_argument('--train', default=False, type=bool)
parser.add_argument('-q', '--qlevel', type=int,  default=4)
# parser.add_argument('--neg', action='store_true')

parser.add_argument('--neg', type=bool,  default=True)
parser.add_argument('--sleep', type=int,  default=12)
parser.add_argument('--margin', type=int,  default=4)
parser.add_argument('--T', type=int,  default=4)
parser.add_argument('--batch_size', type=int,  default=256)
parser.add_argument('--epochs', type=int,  default=7)
parser.add_argument('--emb_size', type=int,  default=300)
parser.add_argument('--hid_size', type=int,  default=300)


args = parser.parse_args()


def train(model, device, train_loader, optimizer, epoch):
    model.train()  # 将模型设置为训练模式
    loss_func = nn.CrossEntropyLoss()
    for step, batch in enumerate(train_loader):
        x, y = batch.text, batch.label - 1
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = model(x)
        loss = loss_func(y_, y)
        loss.backward()
        optimizer.step()  # 这一步一定要有，用于更新参数，之前由于漏写了括号导致参数没有更新，排BUG排了半天

        if (step + 1) % 100 == 0:
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, step * len(x), len(train_loader.dataset),
                       100. * step / len(train_loader), loss.item()
            ))



class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Linear(input_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(200)])

    def forward(self, x, h, t=0):
        h = self.relus[t](self.Wxh(x) + self.Whh(h))
        return h


class Model(nn.Module):
    def __init__(self, vocab, emb_size, hid_size, dropout):
        super(Model, self).__init__()
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout

        self.Embedding = nn.Embedding(len(vocab), self.emb_size)
        self.Embedding.weight.data.copy_(vocab.vectors)
        self.Embedding.weight.requires_grad = True
        self.RNN = MyRNN(self.emb_size, self.hid_size)
        self.fc1 = nn.Linear(self.hid_size, 4)


    def forward(self, x):
        x = self.Embedding(x) if not (x!=0).sum() == 0 else torch.zeros_like(self.Embedding(x)).to(x.device)
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden = torch.zeros(batch_size, self.hid_size).to(x.device)
        output_t = torch.zeros_like(x)
        for t in range(seq_len):
            hidden = self.RNN(x[:, t, :], hidden, t=t)
            output_t[:, t, :] = hidden

        x = F.avg_pool2d(output_t, (output_t.shape[1], 1)).squeeze()
        x = self.fc1(x)
        return x


def test(model, device, test_loader):
    model.eval()
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0.0
    acc = 0
    for step, batch in enumerate(test_loader):
        x, y = batch.text, batch.label - 1
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(x)
        test_loss += loss_func(y_, y)
        pred = y_.max(-1, keepdim=True)[1]
        acc += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set:Average loss:{:.4f},Accuracy:{}/{} ({:.0f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100 * acc / len(test_loader.dataset)
    ))

    return acc / len(test_loader.dataset)


def close_bias(model):
    children = list(model.named_children())
    for i, (name, child) in enumerate(children):
        if isinstance(child, (nn.Conv2d, nn.Linear)) and hasattr(child, 'bias'):
            child.bias.data *= 0
        else:
            close_bias(child)


def evaluate_snn(net, device, data_iter, T, sleep, margin):
    net = net.to(device)
    acc_tot = AverageMeter()
    with torch.no_grad():
        dict1 = deepcopy(net).state_dict()
        net2 = deepcopy(net)
        close_bias(net2)
        dict2 = net2.state_dict()

        start = time.time()
        for step, batch in enumerate(data_iter):
            X, y = batch.text, batch.label - 1
            # print("eval %d/%d ...:" % (ind, len(data)), end='\r')
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            net.eval()
            output, acc_t = 0, []
            net.reset()
            for t in range(T):
                output += net(X.to(device)).detach()
                acc, = accuracy(output / (t + 1), y.to(device), topk=(1,))
                acc_t += [acc.detach().cpu().item()]

                if (t + 1) % margin == 0 and sleep != 0:
                    net.change_sleep()
                    net.load_state_dict(dict2)
                    for ti in range(sleep):
                        output += net(torch.zeros_like(X).to(device)).detach()
                        acc, = accuracy(output / (t + 1), y.to(device), topk=(1,))
                        acc_t += [acc.detach().cpu().item()]
                    net.change_sleep()
                    net.load_state_dict(dict1)

            net.train()
            acc_tot.update(np.array(acc_t), output.shape[0])

    print('<Test>   acc:%.6f, time:%.1f s' % (acc_tot.avg[-1], time.time() - start))
    # print(acc_tot.avg)
    return acc_tot.avg


def get_data_iter(train_csv, test_csv, fix_length, batch_size, word2vec_dir):
    TEXT = data.Field(sequential=True, lower=True, fix_length=fix_length, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train_fields = [("label", LABEL), ("title", None), ("text", TEXT)]
    train = TabularDataset(path=train_csv, format="csv", fields=train_fields, skip_header=True)
    train_iter = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                sort_within_batch=False, repeat=False)
    test_fields = [("label", LABEL), ("title", None), ("text", TEXT)]
    test = TabularDataset(path=test_csv, format="csv", fields=test_fields, skip_header=True)
    test_iter = Iterator(test, batch_size=batch_size, sort=False, sort_within_batch=False, repeat=False)

    vectors = Vectors(name=word2vec_dir)
    TEXT.build_vocab(train, vectors=vectors)
    vocab = TEXT.vocab
    return train_iter, test_iter, vocab


if __name__ == '__main__':
    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    EMB_SIZE = args.emb_size
    HID_SIZE = args.hid_size
    DROPOUT = 0.2
    device = torch.device("cuda:%s" % args.device)

    cqrelu = True
    qlevel = args.qlevel

    neg = args.neg
    sleep = args.sleep
    margin = args.margin
    T = args.T

    train_csv = "/data/datasets/AG_NEWS/train.csv"
    test_csv = "/data/datasets/AG_NEWS/test.csv"
    word2vec_dir = "/data/datasets/AG_NEWS/glove.6B.300d.txt"  # 训练好的词向量文件,写成相对路径好像会报错
    sentence_max_size = 50  # 每篇文章的最大词数量

    train_loader, test_loader, vocab = get_data_iter(train_csv, test_csv, sentence_max_size, BATCH_SIZE, word2vec_dir)

    model = Model(vocab, EMB_SIZE, HID_SIZE, DROPOUT).to(device)
    model = replace_relu_by_cqrelu(model, qlevel)

    ########### training ############

    if args.train:
        # print(model)
        optimizer = optim.Adam(model.parameters())
        best_acc = 0.0

        for epoch in range(args.epochs):
            train(model, device, train_loader, optimizer, epoch)
            acc = test(model, device, test_loader)

            if best_acc < acc:
                best_acc = acc
                torch.save(model.state_dict(), '/data/ly/casc/rnn_model_{}_{}.pt'.format(qlevel, MAX_LEN))
            print("acc is:{:.4f},best acc is {:.4f}\n".format(acc, best_acc))


    # ########## testing ############
    else:
        model.load_state_dict(torch.load('/data/ly/casc/rnn_model_{}_{}.pt'.format(qlevel, MAX_LEN)))
        acc = test(model, device, test_loader)
        print("acc is:{:.4f}".format(acc))

        if cqrelu:
            converter = CQConvertor(
                soft_mode=True,
                lipool=True,
                gamma=1,
                pseudo_convert=False,
                merge=True,
                neg=neg,
                sleep_time=[qlevel, qlevel + sleep])
        else:
            converter = PercentConvertor(
                dataloader=train_loader,
                device=device,
                p=0.999,
                channelnorm=False,
                soft_mode=True,
                lipool=True,
                gamma=1,
                pseudo_convert=False,
                merge=True,
                neg=neg)

        snn = converter(deepcopy(model))
        acc = evaluate_snn(snn, device, test_loader, T, sleep, margin)
        for i in range(len(acc)):
            print("time: %d, acc: %.4f" % (i + 1, acc[i]))
