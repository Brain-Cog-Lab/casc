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
        x = self.Embedding(x)
        # x = self.dp(x)
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden = torch.zeros(batch_size, self.hid_size).to(x.device)
        output_t = torch.zeros_like(x)
        for t in range(seq_len):
            hidden = self.RNN(x[:, t, :], hidden)
            output_t[:, t, :] = hidden

        x = F.avg_pool2d(output_t, (output_t.shape[1], 1)).squeeze()
        x = self.fc1(x)
        return x


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
    MAX_LEN = 200
    BATCH_SIZE = 256
    EMB_SIZE = 300
    HID_SIZE = 300  # rnn隐藏层数量
    DROPOUT = 0.2
    device = torch.device('cuda:6')

    train_csv = "/data/datasets/AG_NEWS/train.csv"
    test_csv = "/data/datasets/AG_NEWS/test.csv"
    word2vec_dir = "/data/datasets/AG_NEWS/glove.6B.300d.txt"  # 训练好的词向量文件,写成相对路径好像会报错
    sentence_max_size = 50  # 每篇文章的最大词数量

    train_loader, test_loader, vocab = get_data_iter(train_csv, test_csv, sentence_max_size, BATCH_SIZE, word2vec_dir)

    model = Model(vocab, EMB_SIZE, HID_SIZE, DROPOUT).to(device)
    model = replace_relu_by_cqrelu(model, 2)

    # print(model)
    optimizer = optim.Adam(model.parameters())
    best_acc = 0.0

    for epoch in range(1, 50):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)

        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), '/data/ly/casc/rnn_model.pt')
        print("acc is:{:.4f},best acc is {:.4f}\n".format(acc, best_acc))

    model.load_state_dict(torch.load('/data/ly/casc/rnn_model.pt'))
    acc = test(model, device, test_loader)
    print("acc is:{:.4f}".format(acc))