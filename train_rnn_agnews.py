import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import pandas as pd
from torchtext.data import Iterator, BucketIterator, TabularDataset
from torchtext import data
from torchtext.vocab import Vectors


class FastText(nn.Module):
    def __init__(self, vocab, vec_dim, label_size, hidden_size):
        super(FastText, self).__init__()
        #创建embedding
        self.embed = nn.Embedding(len(vocab), vec_dim)
        # 若使用预训练的词向量，需在此处指定预训练的权重
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True
        self.fc = nn.Sequential(
            nn.Linear(vec_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, label_size)
        )

    def forward(self, x):
        x = self.embed(x)
        out = self.fc(torch.mean(x, dim=1))
        return out

def train_model(net, train_iter, epoch, lr, batch_size):
    print("begin training")
    net.train()  # 必备，将模型设置为训练模式
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):  # 多批次循环
        for batch_idx, batch in enumerate(train_iter):
            # 注意target=batch.label - 1，因为数据集中的label是1，2，3，4，但是pytorch的label默认是从0开始，所以这里需要减1
            data, target = batch.text, batch.label - 1
            optimizer.zero_grad()  # 清除所有优化的梯度
            output = net(data)  # 传入数据并前向传播获取输出
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            logging.info(
                "train epoch=" + str(i) + ",batch_id=" + str(batch_idx) + ",loss=" + str(loss.item() / batch_size))
    print('Finished Training')


def model_test(net, test_iter):
    net.eval()  # 必备，将模型设置为训练模式
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            # 注意target=batch.label - 1，因为数据集中的label是1，2，3，4，但是pytorch的label默认是从0开始，所以这里需要减1
            data, label = batch.text, batch.label - 1
            logging.info("test batch_id=" + str(i))
            outputs = net(data)
            # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
            _, predicted = torch.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
            total += label.size(0)
            correct += (predicted == label).sum().item()
            print('Accuracy of the network on test set: %d %%' % (100 * correct / total))
            # test_acc += accuracy_score(torch.argmax(outputs.data, dim=1), label)
            # logging.info("test_acc=" + str(test_acc))


def get_data_iter(train_csv, test_csv, fix_length):
    TEXT = data.Field(sequential=True, lower=True, fix_length=fix_length, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    train_fields = [("label", LABEL), ("title", None), ("text", TEXT)]
    train = TabularDataset(path=train_csv, format="csv", fields=train_fields, skip_header=True)
    train_iter = BucketIterator(train, batch_size=batch_size, device=-1, sort_key=lambda x: len(x.text),
                                sort_within_batch=False, repeat=False)
    test_fields = [("label", LABEL), ("title", None), ("text", TEXT)]
    test = TabularDataset(path=test_csv, format="csv", fields=test_fields, skip_header=True)
    test_iter = Iterator(test, batch_size=batch_size, device=-1, sort=False, sort_within_batch=False, repeat=False)

    vectors = Vectors(name=word2vec_dir)
    TEXT.build_vocab(train, vectors=vectors)
    vocab = TEXT.vocab
    return train_iter, test_iter, vocab


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    train_csv = "/data/datasets/AG_NEWS/train.csv"
    test_csv = "/data/datasets/AG_NEWS/test.csv"
    word2vec_dir = "/data/datasets/AG_NEWS/glove.6B.300d.txt"  # 训练好的词向量文件,写成相对路径好像会报错
    sentence_max_size = 50  # 每篇文章的最大词数量
    batch_size = 64
    epoch = 10  # 迭代次数
    emb_dim = 300  # 词向量维度
    lr = 0.001
    hidden_size = 200
    label_size = 4

    train_iter, test_iter, vocab = get_data_iter(train_csv, test_csv, sentence_max_size)
    # 定义模型
    net = FastText(vocab=vocab, vec_dim=emb_dim, label_size=label_size, hidden_size=hidden_size)

    # 训练
    logging.info("开始训练模型")
    train_model(net, train_iter, epoch, lr, batch_size)
    # 保存模型

    logging.info("开始测试模型")
    model_test(net, test_iter)

