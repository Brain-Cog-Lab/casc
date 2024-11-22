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
    def __init__(self, max_words, emb_size, hid_size, dropout):
        super(Model, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout

        self.Embedding = nn.Embedding(self.max_words, self.emb_size)
        self.RNN = MyRNN(self.emb_size, self.hid_size)
        self.fc1 = nn.Linear(self.hid_size, 2)


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
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = model(x)
        loss = loss_func(y_, y)
        loss.backward()
        optimizer.step()  # 这一步一定要有，用于更新参数，之前由于漏写了括号导致参数没有更新，排BUG排了半天

        if (step + 1) % 10 == 0:
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, step * len(x), len(train_loader.dataset),
                       100. * step / len(train_loader), loss.item()
            ))


def test(model, device, test_loader):
    model.eval()
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0.0
    acc = 0
    for step, (x, y) in enumerate(test_loader):
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


if __name__ == '__main__':
    MAX_WORDS = 10000
    MAX_LEN = 100
    BATCH_SIZE = 256
    EMB_SIZE = 300
    HID_SIZE = 300  # rnn隐藏层数量
    DROPOUT = 0.2
    device = torch.device('cuda:6')

    (x_train, y_train), (x_test, y_test) = imdb.load_data(path='/data/datasets/imdb', num_words=MAX_WORDS)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding='post', truncating='post')
    # 将数据转为tensorDataset
    train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
    # 将数据放入dataloader
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

    model = Model(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT).to(device)
    model = replace_relu_by_cqrelu(model, 2)

    print(model)
    optimizer = optim.Adam(model.parameters())
    best_acc = 0.0

    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)

        if best_acc < acc:
            best_acc = acc
            # torch.save(model.state_dict(), '/data/ly/casc/rnn_model.pt')
        print("acc is:{:.4f},best acc is {:.4f}\n".format(acc, best_acc))

    model.load_state_dict(torch.load('/data/ly/casc/rnn_model.pt'))
    acc = test(model, device, test_loader)
    print("acc is:{:.4f}".format(acc))