import torch
from torch import nn
import torch.nn.functional as F
import argparse

from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

def get_agnews_data(root='/data/datasets', batch_size=64, length=50):

    tokenizer = get_tokenizer('basic_english')
    train_iter = AG_NEWS(root=root, split='train')

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    # max_len = 0
    # num = 0
    # for _, text in train_iter:
    #     # if len(tokenizer(text)) > max_len:
    #     #     max_len = len(tokenizer(text))
    #     max_len += len(tokenizer(text))
    #     num += 1
    # print(max_len/num)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    vocab_size = len(vocab)

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    length = 60

    def collate_batch(batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(padding(processed_text, length))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.stack(text_list).unsqueeze(1)
        return text_list, label_list


    def padding(tensor, length=50):
        if len(tensor) >= length:
            return tensor[:length]
        else:
            return F.pad(tensor, (0, length-len(tensor)))


    train_iter = AG_NEWS(split='train')
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)

    train_iter, test_iter = AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    training_iter = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_batch)

    test_iter = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=collate_batch)

    return training_iter, test_iter

