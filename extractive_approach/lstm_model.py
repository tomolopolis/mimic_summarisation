import json
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets import load_metric, load_from_disk

device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMClf(nn.Module):
    def __init__(self):
        super(LSTMClf, self).__init__()
        self.model = LSTM(384, 50, num_layers=2, batch_first=True, dropout=0.5, biderectional=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, X, X_lens):
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lens, batch_first=True)
        X, h = self.model(X)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        X = self.fc(X)
        return X


class MimicHospCourseDataset(Dataset):
    def __init__(self, split, pred_lim, size_lim=-1):
        self.pred_lim = pred_lim
        self.size_lim = size_lim
        ds = load_from_disk('/data/users/k1897038/mimic3_dataset_pre_processed')
        ds = ds.train_test_split(train_size=0.8, test_size=0.2, shuffle=False)
        if split == 'train':
            self.ds = ds['train'].sort('text_embed_len', reverse=True)
        elif split in ('val', 'test'):
            val_test_ds = ds['test'].train_test_split(train_size=0.5, test_size=0.5, shuffle=False)
            self.ds = val_test_ds['train'] if  split == 'val' else val_test_ds['test']
            self.ds = self.ds.sort('text_embed_len', reverse=True)
            self.ref_sum_sents = [''.join(d) for d in self.ds[f'summ_lim_{pred_lim}']]

    def __len__(self):
        if self.size_lim != -1:
            return self.size_lim
        return len(self.ds['text_embed_len'])


class MimicHospCourseTrainDataset(MimicHospCourseDataset):
    def __init__(self, pred_lim, size_lim=-1):
        super(MimicHospCourseTrainDataset, self).__init__('train', pred_lim, size_lim)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return torch.tensor(item['text_embed_limd']), \
               item['text_embed_len'], \
               torch.tensor(item[f'preds_lim_{self.pred_lim}'])


class MimicHospCourseValDataset(MimicHospCourseDataset):
    def __init__(self, pred_lim, size_lim=-1):
        super(MimicHospCourseValDataset, self).__init__('val', pred_lim, size_lim)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return torch.tensor(item['text_embed_limd']), \
               item['text_embed_len'], \
               torch.tensor(item[f'preds_lim_{self.pred_lim}']), \
               item['text_sents_limd'], \
               self.ref_sum_sents[idx]

class MimicHospCourseTestDataset(MimicHospCourseDataset):
    def __init__(self, pred_lim, size_lim=-1):
        super(MimicHospCourseTestDataset, self).__init__('test', pred_lim, size_lim)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return torch.tensor(item['text_embed_limd']), \
               item['text_embed_len'], \
               item['text_sents_limd'], \
               self.ref_sum_sents[idx]


def pad_train_sequence(batch):
    inputs, input_lens, outputs = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)
    return inputs, input_lens, outputs


def pad_val_sequence(batch):
    inputs, input_lens, outputs, input_sents, output_sents = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)
    return inputs, input_lens, outputs, input_sents, output_sents


def pad_test_sequence(batch):
    inputs, input_lens, input_sents, output_sents = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return inputs, input_lens, input_sents, output_sents