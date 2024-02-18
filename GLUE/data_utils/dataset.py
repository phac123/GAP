from torch.utils.data import Dataset
import json
import pandas as pd
import csv

def load_file(file_name):
    data_pd = pd.read_csv(file_name, delimiter = '\t', header = None, encoding = 'utf-8', quoting = csv.QUOTE_NONE)
    data = data_pd.values
    data_list = []
    for data_item in data:
        data_list.append(data_item.item())

    return data_list

def load_file_json(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    return data

class MyDataset(Dataset):
    def __init__(self, dataset_type, input0, input1, label, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = list()
        self.data_type = dataset_type
        self.input0 = []; self.input1 = []; self.label = []

        vocab = tokenizer.vocab
        length = len(input0)
        if dataset_type == 'test':
            if input1 == None:
                for i in range(length):
                    self.input0.append(input0[i]); self.input1.append(0); self.label.append(0)
            else:
                for i in range(length):
                    self.input0.append(input0[i]); self.input1.append(input1[i]); self.label.append(0)
        else:
            if input1 == None:
                for i in range(length):
                    self.input0.append(input0[i]); self.input1.append(0); self.label.append(label[i])
            else:
                f = False
                for i in range(length):
                    f = False
                    if type(input1[i]) != str:
                        f = True
                    self.input0.append(input0[i]); self.input1.append('No Passage' if f == True else input1[i]); self.label.append(label[i])

    def __len__(self):
        return len(self.input0)

    def __getitem__(self, i):
        return self.input0[i], self.input1[i], self.label[i]

class Label2Logit:
    def __init__(self):
        self.cover = ['AX', 'MNLI-m', 'MNLI-mm', 'QNLI', 'RTE']
        self.AX_dict_la2i = {'entailment': 0, 'neutral': 1, 'contradiction': 2, '-': 1}
        self.AX_dict_i2la = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        self.MNLI_dict_la2i = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        self.MNLI_dict_i2la = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
        self.QNLI_dict_la2i = {'entailment': 0, 'not_entailment': 1}
        self.QNLI_dict_i2la = {0: 'entailment', 1: 'not_entailment'}
        self.RTE_dict_la2i = {'entailment': 0, 'not_entailment': 1}
        self.RTE_dict_i2la = {0: 'entailment', 1: 'not_entailment'}
