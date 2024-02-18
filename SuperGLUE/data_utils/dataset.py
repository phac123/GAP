from torch.utils.data import Dataset
import json
import pandas as pd
import csv

# read file; tsv -> list
def load_file(file_name):
    data_pd = pd.read_csv(file_name, delimiter = '\t', header = None, encoding = 'utf-8', quoting = csv.QUOTE_NONE)
    data = data_pd.values
    data_list = []
    for data_item in data:
        data_list.append(data_item.item())

    return data_list

class MyDataset_two(Dataset):
    def __init__(self, dataset_type, input0, input1, label, args):
        super().__init__()
        self.args = args
        self.data = list()
        self.dataset_type = dataset_type
        self.input0 = []; self.input1 = []; self.label = []

        length = len(input0)

        flag = [False, False]
        for i in range(length):
            flag = [False, False]
            if type(input0[i]) != str:
                flag[0] = True
            if type(input1[i]) != str:
                flag[1] = True

            self.input0.append('No Passage' if flag[0] == True else input0[i])
            self.input1.append('No Passage' if flag[1] == True else input1[i])

            if self.dataset_type == 'Test':
                self.label.append(-1)
            else:
                self.label.append(label[i])

    def __len__(self):
        return len(self.input0)

    def __getitem__(self, index):
        return self.input0[index], self.input1[index], self.label[index]

class MyDataset_one(Dataset):
    def __init__(self, dataset_type, input0, label, args):
        super().__init__()
        self.args = args
        self.data = list()
        self.dataset_type = dataset_type
        self.input0 = []; self.label = []

        length = len(input0)

        flag = [False]
        for i in range(length):
            flag = [False]
            if type(input0[i]) != str:
                flag[0] = True

            self.input0.append('No Passage' if flag[0] == True else input0[i])

            if self.dataset_type == 'Test':
                self.label.append(-1)
            else:
                self.label.append(label[i])

    def __len__(self):
        return len(self.input0)

    def __getitem__(self, index):
        return self.input0[index], self.label[index]

class MyDataset_four(Dataset):
    def __init__(self, dataset_type, input0, input1, input2, input3, label, args):
        super().__init__()
        self.args = args
        self.data = list()
        self.dataset_type = dataset_type
        self.input0 = []; self.input1 = []; self.input2 = []; self.input3 = []; self.label = []

        length = len(input0)

        flag = [False, False, False, False]
        for i in range(length):
            flag = [False, False, False, False]
            if type(input0[i]) != str:
                flag[0] = True
            if type(input1[i]) != str:
                flag[1] = True
            if type(input2[i]) != str:
                flag[2] = True
            if type(input3[i]) != str:
                flag[3] = True

            self.input0.append('No Passage' if flag[0] == True else input0[i])
            self.input1.append('No Passage' if flag[1] == True else input1[i])
            self.input2.append('No Passage' if flag[2] == True else input2[i])
            self.input3.append('No Passage' if flag[3] == True else input3[i])

            if self.dataset_type == 'Test':
                self.label.append('-1')
            else:
                self.label.append(label[i])

    def __len__(self):
        return len(self.input0)

    def __getitem__(self, index):
        return self.input0[index], self.input1[index], self.input2[index], self.input3[index], self.label[index]

class Label2Logit:
    def __init__(self):
        self.cover = ['AX-b', 'AX-g', 'BoolQ', 'CB', 'RTE', 'WiC', 'WSC']
        self.AX_dict_la2i = {'not_entailment': 0, 'entailment': 1}
        self.AX_dict_i2la = {0: 'not_entailment', 1: 'entailment'}
        self.BoolQ_dict_la2i = {'False': 0, 'True': 1}
        self.BoolQ_dict_i2la = {0: 'False', 1: 'True'}
        self.CB_dict_la2i = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        self.CB_dict_i2la = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
        self.RTE_dict_la2i = {'not_entailment': 0, 'entailment': 1}
        self.RTE_dict_i2la = {0: 'not_entailment', 1: 'entailment'}
        self.WiC_dict_la2i = {'False': 0, 'True': 1}
        self.WiC_dict_i2la = {0: 'False', 1: 'True'}
        self.WSC_dict_la2i = {'False': 0, 'True': 1}
        self.WSC_dict_i2la = {0: 'False', 1: 'True'}
