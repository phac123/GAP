import json
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import pdb

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer
from os.path import join, abspath, dirname

from data_utils.dataset import load_file, MyDataset_two, MyDataset_one, MyDataset_four, Label2Logit
from p_tuning.modeling import PTuneForGlue

import gc

from sklearn.metrics import matthews_corrcoef

import warnings
warnings.filterwarnings("ignore")

SUPPORT_MODELS = ['gpt2-m', 'gpt2-l']
evaluate_support = ["Accuracy", "MCC", "Gender_Parity", 'F1-score']

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def construct_generation_args():
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    parser.add_argument("--relation_id", type=str, default="RTE")
    parser.add_argument("--model_name", type=str, default='gpt2-m')
    parser.add_argument("--tokenizer_src", type=str, default='gpt2-m')
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--template", type=str, default="(3, 3, 3)")
    parser.add_argument("--early_stop", type=int, default=450)

    parser.add_argument("--p_lr", type=float, default=4e-4)
    parser.add_argument("--a_lr", type=float, default=3e-5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--use_lm_finetune", type=bool, default=False)
    parser.add_argument("--use_original_template", type=bool, default=False)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    parser.add_argument("--evaluate_type", type=str, default="Accuracy")

    parser.add_argument("--use_dev", type=bool, default=True)
    parser.add_argument("--use_test", type=bool, default=False)

    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--num_class", type=int, default=2)

    parser.add_argument("--data_dir", type=str, default=join(abspath(dirname(__file__)), 'dataset'))
    parser.add_argument("--out_dir", type=str, default="/raid/phac123/Model_Output/GPT2-M-EKF-AdaLoRA_SuperGLUE/A_P/out")

    parser.add_argument("--checkpoint_dir", type=str, default=join(abspath(dirname(__file__)), 'checkpoints'))

    parser.add_argument("--cuda_device", type=str, default='cuda:2')

    parser.add_argument("--a_epochs", type=int, default=15)
    parser.add_argument("--p_epochs", type=int, default=25)
    parser.add_argument("--gan_epochs", type=int, default=3)

    args = parser.parse_args()

    args.device = torch.device(args.cuda_device if torch.cuda.is_available() else 'cpu')
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)

    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(self.args.cuda_device if torch.cuda.is_available() else 'cpu')

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.tokenizer_src)
        
        if self.args.relation_id  == 'WSC':
            self.relation, self.data_path_pre, self.data_path_post = self.get_TREx_parameters()
            self.train_input0 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'train_raw_input0' + self.data_path_post))
            self.train_label = load_file(
                join(self.args.data_dir, self.data_path_pre + 'train_label' + self.data_path_post))
            self.dev_input0 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'dev_raw_input0' + self.data_path_post))
            self.dev_label = load_file(
                join(self.args.data_dir, self.data_path_pre + 'dev_label' + self.data_path_post))
            self.test_input0 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'test_raw_input0' + self.data_path_post))
        elif self.args.relation_id == 'COPA':
            self.relation, self.data_path_pre, self.data_path_post = self.get_TREx_parameters()
            self.train_input0 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'train_raw_input0' + self.data_path_post))
            self.train_input1 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'train_raw_input1' + self.data_path_post))
            self.train_input2 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'train_raw_input2' + self.data_path_post))
            self.train_input3 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'train_raw_input3' + self.data_path_post))
            self.train_label = load_file(
                join(self.args.data_dir, self.data_path_pre + 'train_label' + self.data_path_post))
            self.dev_input0 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'dev_raw_input0' + self.data_path_post))
            self.dev_input1 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'dev_raw_input1' + self.data_path_post))
            self.dev_input2 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'dev_raw_input2' + self.data_path_post))
            self.dev_input3 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'dev_raw_input3' + self.data_path_post))
            self.dev_label = load_file(
                join(self.args.data_dir, self.data_path_pre + 'dev_label' + self.data_path_post))
            self.test_input0 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'test_raw_input0' + self.data_path_post))
            self.test_input1 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'test_raw_input1' + self.data_path_post))
            self.test_input2 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'test_raw_input2' + self.data_path_post))
            self.test_input3 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'test_raw_input3' + self.data_path_post))
        else:
            self.relation, self.data_path_pre, self.data_path_post = self.get_TREx_parameters()
            self.train_input0 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'train_raw_input0' + self.data_path_post))
            self.train_input1 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'train_raw_input1' + self.data_path_post))
            self.train_label = load_file(
                join(self.args.data_dir, self.data_path_pre + 'train_label' + self.data_path_post))
            self.dev_input0 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'dev_raw_input0' + self.data_path_post))
            self.dev_input1 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'dev_raw_input1' + self.data_path_post))
            self.dev_label = load_file(
                join(self.args.data_dir, self.data_path_pre + 'dev_label' + self.data_path_post))
            self.test_input0 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'test_raw_input0' + self.data_path_post))
            self.test_input1 = load_file(
                join(self.args.data_dir, self.data_path_pre + 'test_raw_input1' + self.data_path_post))

        tmp_la2lo = Label2Logit()
        self.train_label_logit, self.dev_label_logit = [], []
        if self.args.relation_id == 'AX-b' or self.args.relation_id == 'AX-g':
            for i in range(len(self.train_label)):
                self.train_label_logit.append(tmp_la2lo.AX_dict_la2i[self.train_label[i]])
            for i in range(len(self.dev_label)):
                self.dev_label_logit.append(tmp_la2lo.AX_dict_la2i[self.dev_label[i]])
        elif self.args.relation_id == 'BoolQ':
            for i in range(len(self.train_label)):
                self.train_label_logit.append(tmp_la2lo.BoolQ_dict_la2i[str(self.train_label[i])])
            for i in range(len(self.dev_label)):
                self.dev_label_logit.append(tmp_la2lo.BoolQ_dict_la2i[str(self.dev_label[i])])
        elif self.args.relation_id == 'CB':
            for i in range(len(self.train_label)):
                self.train_label_logit.append(tmp_la2lo.CB_dict_la2i[self.train_label[i]])
            for i in range(len(self.dev_label)):
                self.dev_label_logit.append(tmp_la2lo.CB_dict_la2i[self.dev_label[i]])
        elif self.args.relation_id == 'RTE':
            for i in range(len(self.train_label)):
                self.train_label_logit.append(tmp_la2lo.RTE_dict_la2i[self.train_label[i]])
            for i in range(len(self.dev_label)):
                self.dev_label_logit.append(tmp_la2lo.RTE_dict_la2i[self.dev_label[i]])
        elif self.args.relation_id == 'WiC':
            for i in range(len(self.train_label)):
                self.train_label_logit.append(tmp_la2lo.WiC_dict_la2i[str(self.train_label[i])])
            for i in range(len(self.dev_label)):
                self.dev_label_logit.append(tmp_la2lo.WiC_dict_la2i[str(self.dev_label[i])])
        elif self.args.relation_id == 'WSC':
            for i in range(len(self.train_label)):
                self.train_label_logit.append(tmp_la2lo.WSC_dict_la2i[str(self.train_label[i])])
            for i in range(len(self.dev_label)):
                self.dev_label_logit.append(tmp_la2lo.WSC_dict_la2i[str(self.dev_label[i])])
        else:     
            for i in range(len(self.train_label)):
                self.train_label_logit.append(self.train_label[i])
            for i in range(len(self.dev_label)):
                self.dev_label_logit.append(self.dev_label[i])

        if self.args.relation_id == 'WSC':
            self.train_set = MyDataset_one('Train', self.train_input0, self.train_label_logit, self.args)
            self.dev_set = MyDataset_one('Dev', self.dev_input0, self.dev_label_logit, self.args)
            self.test_set = MyDataset_one('Test', self.test_input0, -1, self.args)
        elif self.args.relation_id == 'COPA':
            self.train_set = MyDataset_four('Train', self.train_input0, self.train_input1, self.train_input2, self.train_input3, self.train_label_logit, self.args)
            self.dev_set = MyDataset_four('Dev', self.dev_input0, self.dev_input1, self.dev_input2, self.dev_input3, self.dev_label_logit, self.args)
            self.test_set = MyDataset_four('Test', self.test_input0, self.test_input1, self.test_input2, self.test_input3, -1, self.args)
        else:
            self.train_set = MyDataset_two('Train', self.train_input0, self.train_input1, self.train_label_logit, self.args)
            self.dev_set = MyDataset_two('Dev', self.dev_input0, self.dev_input1, self.dev_label_logit, self.args)
            self.test_set = MyDataset_two('Test', self.test_input0, self.test_input1, -1, self.args)

        self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.dev_loader = DataLoader(self.dev_set, batch_size=self.args.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.batch_size, shuffle=False)

        self.model= PTuneForGlue(args, self.device, self.args.template)
        self.model = self.model.to(self.device)

    def get_TREx_parameters(self):
        relation = self.args.relation_id
        data_path_pre = "{}/".format(self.args.relation_id)
        data_path_post = '.tsv'

        return relation, data_path_pre, data_path_post

    def get_task_name(self):
        names = [self.args.model_name,
                 "template_{}".format(self.args.template if not self.args.use_original_template else 'original'),
                 "fixed" if not self.args.use_lm_finetune else "fine-tuned",
                 "seed_{}".format(self.args.seed)]

        return "_".join(names)  

    def get_save_path(self):
        return join(self.args.out_dir, 'prompt_model', self.args.model_name, self.get_task_name(), self.args.relation_id)

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt['partial_state_dict'], join(path, ckpt_name))
        print(join(path, ckpt_name))
        print("# {} Checkpoint {} saved.".format(self.args.relation_id, ckpt_name))

        return join(path, ckpt_name)

    def check_model_update(self, model1, model2):
        for name, param in model1.named_parameters():
            if torch.equal(param, model2.state_dict()[name]):
                print(f"Parameter_mode1:'{name}' and model2has not been updated.")
            else:
                print(f"Parameter '{name}' has been updated.")

    def get_checkpoint(self, outer_epoch, epoch_idx, dev_score, dev_loss, test_loss):
        if self.args.relation_id in ['AX-b', 'AX-g', 'BoolQ', 'CB', 'COPA', 'RTE', 'WiC', 'WSC']:
            ckpt_name = "outerEpoch_{}_epoch_{}_dev_score_{}_dev_loss_{}_test_loss_{}.ckpt".format(outer_epoch, epoch_idx, round(dev_score.item() * 100, 4), round(dev_loss, 4), round(test_loss, 4))
        else:
            ckpt_name = "outerEpoch_{}_epoch_{}_dev_score_{}_dev_loss_{}_test_loss_{}.ckpt".format(outer_epoch, epoch_idx, round(dev_score * 100, 4), round(dev_loss, 4), round(test_loss, 4))
        return {
            'partial_state_dict': {
                **self.model.prompt_encoder.state_dict(),
                **self.model.fc.state_dict(),
                **self.model.transition_function.state_dict(),
                **{k: v for k, v in self.model.model.state_dict().items() if 'lora' in k}
            },
            'dev_score': dev_score,
            'dev_loss': dev_loss,
            'test_loss': test_loss,
            'test_size': len(self.test_set),
            'ckpt_name': ckpt_name,
            'time': datetime.now(),
            'args': self.args
        }

    def get_accuracy(self, preds, labels):
        preds = np.argmax(preds, axis = 1).flatten()
        labels = labels
        labels = labels.flatten()
        acc = (preds == labels).sum() / preds.shape[0]

        return acc

    def evaluate(self, epoch_idx, evaluate_type):
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader
            dataset = self.test_set
        else:
            loader = self.dev_loader
            dataset = self.dev_set

        with torch.no_grad():
            self.model.eval()
            tot_loss, num_of_samples, score= 0, 0, []
            for batch in loader:
                if self.args.relation_id == 'WSC':
                    input0 = batch[0]; label = batch[1]
                    preds_y = self.model(input0, None, None, None, label)
                elif self.args.relation_id == 'COPA':
                    input0 = batch[0]; label = batch[4]
                    preds_y = self.model(batch[0], batch[1], batch[2], batch[3], label)
                else:
                    input0 = batch[0]; label = batch[2]
                    preds_y = self.model(batch[0], batch[1], None, None, label)
                preds_y = preds_y.squeeze()
                preds_y = preds_y.to('cpu')
                label = torch.round(label).long()
                tot_loss += criterion(preds_y.float(), label).item()
                num_of_samples += len(input0)
                if evaluate_type == 'Test':
                    score = None
                else:
                    if self.args.evaluate_type == 'Accuracy':
                        score.append(self.get_accuracy(preds_y, label))
                    elif self.args.evaluate_type == 'MCC':
                        preds_y = np.argmax(preds_y, axis = 1).flatten()
                        label = label.flatten()
                        score.append(matthews_corrcoef(label, preds_y))
                    else:
                        preds_y = np.argmax(preds_y, axis = 1).flatten()
                        label = label.flatten()
                        score.append(f1_score(preds_y, label, average = 'macro'))

            return tot_loss/num_of_samples, sum(score) / len(score) if evaluate_type == 'Dev' else None

    def gan_train(self):
        best_dev_score, best_model_save_path, best_ckpt = -10000, None, None
        for i in range(self.args.gan_epochs):
            if i > 0:
                print("best_dev_score:{}, best_ckpt's name:{}.".format(best_dev_score, best_ckpt['ckpt_name']))
                print("best_model_save_path: {}".format(best_model_save_path))
                partial_state_dict = torch.load(best_model_save_path)
                self.model.prompt_encoder.load_state_dict(partial_state_dict, strict = False)
                self.model.fc.load_state_dict(partial_state_dict, strict = False)
                self.model.transition_function.load_state_dict(partial_state_dict, strict = False)
                self.model.model.load_state_dict(partial_state_dict, strict = False)
                self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
            unfreeze_layers = ['lora']
            for name, param in self.model.model.named_parameters():
                for ele in unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break
            self.model.to(self.device)
            if i == 0:
                self.args.use_original_template = True
                self.model.init_weights()
            self.args.lr = self.args.a_lr
            self.model.print_trainable_parameters(self.model)
            permt_best_ckpt, permt_best_dev_score, permt_model_path = self.train('Train: A_P_A', i, self.args.a_epochs, best_dev_score, best_ckpt)
            if permt_best_dev_score > best_dev_score:
                best_dev_score = permt_best_dev_score
                best_model_save_path = permt_model_path
                best_ckpt = permt_best_ckpt

            partial_state_dict = torch.load(best_model_save_path)
            self.model.fc.load_state_dict(partial_state_dict, strict = False)
            self.model.model.load_state_dict(partial_state_dict, strict = False)
            self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = True
            for param in self.model.transition_function.parameters():
                param.requires_grad = True
            self.model.to(self.device)

            if i == 0:
                self.args.use_original_template = False
                self.model.init_weights()

            self.args.lr = self.args.p_lr
            self.model.print_trainable_parameters(self.model)
            permt_best_ckpt, permt_best_dev_score, permt_model_path = self.train('Train: A_P_P', i, self.args.p_epochs, best_dev_score, best_ckpt)
            if permt_best_dev_score > best_dev_score:
                best_dev_score = permt_best_dev_score
                best_model_save_path = permt_model_path
                best_ckpt = permt_best_ckpt

        return best_ckpt

    def train(self, train_type, outer_epoch, epochs, best_dev_score, best_ckpt):
        early_stop = 0
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.args.lr, weight_decay = self.args.weight_decay, eps=1e-8)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = self.args.decay_rate)
        criterion = nn.CrossEntropyLoss()
        print("Train begining!!!!!!!!!!")
        for epoch_idx in range(epochs):
            num_of_samples = 0
            tot_loss = 0
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                self.model.train()
                optimizer.zero_grad()

                if self.args.relation_id == 'WSC': 
                    predict_y = self.model(batch[0], None, None, None, batch[1])
                    predict_y = predict_y.squeeze()
                    bz_loss = criterion(predict_y.to(self.device), batch[1].to(self.device))
                elif self.args.relation_id == 'COPA':
                    predict_y = self.model(batch[0], batch[1], batch[2], batch[3], batch[4])
                    predict_y = predict_y.squeeze()
                    bz_loss = criterion(predict_y.to(self.device), batch[4].to(self.device))
                else:
                    predict_y = self.model(batch[0], batch[1], None, None, batch[2])
                    predict_y = predict_y.squeeze()
                    bz_loss = criterion(predict_y.to(self.device), batch[2].to(self.device))
                tot_loss += bz_loss.item()
                num_of_samples += len(batch[0])
                bz_loss.backward()
                optimizer.step()
            my_lr_scheduler.step()
            dev_loss, dev_score = self.evaluate(epoch_idx, 'Dev')
            print("train_type: {}{}. Epoch: {}. dev_loss: {}. dev_score: {}".format(train_type, outer_epoch, epoch_idx, dev_loss, dev_score))
            if dev_score > best_dev_score:
                print("Evaluate Beginging!!!!!!")
                best_ckpt = self.get_checkpoint(outer_epoch, epoch_idx, dev_score, dev_loss, 0)
                early_stop = 0
                best_dev_score = dev_score

                self.save(best_ckpt)
            else:
                early_stop += 1
                if early_stop >= self.args.early_stop:
                    print("{} Early stopping at epoch {}.".format(self.args.relation_id, epoch_idx))
                    print("best_ckpt'name: {}. best_dev_score: {}.".format(best_ckpt['ckpt_name'], best_dev_score))

                    return best_ckpt, best_dev_score, join(self.get_save_path(), best_ckpt['ckpt_name'])
        
        return best_ckpt, best_dev_score, join(self.get_save_path(), best_ckpt['ckpt_name'])

def main(relation_id=None):
    args = construct_generation_args()
    if relation_id:
        args.relation_id = relation_id
    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    print("relation_id: {}".format(args.relation_id))
    print("model_name: {}".format(args.model_name))
    trainer = Trainer(args)
    trainer.gan_train()

if __name__ == '__main__':
    main()

