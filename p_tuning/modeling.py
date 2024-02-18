import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join

import re

import math
import torch.nn.init as init

from transformers import AutoTokenizer
from data_utils.dataset import load_file_json
from p_tuning.models import create_model, get_embedding_layer
from transformers import BertForSequenceClassification
from p_tuning.prompt_encoder import PromptEncoder
from Classification.NN import ClassifierDecoder
from p_tuning.TransitionNN import transition_function_transformer

from peft import LoraConfig, get_peft_model

class PTuneForGlue(torch.nn.Module):
    def __init__(self, args, device, template):
        super().__init__()
        self.args = args
        self.device = device

        self.relation_templates = dict(
            (d['relation'], d['template']) for d in load_file_json(join(self.args.data_dir, 'relations.jsonl')))

        tokenizer_src = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
        
        self.model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=self.args.num_class)
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
        self.embeddings = get_embedding_layer(self.args, self.model)
        self.Lora_Config = config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS" 
        )
        self.model = get_peft_model(self.model, self.Lora_Config)
        self.vocab = self.tokenizer.get_vocab()
        self.template = template

        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device, self.args)
        self.prompt_encoder = self.prompt_encoder.to(self.device)

        self.transition_function = transition_function_transformer(self.hidden_size, 0.1)

    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def get_query(self, x_input0, x_input1, prompt_tokens, y = None):
        if type(x_input1) != str and x_input1 != 0:
            if math.isnan(x_input1):
                x_input1 = 'No passage'
        if self.args.use_original_template:
            if x_input1 == 0:
                query = ['[CLS]' + x_input0 + '[SEP]']
                return self.tokenizer(query)['input_ids']
            else:
                query = ['[CLS]' + x_input0 + '[SEP]' + x_input1 + '[SEP]']
                return self.tokenizer(query)['input_ids']  
        else:
            if x_input1 == 0:
                return [[self.tokenizer.cls_token_id]
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_input0))
                    + prompt_tokens * self.template[0]
                    + prompt_tokens * self.template[1]
                    + prompt_tokens * self.template[2]
                    + self.tokenizer.convert_tokens_to_ids(['.'])
                    + [self.tokenizer.sep_token_id]]

            return [[self.tokenizer.cls_token_id]
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_input0))
                    + [self.tokenizer.sep_token_id]
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_input1))
                    + prompt_tokens * self.template[0]
                    + prompt_tokens * self.template[1]
                    + prompt_tokens * self.template[2]
                    + self.tokenizer.convert_tokens_to_ids(['.'])
                    + [self.tokenizer.sep_token_id]]

    def init_weights(self):
        for name, param in self.model.named_parameters():
            if  param.requires_grad:
                init.normal_(param, mean=0.0, std=0.01)
        for name, param in self.prompt_encoder.named_parameters():
            if  param.requires_grad:
                init.normal_(param, mean=0.0, std=0.01)

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)
        raw_embeds_permt = raw_embeds.clone()

        if self.args.use_original_template:
            return raw_embeds_permt

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds_permt[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        
        raw_embeds_permt = self.transition_function(raw_embeds_permt)

        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = raw_embeds_permt[bidx, blocked_indices[bidx, i], :]
        
        return raw_embeds

    def cut_function(self, embeddings, args):
        max_length = args.max_length
        permt_embeddings = embeddings.clone()
        final_embeddings = permt_embeddings[:, : max_length, :]

        return final_embeddings

    def get_classifier_state_dict(self):
        classifier_name = 'classifier'
        classifier_dict = {}
        for name, param in self.model.named_parameters():
            if classifier_name in name:
                classifier_dict[name] = self.model.state_dict()[name]

        return classifier_dict

    def forward(self, x_input0, x_input1, labels):
        bz = len(x_input0)

        prompt_tokens = [self.pseudo_token_id]
        queries = [torch.LongTensor(self.get_query(x_input0[i], x_input1[i], prompt_tokens)).squeeze() for i in range(bz)]
        queries = pad_sequence(queries, True, padding_value = self.pad_token_id).long().to(self.device)
        attention_mask = queries != self.pad_token_id

        inputs_embeds = self.embed_input(queries)

        inputs_embeds = self.cut_function(inputs_embeds, self.args)
        attention_mask = attention_mask[:, : self.args.max_length]
        inputs_embeds.requires_grad_(True)
        output = self.model(inputs_embeds = inputs_embeds.to(self.device),
                                attention_mask = attention_mask.bool().to(self.device),
                                labels = labels.long().to(self.device))

        loss, outputs = output.loss, output.logits

        return outputs
