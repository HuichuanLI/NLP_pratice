# coding=utf-8

import torch
import torch.nn as nn

"""
num_prompt_tokens: 6 <prompt_1> <propmt_2> ... <prompt_6>
tokenizer.add_special_tokens(prompt_tokens)
bert tokenizer: 21128
extended bert tokenizer: 21134
<prompt_1>: 21129 - 21128 = 1

              precision    recall  f1-score   support

           0       0.61      0.56      0.58       129
           1       0.80      0.83      0.81       271

    accuracy                           0.74       400
   macro avg       0.70      0.69      0.70       400
weighted avg       0.74      0.74      0.74       400

"""


class PromptEncoder(nn.Module):
    def __init__(self, num_prompt_tokens, offset, embdding_dim=768):
        super(PromptEncoder, self).__init__()
        self.offset = offset
        self.embedding = torch.nn.Embedding(num_prompt_tokens,embdding_dim) # [1,2,3,4,5,6]
    
    def forward(self,prompt_token_ids,prompt_ids=None):
        """
            1. LSTM
            2. MLP
        """
        prompt_token_ids = prompt_token_ids - self.offset

        return self.embedding(prompt_token_ids)

class SoftPromptWrapper(nn.Module):
    def __init__(self,model, prompt_embdding, replacing_token_id=0):
        """
        SoftPromptWrapper for Huggingface transformer models (Encoder Models).
        Args:
            model:  transformer pretrained Masked Language Model
            prompt_embdding: embedding lookup table for prompt tokens
            replacing_token_id: 
        """
        super(SoftPromptWrapper, self).__init__()
        self.underlying_model = model
        self.bert_embedding = model.get_input_embeddings()
        self.prompt_embdding = prompt_embdding
        self.replacing_token_id = replacing_token_id
        
        self.original_vocab_size = self.underlying_model.config.vocab_size # 21128
        self.prompt_token_fn = lambda t:(t>=self.original_vocab_size) # 21128 

    def forward(self, input_ids, attention_masks, segment_ids):
        # '<prompt_1><prompt_2>[MASK]<prompt_3>...<prompt_n-2><prompt_n-1><prompt_n>' + 原始数据
        prompt_masks = self.prompt_token_fn(input_ids) # 原始的bert vocab + prompt tokens

        input_ids_ = input_ids.clone()

        input_ids_[prompt_masks]=self.replacing_token_id # 方便直接调用 ber_embedding 
        
        inputs_embeds = self.bert_embedding(input_ids_)
        prompt_embeds = self.prompt_embdding(input_ids[prompt_masks]).to(device=inputs_embeds.device)
        
        inputs_embeds[prompt_masks]=prompt_embeds
        
        return self.underlying_model(inputs_embeds=inputs_embeds, attention_mask=attention_masks, \
            token_type_ids=segment_ids)

    