import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BertCRFModel(nn.Module):

    def __init__(self, bert_base_model_dir, label_size, drop_out_rate=0.5):
        super(BertCRFModel, self).__init__()
        self.label_size = label_size

        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
        self.bert_model = BertModel.from_pretrained(bert_base_model_dir)

        self.dropout = nn.Dropout(drop_out_rate)
        self.linear = nn.Linear(self.bert_model.config.hidden_size, label_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None):

        output = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=None
        )
        last_hidden_state = output.last_hidden_state
        pooled_output = output.pooler_output

        seq_outs = self.dropout(last_hidden_state)
        logits = self.linear(seq_outs)

        if labels is not None:
            loss_func = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略值为-1的label，不参与loss计算
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.label_size)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_func(active_logits, active_labels)
            else:
                loss = loss_func(logits.view(-1, self.label_size), labels.view(-1))
            return logits, loss

        return logits

    def get_bert_tokenizer(self):
        return self.bert_tokenizer
