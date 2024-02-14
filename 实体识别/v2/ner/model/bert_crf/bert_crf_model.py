import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AlbertModel
from transformers import BertTokenizer, BertModel
from transformers import ElectraTokenizer, ElectraModel

from ner.focal_loss import FocalLoss
from ner.model.crf.crf_layer import CRF


class BertCRFModel(nn.Module):

    def __init__(self, bert_base_model_dir, label_size, drop_out_rate=0.5,
                 loss_type='crf_loss', focal_loss_gamma=2, focal_loss_alpha=None):  # # # 增加loss参数
        super(BertCRFModel, self).__init__()
        self.label_size = label_size

        assert loss_type in ('crf_loss', 'cross_entropy_loss', 'focal_loss')  # # # 确保loss合法
        if focal_loss_alpha:  # # # 确保focal_loss_alpha合法，必须是一个label的概率分布
            assert isinstance(focal_loss_alpha, list) and len(focal_loss_alpha) == label_size
        self.loss_type = loss_type  # # # 添加loss_type
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if 'albert' in bert_base_model_dir.lower():
            # 注意albert base使用bert tokenizer，参考https://huggingface.co/voidful/albert_chinese_base
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = AlbertModel.from_pretrained(bert_base_model_dir)
        elif 'electra' in bert_base_model_dir.lower():
            self.bert_tokenizer = ElectraTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = ElectraModel.from_pretrained(bert_base_model_dir)
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = BertModel.from_pretrained(bert_base_model_dir)

        self.dropout = nn.Dropout(drop_out_rate)
        self.linear = nn.Linear(self.bert_model.config.hidden_size, label_size)

        self.crf = CRF(label_size)  # 定义CRF层

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None):
        bert_out = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=None
        )
        if isinstance(self.bert_model, ElectraModel):
            last_hidden_state, = bert_out.last_hidden_state
        else:
            last_hidden_state, pooled_output = bert_out.last_hidden_state, bert_out.pooler_output
        seq_outs = self.dropout(bert_out.last_hidden_state)
        logits = self.linear(seq_outs)

        lengths = attention_mask.sum(axis=1)
        # # # 根据loss_type，选择使用维特比解码或直接argmax
        if self.loss_type == 'crf_loss':
            best_paths = self.crf.get_batch_best_path(logits, lengths)
        else:
            best_paths = torch.argmax(logits, dim=-1)
        best_paths = best_paths.to(self.device)  # # data parallel必须确保返回值在gpu

        if labels is not None:
            # # # 根据不同的loss_type，选择不同的loss计算
            active_loss = attention_mask.view(-1) == 1  # 通过attention_mask忽略pad
            active_logits, active_labels = logits.view(-1, self.label_size)[active_loss], labels.view(-1)[active_loss]
            if self.loss_type == 'crf_loss':
                # 计算loss时，忽略[CLS]、[SEP]以及PAD部分
                loss = self.crf.negative_log_loss(inputs=logits[:, 1:, :], length=lengths - 2, tags=labels[:, 1:])
            elif self.loss_type == 'cross_entropy_loss':
                loss = CrossEntropyLoss(ignore_index=-1)(active_logits, active_labels)  # label=-1被忽略
            else:
                active_loss = active_labels != -1  # 进一步忽略-1的部分，即[CLS]和[SEP]
                active_logits, active_labels = active_logits[active_loss], active_labels[active_loss]
                loss = FocalLoss(gamma=self.focal_loss_gamma, alpha=self.focal_loss_alpha)(active_logits, active_labels)
            return best_paths, loss

        return best_paths  # 直接返回预测的labels

    def get_bert_tokenizer(self):
        return self.bert_tokenizer
