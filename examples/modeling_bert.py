# Source: https://github.com/Tarpelite/UniNLP/blob/176c2a0f88c8054bf69e1f92693d353737367c34/transformers/modeling_bert.py#L2555
# class BertForParsingV2

import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


class BertForParsing(BertPreTrainedModel):
    def __init__(self, config, mlp_dim, num_labels):
        super(BertForParsing, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp_dim = mlp_dim

        self.mlp_head_arc = nn.Linear(config.hidden_size, mlp_dim)
        self.mlp_dep_arc = nn.Linear(config.hidden_size, mlp_dim)
        self.biaffine_arc = BiAffine(mlp_dim, 1)

        self.mlp_head_label = nn.Linear(config.hidden_size, mlp_dim)
        self.mlp_dep_label = nn.Linear(config.hidden_size, mlp_dim)
        self.biaffine_label = BiAffine(mlp_dim, num_labels)
        self.num_labels = num_labels

        self.init_weights()

    def decode(self, scores):
        # do viterbi decoding, build graph for every instance

        pass

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, heads=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        s_head_arc = self.mlp_head_arc(sequence_output)
        s_dep_arc = self.mlp_dep_arc(sequence_output)

        s_head_label = self.mlp_head_label(sequence_output)
        s_dep_label = self.mlp_head_label(sequence_output)

        # print("s_head", s_head)
        # print("s_dep", s_dep)

        # do the mask
        # attention_mask [batch_size, seq_len]

        # attention_mask = attention_mask.unsqueeze(-1).expand(attention_mask.shape + (attention_mask.size(-1),))

        logits_arc = self.biaffine_arc(s_head_arc, s_dep_arc)  # [batch_size, seq_len, seq_len]

        logits_arc = logits_arc.transpose(-1, -2)

        logits_label = self.biaffine_label(s_head_label, s_dep_label)  # [batch_size, num_labels, seq_len, seq_len]

        logits_label = logits_label.transpose(-1, -3)  # [batch_size, seq_len, seq_len, num_labels]

        preds = torch.argmax(logits_arc, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]

        indices = preds.unsqueeze(-1).expand(preds.shape + (self.num_labels,))  # [batch_size, seq_len, 1 , num_labels]

        # print("logits_label_shape", logits_label.shape)
        # print("indices shape", indices.shape)
        logits_label = torch.gather(logits_label, -2, indices).squeeze(-2)  # [batch_size, seq_len,num_labels]

        outputs = (logits_arc, logits_label) + outputs[2:]
        if heads is not None and labels is not None:
            loss_fct = CrossEntropyLoss()
            logits_arc = logits_arc.contiguous().view(-1, logits_arc.size(-1))
            heads = heads.view(-1)
            loss = loss_fct(logits_arc, heads)

            logits_label = logits_label.contiguous().view(-1, self.num_labels)
            labels = labels.view(-1)
            loss_labels = loss_fct(logits_label, labels)
            # print("loss heads", loss)
            # print("loss labels", loss_labels)
            # print("loss", loss)
            # print("preds", preds)
            # print("labels", labels)
            loss = loss + loss_labels
            outputs = (loss,) + outputs

        return outputs


class BiAffine(nn.Module):
    """Biaffine attention layer."""

    def __init__(self, input_dim, output_dim):
        super(BiAffine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.FloatTensor(output_dim, input_dim, input_dim))
        nn.init.xavier_uniform_(self.U)

    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1)
