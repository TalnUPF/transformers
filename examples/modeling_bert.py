# Source: https://github.com/Tarpelite/UniNLP/blob/176c2a0f88c8054bf69e1f92693d353737367c34/transformers/modeling_bert.py#L2555
# class BertForParsingV2

import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel

logger = logging.getLogger(__name__)


class BertForParsing(BertPreTrainedModel):
    """ Implements Dozat&Maning DEEP BIAFFINE ATTENTION FOR NEURAL DEPENDENCY PARSING
    https://arxiv.org/pdf/1611.01734.pdf
    """

    def __init__(self, config, mlp_dim, num_labels):
        super(BertForParsing, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp_dim = mlp_dim
        self.num_labels = num_labels

        self.mlp_arc_head = nn.Linear(config.hidden_size, mlp_dim)  # Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
        self.mlp_arc_dep = nn.Linear(config.hidden_size, mlp_dim)
        self.biaffine_classifier_arcs = BiAffine(mlp_dim, 1)

        self.mlp_label_head = nn.Linear(config.hidden_size, mlp_dim)
        self.mlp_label_dep = nn.Linear(config.hidden_size, mlp_dim)
        self.biaffine_classifier_labels = BiAffine(mlp_dim, num_labels)

        self.init_weights()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                heads=None,
                labels=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # Dozat&Manning: "rather than using the top recurrent states of the
        #   LSTM in the biaffine transformations, we first put them
        #   through MLP operations that reduce their dimensionality."
        s_arc_head = self.mlp_arc_head(sequence_output)
        s_arc_dep = self.mlp_arc_dep(sequence_output)
        logits_arc = self.biaffine_classifier_arcs(s_arc_head, s_arc_dep)  # [batch_size, seq_len, seq_len]
        logits_arc = logits_arc.transpose(-1, -2)

        s_label_head = self.mlp_label_head(sequence_output)
        s_label_dep = self.mlp_label_dep(sequence_output)  # TODO lpmayos: I change this to mlp_label_dep (it was mlp_label_head)
        logits_label = self.biaffine_classifier_labels(s_label_head, s_label_dep)  # [batch_size, num_labels, seq_len, seq_len]
        logits_label = logits_label.transpose(-1, -3)  # [batch_size, seq_len, seq_len, num_labels]

        preds = torch.argmax(logits_arc, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        indices = preds.unsqueeze(-1).expand(preds.shape + (self.num_labels,))  # [batch_size, seq_len, 1 , num_labels]
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


class BertForSRL(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSRL, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.num_labels = config.num_labels
        self.num_BIO_labels = 3
        self.num_CRO_labels = 3
        self.num_SRL_labels = 22

        self.BIO_classifier = nn.Bilinear(config.hidden_size, 1, self.num_BIO_labels)
        self.CRO_classifier = nn.Bilinear(config.hidden_size, self.num_BIO_labels, self.num_CRO_labels)
        self.SRL_classifier = nn.Bilinear(config.hidden_size, self.num_CRO_labels, self.num_SRL_labels)
        self.label_classifier = nn.Bilinear(config.hidden_size, self.num_SRL_labels, self.num_labels)

        self.init_weights()

    def forward(self,
                input_ids=None,
                verb_seq_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                label_BIO_ids=None,
                label_CRO_ids=None,
                label_SRL_ids=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        BIO_logits = self.BIO_classifier(sequence_output, verb_seq_ids.unsqueeze(-1).float())
        CRO_logits = self.CRO_classifier(sequence_output, BIO_logits)
        SRL_logits = self.SRL_classifier(sequence_output, CRO_logits)

        logits = self.label_classifier(sequence_output, SRL_logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()

            loss_BIO = loss_fct(BIO_logits.view(-1, self.num_BIO_labels), label_BIO_ids.view(-1))
            loss_CRO = loss_fct(CRO_logits.view(-1, self.num_CRO_labels), label_CRO_ids.view(-1))
            loss_SRL = loss_fct(SRL_logits.view(-1, self.num_SRL_labels), label_SRL_ids.view(-1))

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss = loss + loss_BIO + loss_CRO + loss_SRL
            outputs = (loss,) + outputs

        return outputs
