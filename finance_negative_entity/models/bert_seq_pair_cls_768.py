# encoding: utf-8
"""
@author: banifeng
@contact: banifeng@126.com

@version: 1.0
@file: bert_seq_pair_cls.py
@time: 2019-08-09 16:39
"""

import logging
from typing import Dict, Optional, List, Any

import numpy
import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

logger = logging.getLogger(__name__)
device = torch.device('cuda:1')

# logger.setLevel(logging.DEBUG)

@Model.register("bert_seq_pair_clf_768")
class BertSeqPairClsfModel_768(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 class_weights: List[float] = (1.0, 1.0),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self.classifier_feedforward = classifier_feedforward
        self.num_classes = self.vocab.get_vocab_size('labels')

        assert self.num_classes == classifier_feedforward.get_output_dim()

        # if classifier_feedforward.get_input_dim() != 768:
        #     raise ConfigurationError(F"The input dimension of the classifier_feedforward, "
        #                              F"found {classifier_feedforward.get_input_dim()}, must match the "
        #                              F" output dimension of the bert embeder, {768}")
        index = 0
        if self.num_classes == 2:
            index = self.vocab.get_token_index("正类", "labels")
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": F1Measure(index)
        }
        # weights = torch.Tensor(class_weights)
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                content: Dict[str, torch.LongTensor],
                sample_weights: torch.Tensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, str]] = None) -> Dict[str, Any]:

        embeded_content = self._text_field_embedder(content)
        # # print("embeded_content大小", embeded_content.size())
        bert_cls_vec = embeded_content[:, 0, :]

        logits = self.classifier_feedforward(bert_cls_vec)
        output_dict = {'logits': logits}

        if label is not None:
            loss = self.loss(logits, label)
            output_dict['loss'] = loss
            for metric in self.metrics.values():
                metric(logits, label)

        output_dict = self._decode(output_dict)

        # debug here
        # if label is not None:
        #     for i, label in enumerate(output_dict['label']):
        #         if metadata[i]['label'] != label:
        #             logger.debug(metadata[i])
        #             logger.debug(label)
        #             print(metadata[i])
        #             print(label)
        #             print()
        return output_dict

    def _decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
                Does a simple argmax over the class probabilities, converts indices to string labels, and
                adds a ``label`` key to the dictionary with the result
                """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        # 如果label是str则需要以下这两行
        labels = [self.vocab.get_token_from_index(x, namespace='labels') for x in argmax_indices]
        output_dict['label'] = labels
        # output_dict['label'] = argmax_indices
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items() if metric_name != "f1"}
        precision, recall, f1 = self.metrics['f1'].get_metric(reset)
        update_dict = {"precision": precision, "recall": recall, "f1": f1}
        metrics_to_return.update(update_dict)
        return metrics_to_return
