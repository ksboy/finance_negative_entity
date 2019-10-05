#!/usr/bin/env python
import logging
from typing import Dict, List, Optional, Any

import torch
from torch.nn.modules.linear import Linear
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from overrides import overrides
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@Model.register("char_bert_crf")
class CharBertCrfModel(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 dropout: Optional[float] = 0,
                 label_encoding: Optional[str] = 'BIO',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(CharBertCrfModel, self).__init__(vocab, regularizer)
        self._text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size('labels')

        self._labels_predictor = Linear(self._text_field_embedder.get_output_dim(), self.num_tags)
        self.dropout = torch.nn.Dropout(dropout)
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self._f1_metric = SpanBasedF1Measure(vocab, tag_namespace='labels', label_encoding=label_encoding)
        labels = self.vocab.get_index_to_token_vocabulary('labels')
        constraints = allowed_transitions(label_encoding, labels)
        self.label_to_index = self.vocab.get_token_to_index_vocabulary('labels')
        self.crf = ConditionalRandomField(self.num_tags, constraints, include_start_end_transitions=False)
        # self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                text_tokens: Dict[str, torch.LongTensor],
                labels: torch.IntTensor = None,
                metadata = None) -> Dict[str, torch.Tensor]:

        embedded_question_and_passage = self._text_field_embedder(text_tokens)
        tag_logits = self._labels_predictor(embedded_question_and_passage)
        predicted_probability = torch.nn.functional.softmax(tag_logits, dim=-1)
        # print(tag_logits, text_tokens["mask"])
        try:
            best_paths = self.crf.viterbi_tags(tag_logits, text_tokens["mask"])
        except Exception as e:
            print(e)
            print(tag_logits)
            print(text_tokens)
            raise Exception('crf bug~')
        # Just get the tags and ignore the score.
        # print(best_paths)
        predicted_tags = [x for x, y in best_paths]
        output_dict = {"logits": tag_logits, "text_tokens": text_tokens, "tags": predicted_tags,
                       "probabilities": predicted_probability}
        if labels is not None:
            log_likelihood = self.crf(tag_logits, labels, text_tokens["mask"])
            output_dict["loss"] = -log_likelihood
            class_probabilities = tag_logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    if tag_id >= len(self.label_to_index):
                        tag_id = self.label_to_index['O']
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, labels, text_tokens["mask"].float())
            self._f1_metric(class_probabilities, labels, text_tokens["mask"].float())
        output_dict["metadata"] = metadata
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag, namespace="labels")
             for tag in instance_tags]
            for instance_tags in output_dict["tags"]
        ]
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}

        f1_dict = self._f1_metric.get_metric(reset=reset)

        metrics_to_return.update({x: y for x, y in f1_dict.items() if "overall" in x})
        return metrics_to_return