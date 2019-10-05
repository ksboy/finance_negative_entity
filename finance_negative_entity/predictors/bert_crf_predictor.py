#!/usr/bin/env python
from typing import List
import logging
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Predictor.register('bert_crf_predictor')
class BertCrfPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """

    def predict(self, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"passage": passage})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        # question_text = " ".join(list(json_dict["question"]))
        # passage_text = " ".join(list(json_dict["passage"]))
        passage_text = json_dict["passage"]
        return self._dataset_reader.text_to_instance(passage_text)



