# encoding: utf-8
"""
@author: banifeng
@contact: banifeng@126.com

@version: 1.0
@file: bert_seq_pair_cls_reader.py
@time: 2019-08-09 16:39
"""

import json
import logging

from overrides import overrides
from allennlp.data import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from typing import Dict

logger = logging.getLogger(__name__)


@DatasetReader.register("bert_seq_pair_cls")
class BertSeqPairReader(DatasetReader):
    def __init__(self,
                 tokenizers: Dict[str, Tokenizer]=None,
                 token_indexers: Dict[str, TokenIndexer]=None,
                 lazy: bool = False,
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizers.get("bert_tokenizer")
        self._token_indexers = token_indexers
        self.oov_tokens = {
            "“": '"',
            "”": '"',
            " ": "，"
        }

    @overrides
    def _read(self, file_path: str):
        with open(file_path) as f:
            for line in f:
                json_line = json.loads(line)
                instance = self.text_to_instance(json_line['passage'], json_line['question'], json_line['label'])
                yield instance

    def replace_oov(self, input_string: str):
        input = list(input_string)
        output = []
        for c in input:
            if c in self.oov_tokens:
                c = self.oov_tokens[c]
            output.append(c)
        return "".join(output)

    @overrides
    def text_to_instance(self,
                         passage: str,
                         question: str,
                         label: str = None) -> Instance:
        passage = self.replace_oov(passage)
        question = self.replace_oov(question)
        passage_tokens = self._tokenizer.tokenize(passage)
        question_tokens = self._tokenizer.tokenize(question)
        content_tokens = [Token('[CLS]') ]
        content_tokens.extend(passage_tokens)
        content_tokens.append(Token('[SEP]'))
        content_tokens.extend(question_tokens)
        # content_tokens = + passage_tokens +  + question_tokens
        fields = {'content': TextField(content_tokens, self._token_indexers)}
        if label:
            fields['label'] = LabelField(label)
            fields['metadata'] = MetadataField({'passage': passage,
                                                'question': question,
                                                'label': label})
        return Instance(fields)
