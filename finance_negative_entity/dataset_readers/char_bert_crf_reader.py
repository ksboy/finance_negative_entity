import json
import logging

from allennlp.data import Tokenizer
from overrides import overrides
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from typing import Dict, List
import re
logger = logging.getLogger(__name__)

@DatasetReader.register("char_bert_crf_reader")
class CharBertCrfReader(DatasetReader):
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
        logger.info("Reading file at %s", file_path)
        with open(file_path) as data_file:
            for line in data_file:
                line = line.rstrip("\n")
                if not line:
                    continue
                line_json = json.loads(line)
                passage_text = self.replace_oov(line_json['passage'])
                entities = line_json['entity']
                key_entities = line_json['key_entity']
                instance = self.text_to_instance(passage_text, entities, key_entities)
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
                         entities: List[List[str]] = None,
                         key_entities: List[List[str]] = None
                         ) -> Instance:
        passage = self.replace_oov(passage)
        passage_tokens = self._tokenizer.tokenize(passage)
        assert len(passage_tokens) == len(passage)
        # 设置第一个位置为[CLS]
        passage_tokens.insert(0, Token("[CLS]"))
        metadata = {
            "passage": passage,
            "entities": entities,
            "key_entities": key_entities,
        }
        fields = {
            "text_tokens": TextField(passage_tokens, self._token_indexers),
            "metadata": MetadataField(metadata)
        }
        if entities is not None:
            labels = ["O"] * len(passage)
            # for entity in entities:
            #     labels[entity[1]] = "B"
            #     for i in range(entity[1]+1, entity[2]):
            #         labels[i] = "I"
            for key_entitiy in key_entities:
                labels[key_entitiy[1]] = "B-NEG"
                for i in range(key_entitiy[1]+1, key_entitiy[2]):
                    labels[i] = "I-NEG"
            # 对应第一个位置的[CLS]
            labels.insert(0, "O")
            fields["labels"] = SequenceLabelField(labels, TextField(passage_tokens, self._token_indexers))
        return Instance(fields)
