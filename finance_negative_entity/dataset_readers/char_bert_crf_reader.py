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
                passage_text = self.replace_oov(self.replace_emoji(line_json['passage']))
                aspects = line_json['aspects']
                opinions = line_json['opinions']
                instance = self.text_to_instance(passage_text, aspects, opinions)
                yield instance

    def replace_oov(self, input_string: str):
        input = list(input_string)
        output = []
        for c in input:
            if c in self.oov_tokens:
                c = self.oov_tokens[c]
            output.append(c)
        return "".join(output)

    def replace_emoji(self, input_string: str):
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"🤮"  # passage_id =7371
                           "]", flags=re.UNICODE)
        return emoji_pattern.sub(r'，，', input_string)

    """
    {
        'passage': '使用一段时间才来评价，淡淡的香味，喜欢！',
        'aspects': [['香味', 14, 16]],
        'opinions': [['喜欢', 17, 19]]
    }
    """
    @overrides
    def text_to_instance(self,
                         passage: str,
                         aspects: List[List[str]] = None,
                         opinions: List[List[str]] = None
                         ) -> Instance:
        passage = self.replace_oov(passage)
        passage_tokens = self._tokenizer.tokenize(passage)
        assert len(passage_tokens) == len(passage)
        # 设置第一个位置为[CLS]
        passage_tokens.insert(0, Token("[CLS]"))
        metadata = {
            "passage": passage,
            "aspects": aspects,
            "opinions": opinions,
        }
        fields = {
            "text_tokens": TextField(passage_tokens, self._token_indexers),
            "metadata": MetadataField(metadata)
        }
        if aspects is not None:
            labels = ["O"] * len(passage)
            for aspect in aspects:
                labels[aspect[1]] = "B-ASP"
                for i in range(aspect[1]+1, aspect[2]):
                    labels[i] = "I-ASP"
            for opinion in opinions:
                labels[opinion[1]] = "B-OPI"
                for i in range(opinion[1]+1, opinion[2]):
                    labels[i] = "I-OPI"
            # 对应第一个位置的[CLS]
            labels.insert(0, "O")
            fields["labels"] = SequenceLabelField(labels, TextField(passage_tokens, self._token_indexers))
        return Instance(fields)
