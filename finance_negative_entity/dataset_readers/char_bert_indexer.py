# pylint: disable=no-self-use
import collections
from typing import Dict, List, Callable
import logging

from overrides import overrides

from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)

# TODO(joelgrus): Figure out how to generate token_type_ids out of this token indexer.

# This is the default list of tokens that should not be lowercased.
_NEVER_LOWERCASE = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        vocab[token.rstrip()] = index
    return vocab

@TokenIndexer.register("char_bert_token_indexer")
class CharBertTokenIndexer(TokenIndexer[int]):
    def __init__(self,
                 vocab_file: str,
                 namespace: str = "bert",
                 separator_token: str = "[SEP]"
                ) -> None:
        self.vocab = load_vocab(vocab_file)
        self._namespace = namespace
        self._added_to_vocabulary = False
        self._separator_ids = [self.vocab[separator_token]]

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        # pylint: disable=protected-access
        for word, idx in self.vocab.items():
            vocabulary._token_to_index[self._namespace][word] = idx
            vocabulary._index_to_token[self._namespace][idx] = word

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True
        token_ids = [self.vocab.get(token.text, self.vocab.get("[UNK]")) for token in tokens]
        mask = [1 for _ in token_ids]
        token_type_ids = _get_token_type_ids(token_ids,
                                             self._separator_ids)
        return {
            index_name: token_ids,
            f"{index_name}-type-ids": token_type_ids,
            "mask": mask
        }

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        """
        We need to override this because the indexer generates multiple keys.
        """
        # pylint: disable=no-self-use
        return [index_name, f"{index_name}-type-ids"]


def _get_token_type_ids(token_ids: List[int],
                        separator_ids: List[int]) -> List[int]:
    num_wordpieces = len(token_ids)
    token_type_ids: List[int] = []
    type_id = 0
    cursor = 0
    while cursor < num_wordpieces:
        # check length
        if num_wordpieces - cursor < len(separator_ids):
            token_type_ids.extend(type_id
                                  for _ in range(num_wordpieces - cursor))
            cursor += num_wordpieces - cursor
        # check content
        # when it is a separator
        elif all(token_ids[cursor + index] == separator_id
                 for index, separator_id in enumerate(separator_ids)):
            token_type_ids.extend(type_id for _ in separator_ids)
            type_id += 1
            cursor += len(separator_ids)
        # when it is not
        else:
            cursor += 1
            token_type_ids.append(type_id)
    return token_type_ids




