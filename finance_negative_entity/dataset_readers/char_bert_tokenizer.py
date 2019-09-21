# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List

from allennlp.data import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
import collections
import logging
import os
import unicodedata
from io import open

from overrides import overrides

logger = logging.getLogger(__name__)

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        vocab[token.rstrip()] = index
    return vocab

@Tokenizer.register("char_bert_tokenizer")
class CharBertTokenizer(Tokenizer):
    def __init__(self, vocab_file, do_lower_case=True, unk_token="[UNK]", white_space_token="[unused1]"):
        self.unk_token = unk_token,
        self.white_space_token =  white_space_token
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_lower_case = do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    @overrides
    def tokenize(self, text):
        output_tokens = []
        if self.do_lower_case:
            text = text.lower()
        for char in text:
            if char in self.vocab.keys():
                token = Token(char)
            elif _is_whitespace(char):
                token = Token(self.white_space_token)
            else:
                token = Token(self.unk_token)
            output_tokens.append(token)
        return output_tokens

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

