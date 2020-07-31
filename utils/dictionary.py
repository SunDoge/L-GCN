import re
import pickle
from typing import Dict, List
import spacy
import numpy as np
import torch


class Dictionary:
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        if idx2word is None:
            idx2word = list(word2idx.keys())
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.nlp = spacy.load('en')

    def _tokenize(self, sentence: str) -> List[str]:
        sentence = sentence.lower()
        sentence = sentence.replace('.', ' . ')
        doc = self.nlp.tokenizer(sentence)
        tokens = [token.text for token in doc if not token.is_space]
        tokens.append('<eos>')
        return tokens

    def tokenize(self, sentence: str, add_word: bool = False, extra_dict: Dict[str, int] = None):
     #       print(self.word2idx)
        # sentence = sentence.lower()
        # sentence = sentence.replace(',', '').replace(
        #     '?', '').replace('\'s', ' \'s')

        # words = sentence.split()

        # words = self.nlp.tokenizer(sentence)
        words = self._tokenize(sentence)

        tokens = []
        if add_word:
            for w in words:
                if extra_dict is None:
                    tokens.append(self.add_word(w))
                else:
                    if w in extra_dict:
                        tokens.append(self.add_word(w))

        else:
            for w in words:
                #                print(words)
                # w = w.lower()
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(self.word2idx['<unk>'])
        return tokens

    def dump_to_file(self, path):
        with open(path, 'wb') as f:
            pickle.dump([self.word2idx, self.idx2word], f)
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        with open(path, 'rb') as f:
            word2idx, idx2word = pickle.load(f)
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class CharDictionary(Dictionary):

    def __init__(self, word2idx=None, idx2word=None, max_length=16):
        if word2idx is None:
            word2idx = {'<pad>': 0, '<oov>': 1}
        if idx2word is None:
            idx2word = list(word2idx.keys())
        self.max_length = max_length

        super().__init__(word2idx=word2idx, idx2word=idx2word)

    def tokenize(self, sentence, add_word=False, extra_dict=None):
        words = self._tokenize(sentence)
        tokens = torch.zeros(len(words), self.max_length, dtype=torch.long)
        if add_word:
            for w in words:
                if extra_dict is None:
                    # tokens.append(self.add_word(w))
                    for c in w[:self.max_length]:
                        self.add_word(c)
                else:
                    if w in extra_dict:
                        # tokens.append(self.add_word(w))
                        for c in w[:self.max_length]:
                            self.add_word(c)

        else:
            for wi, w in enumerate(words):
                #                print(words)
                for ci, c in enumerate(w):
                    if ci < self.max_length:
                        if c in self.word2idx:
                            tokens[wi, ci] = self.word2idx[c]
                        else:
                            tokens[wi, ci] = self.word2idx['<unk>']

        return tokens


def clean_str(string, downcase=True):
    """
    Currently we don't use it.
    Tokenization/string cleaning for strings.
    Taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`(_____)]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower() if downcase else string.strip()
