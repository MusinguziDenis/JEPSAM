
import numpy as np
import torch
from typing import Union, Any, List

class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.TOKENS_MAPPING, self.REVERSE_TOKENS_MAPPING  = vocab.load_vocab() 

    def tokenize_by_space(
        self, 
        full_string:str
    )->list:
        return full_string.split()


    def idx2token(
        self, 
        i:int
    )->str:
        return self.REVERSE_TOKENS_MAPPING[i]

    def token2idx(
        self, 
        t:str
    )->int:
        return self.TOKENS_MAPPING[t]

    def encode(self, input:str):
        tokens = self.tokenize_by_space(input)
        tokens = [self.vocab.SPECIAL_TOKENS[0]] + tokens + [self.vocab.SPECIAL_TOKENS[-1]]
        token_ids = torch.tensor([self.token2idx(t) for t in tokens])
        # enc = {
        #     "tokens": tokens,
        #     "input_ids": torch.tensor(token_ids)
        # }
        return token_ids
    
    def batch_encode(self, input:Union[List, np.ndarray]):
        batch_size = len(input)
        batch_token_ids = [self.encode(inp) for inp in input]
        return batch_token_ids
    
    def decode(self, token_ids:torch.tensor)->list:
        tokens = [self.idx2token(i) for i in token_ids.tolist()]
        return tokens
    
    def batch_decode(self, input:Union[List, np.ndarray]):

        batch_size = len(input)

        batch_tokens = [self.decode(inp) for inp in input]

        return batch_tokens