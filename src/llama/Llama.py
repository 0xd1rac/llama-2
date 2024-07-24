import torch
import torch.nn as nn
from .Transformer import Transformer
from src.transformer_code_blocks import ModelArgs
from sentencepiece import SentencePieceProcessor


class Llama(nn.Module):
    def __init__(self,
                 model: Transformer,
                 tokenizer: SentencePieceProcessor,
                 args: ModelArgs
                ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.args = args    
        