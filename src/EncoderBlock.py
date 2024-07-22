import torch
import torch.nn as nn 
from ModelArgs import ModelArgs
from RMSNorm import RMSNorm
from SelfAttention import SelfAttention

class EncoderBlock(nn.Module):
    def __init__(self, 
                 args: ModelArgs
                 ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dimm = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the self attention
        self.attention_norm = RMSNorm(args.dim , eps=args.norm_eps)

        # Normalization BEFORE the feedforward block
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    
    def forward(self, 
                x: torch.Tensor,
                start_pos: int,
                freqs_complex: torch.Tensor
                ):
        
        h = x + self.attention.forward(self.attention_norm(x),
                                       start_pos,
                                       freqs_complex
                                       )
    
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out
