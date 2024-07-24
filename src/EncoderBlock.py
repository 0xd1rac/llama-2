from common_imports import *

from ModelArgs import ModelArgs
from RMSNorm import RMSNorm
from src.GroupQueryAttention import GroupQueryAttention
from FeedForward import FeedForward

class EncoderBlock(nn.Module):
    def __init__(self, 
                 args: ModelArgs
                 ):
        super().__init__()
        self.num_Q_heads = args.num_Q_heads
        self.d_model = args.d_model
        self.Q_head_size = args.d_model // args.num_Q_heads

        self.attention = GroupQueryAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the self attention
        self.attention_norm = RMSNorm(args.d_model , eps=args.norm_eps)

        # Normalization BEFORE the feedforward block
        self.ffn_norm = RMSNorm(args.d_model, args.norm_eps)

    
    def forward(self, 
                x: torch.Tensor,
                start_pos: int,
                ) -> torch.Tensor:
        
        h = x + self.attention.forward(self.attention_norm(x),
                                       start_pos,
                                       )
    
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out
