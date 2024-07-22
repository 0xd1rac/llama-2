import torch
import torch.nn as nn
from ModelArgs import ModelArgs


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # indicates the number of heads for the Keys, K and Values, V 
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        
        # indicatess the number of heads of queries, Q 
        self.n_heads_q = args.n_heads

        # indicates how many times the heads of Keys and Values should be repeated to match the head of queries 
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, 
                            args.n_heads * self.head_dim,
                            bias=False
                            )
        
        self.wk = nn.Linear(args.dim,
                            self.n_kv_heads * self.head_dim,
                            bias=False
                            )
        
        self.wk = nn.Linear(args.dim,
                            self.n_kv_heads * self.head_dim,
                            bias=False
                            )



