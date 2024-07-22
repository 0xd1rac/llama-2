
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math
from ModelArgs import ModelArgs
from RMSNorm import RMSNorm


# I dont rlly get this part 
def precompute_theta_pos_frequencies(head_dim: int, 
                                     seq_len: int,
                                     device: str,
                                     theta: float = 10000.0
                                     ):
    # as written in paper,. the dimension of the embedding must be even 
    assert head_dim % 2 == 0, "Embedding Dimension must be divisible by 2"
    # Build the theta parameters
    # according to the formula, theta_i = 10000 ^ 
    # shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    m = torch.arange()



class Transformer(nn.Module):

    def __init__(self, 
                 args: ModelArgs
                 ) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()

        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
                                        args.dim // self.args.n_head,
                                        args.max_seq_len * 2,
                                        device=args.device
                                        )
    
    def forward(self, 
                tokens: torch.Tensor,
                start_pos: int
                ):
        # (batch_size, seq_len)
        batch_size, seq_len = tokens.shape

        assert batch_size == 1, "Only one token at a time can be processed"

        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # retrieve the paris (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h).float()

        return output 
