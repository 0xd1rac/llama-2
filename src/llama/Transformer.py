from src.transformer_code_blocks import ModelArgs, RMSNorm, EncoderBlock
import torch
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, 
                 args: ModelArgs
                 ) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.num_encoder_blocks = args.num_encoder_blocks
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.d_model)

        self.encoder_blocks = nn.ModuleList()

        for _ in range(args.num_encoder_blocks):
            self.encoder_blocks.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.d_model, eps=args.norm_eps)

        self.output = nn.Linear(args.d_model, self.vocab_size, bias=False)

    
    def forward(self, 
                x: torch.Tensor,
                start_pos: int
                ):
        # (batch_size, seq_len)
        batch_size, seq_len = x.shape

        assert seq_len == 1, "Only one token at a time can be processed"

        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        x = self.tok_embeddings(x)

        # consecutively apply all the encoder_blocks
        for block in self.encoder_blocks:
            x = block(x, start_pos)
        
        x = self.norm(x)
        x = self.output(x)

        return x
