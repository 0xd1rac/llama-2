from common_imports import *
from ModelArgs import ModelArgs

class FeedForward(nn.Module):
    def __init__(self,
                 args: ModelArgs
                 ):
        super().__init__()

        hiddem_dim = 4 * args.d_model
        hidden_dim = int(2 * hiddem_dim /3)

        # adjust the hidden dimenison based on ffn_multiplier (if provided)
        if args.ffn_dim_multiplier is not None:
            hiddem_dim = int(args.ffn_dim_multiplier * hiddem_dim)
        
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # Define linear lyaers for the feedforward network 
        self.fc1 = nn.Linear(args.d_model, hiddem_dim, bias=False)
        self.fc2 = nn.Linear(hiddem_dim, args.d_model, bias=False)
        self.fc3 = nn.Linear(args.d_model, hiddem_dim, bias=False)
    
    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        # x shape => (batch_size, seq_len, d_model)

        # Apply the first linear transformation + activation (swish)
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, hidden_dim)
        o_1 = F.silu(self.fc1(x))

        # apply the second linear transformtation
        o_2 = self.fc2(o_1)

        # elemnent wise multiplication
        x = o_1 * o_2 

        # apply the third linear transformation
        x = self.fc3(x)

        return x 



