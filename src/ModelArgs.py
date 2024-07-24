from common_imports import *

@dataclass 
class ModelArgs:
    d_model: int  = 4096 #embedding vector dimension
    num_encoder_blocks: int = 32
    num_Q_heads: int = 32 # Number of heads for the queries
    num_KV_heads: Optional[int] = 4 #Number of heads for the K and V 
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float]= None 
    norm_eps: float = 1e-5
    
    # Needed for KV cache 
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None