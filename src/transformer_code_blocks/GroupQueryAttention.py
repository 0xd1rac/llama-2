from .common_imports import *
from .ModelArgs import ModelArgs
from .RotaryPositionEmbedding import RotaryPositionEmbedding


class GroupQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        """
        Number of heads for keys (K) and values(V). 
        
        If not specified defaults to the number of heads 
        for quries (Q)
        """
        self.num_Q_heads = args.num_Q_heads
        self.num_KV_heads = args.num_Q_heads if args.num_KV_heads is None else args.num_KV_heads
        
        self.Q_head_size = args.d_model // self.num_Q_heads

        # Now we need to figure out how many times the KV head needs to be replicated per group
        self.KV_head_replication_factor = self.num_Q_heads // self.num_KV_heads

        """
        KV heads in the same group are repeated to match the number of Q heads and then aligned to the Q heads 
        before computing the attention.
        """

        self.W_Q = nn.Linear(self.d_model,
                            self.num_Q_heads * self.Q_head_size,
                            bias=False
                            )
        self.W_K = nn.Linear(self.d_model,
                            self.num_KV_heads * self.Q_head_size,
                            bias=False
                            )
        
        self.W_V = nn.Linear(self.d_model,
                            self.num_KV_heads * self.Q_head_size,
                            bias=False
                            )
        
        self.WO = nn.Linear(self.num_Q_heads * self.Q_head_size,
                            self.d_model,
                            bias=False
                            )
        
        self.cache_k = torch.zeroes((
                            args.max_batch_size,
                            args.max_seq_len,
                            self.num_KV_heads,
                            self.Q_head_size
                            ))
        self.cache_v = torch.zeroes((
                            args.max_batch_size,
                            args.max_seq_len,
                            self.num_KV_heads,
                            self.Q_head_size
                            ))
        
        self.rope = RotaryPositionEmbedding(self.Q_head_size, 
                                            args.max_seq_len,
                                            args.device
                                            )

    @staticmethod
    def repeat_heads(x: torch.Tensor,
                     KV_head_replication_factor: int
                     ) -> torch.Tensor:
        # repeat the heads of K and V to match the number of heads in Q
        batch_size, seq_len, num_KV_heads, Q_head_size = x.shape
        if KV_head_replication_factor == 1:
            return x 
        else:
            return (
                x[:, :, :, None, :]
                .expand(batch_size, seq_len, num_KV_heads, KV_head_replication_factor, Q_head_size)
                .reshape(batch_size, seq_len, num_KV_heads * KV_head_replication_factor, Q_head_size)
            )


    def forward(self, 
                x: torch.Tensor,
                start_pos:int
                ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape  # (batch_size, seq_len, d_model)

        Q = self.W_Q(x) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_Q_heads * Q_head_size)
        K = self.W_K(x) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_KV_heads * Q_head_size)
        V = self.W_V(x) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_KV_heads * Q_head_size)

        # (batch_size, seq_len, num_Q_heads * Q_head_size) -> (batch_size, seq_len, num_Q_heads, Q_head_size)
        Q = Q.view(batch_size, seq_len, self.num_Q_heads, self.Q_head_size)

        # (batch_size, seq_len, num_KV_heads * Q_head_size) -> (batch_size, seq_len, num_KV_heads, Q_head_size)
        K = K.view(batch_size, seq_len, self.num_KV_heads, self.Q_head_size) 

        # (batch_size, seq_len, num_KV_heads * Q_head_size) -> (batch_size, seq_len, num_KV_heads, Q_head_size)
        V = V.view(batch_size, seq_len, self.num_KV_heads, self.Q_head_size)

        Q = self.rope(Q, start_pos)
        K = self.rope(Q, start_pos)

        # Update key and value caches 
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = K
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = V

        # Retrive key and value caches 
        K_cache = self.cache_k[:batch_size, :start_pos + seq_len]
        V_cache = self.cache_v[:batch_size, :start_pos + seq_len]

        # Repeat the heads of K and V to match the number of heads in Q
        # (batch_size, seq_len, num_KV_heads, Q_head_size) -> (batch_size, seq_len, num_Q_heads, Q_head_size)
        K_cache = self.repeat_heads(K_cache, self.KV_head_replication_factor)
        V_cache = self.repeat_heads(V_cache, self.KV_head_replication_factor)


        # (batch_size, seq_len, num_Q_heads, Q_head_size) -> (batch_size, num_Q_heads, seq_len, Q_head_size)
        # reordering is necessary for the attention score computation. 
        Q = Q.transpose(1,2)
        K_cache = K_cache.transpose(1,2)
        V_cache = V_cache.transpose(1,2)

        # Compute Attention Scores
        """
            Q: 
                (batch_size, num_Q_heads, seq_len, Q_head_size)

            K_cach.transpose(-2, -1): 
                batch_size, num_Q_heads, seq_len, Q_head_size) -> (batch_size, num_Q_heads, Q_head_size, seq_len)

            matmul(Q, K_cache.tranpose(-2,-1))
                (batch_size, num_Q_heads, seq_len, Q_head_size) @ (batch_size, num_Q_heads, Q_head_size, seq_len)
                = (batch_size, num_Q_heads, seq_len, seq_len)

         """

        # after matmul: (batch_size, num_Q_heads, seq_len, seq_len)
        attention_proba = torch.matmul(Q,K_cache.transpose(-2,-1)) / math.sqrt(self.Q_head_size)
        attention_proba = F.softmax(attention_proba.float(), dim=-1).type_as(Q)

        attention_scores = torch.matmul(attention_proba, V_cache)
        
        attention_scores = attention_scores.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        
        output = self.W_O(attention_scores)
        return output
