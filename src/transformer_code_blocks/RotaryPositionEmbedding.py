from .common_imports import *


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, seq_len: int, device: str) -> None:
        super().__init__()
        self.dim = head_dim
        assert self.dim % 2 == 0, "head_dim must be divisible by 2"

        # Calculate the rotation frequencies for positional embeddings
        theta_numerator = torch.arange(0, self.dim, 2, dtype=torch.float32)
        theta = 1.0 / torch.pow(10000, theta_numerator / self.dim).to(device)

        # Generate frequency values for positional embeddings
        m = torch.arange(seq_len, dtype=torch.float32).to(device)
        freqs = torch.outer(m, theta).float()

        # Convert frequency values to complex numbers (polar form)
        self.freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        # self.register_buffer("freqs_complex", self.freqs_complex)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        assert dim == self.dim, "dim must be equal to self.dim"

        # Reshape the input into a complex tensor for rotational operations
        # (B, SeqLen, H, Head_Dim) -> (B, SeqLen, H, Head_Dim // 2)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        # Extract rotational frequencies for the given sequence length and start position
        # (SeqLen, Head_Dim // 2) -> (1, SeqLen, 1, Head_Dim // 2)
        freq_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)

        # Apply rotational transformation to the input using frequency values
        # (B, SeqLen, H, Head_Dim // 2) * (1, SeqLen, 1, Head_Dim // 2) -> (B, SeqLen, H, Head_Dim // 2)
        x_rotated = x_complex * freq_complex

        # Convert the rotated complex tensor back to real-valued tensor
        # (B, SeqLen, H, Head_Dim // 2) -> (B, SeqLen, H , Head_Dim // 2, 2)
        x_out = torch.view_as_real(x_rotated)

        # Reshape to match the original input shape
        # (B, SeqLen, H , Head_Dim // 2, 2) -> (B, SeqLen, H, Head_Dim)
        x_out = x_out.reshape(*x.shape)

        return x_out.type_as(x).to(x.device)