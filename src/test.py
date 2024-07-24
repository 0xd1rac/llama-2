from GroupQueryAttention import GroupQueryAttention
from ModelArgs import ModelArgs
import torch
args = ModelArgs()
gqa = GroupQueryAttention(args)
batch_size = args.max_batch_size
seq_len = args.max_batch_size
x = torch.rand(batch_size, seq_len, args.d_model)

model = GroupQueryAttention(args)

output = model(x)

expected_output_size = (batch_size, seq_len, args.d_model)
assert output.size() == expected_output_size, f"Output size {output.size()} does not match expected size {expected_output_size}"
print("Test passed! Output size matches expected size.")
