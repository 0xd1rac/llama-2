from src.llama.LlamaBuilder import LlamaBuilder
from src.transformer_code_blocks import ModelArgs

WEIGHTS_DIR = "llama-2-7b"
TOKENIZER_PATH = "tokenizers/tokenizer.model"
args = ModelArgs()


lb = LlamaBuilder()

lb.load_weights(weights_dir = WEIGHTS_DIR,
                 tokenizer_path = TOKENIZER_PATH,
                 load_model= True,
                 max_seq_len=args.max_seq_len,
                 max_batch_size= args.max_batch_size,
                 device=args.max_batch_size,
                 model_args=args
                )