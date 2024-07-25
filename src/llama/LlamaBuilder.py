import torch
import torch.nn as nn
from src.transformer_code_blocks import ModelArgs
from .Transformer import Transformer
from .Llama import Llama
import time
from tqdm import tqdm
from pathlib import Path
from sentencepiece import SentencePieceProcessor


class LlamaBuilder:
    @staticmethod
    def load_weights(weights_dir: str,
                     tokenizer_path: str,
                     load_model: bool,
                     max_seq_len: int,
                     max_batch_size: int,
                     device: str,
                     model_args: ModelArgs
                    ):
        
        prev_time = time.time()

        if load_model:
            """
            1.  Convert weights_dir into a Path object, this allows you 
                to use pathlib methods on it
            
            2. glob("*.pth") searches weights_dir for all files ending
                with .pth extension

            3. sort the list of found '.pth' files
            """
            checkpoint_files = sorted(Path(weights_dir).glob("*.pth"))
            assert len(checkpoint_files) > 0, "No weights found"

            target_file = checkpoint_files[0]

            print(f"Loading checkpoint from {target_file}")

            checkpoint = torch.load(target_file, map_location="cpu")

            print("Loaded checkpoint in {:.2f}s".format(time.time()-prev_time))

            prev_time = time.time()
        
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)


        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        
        model_args.vocab_size = tokenizer.vocab_size()
        model = Transformer(model_args).to(device)

        if load_model:
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded model in {time.time() - prev_time}s")
            
        return Llama(model, tokenizer, model_args)