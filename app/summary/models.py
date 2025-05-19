from transformers import BartTokenizer, BartForConditionalGeneration
import torch

__save_dir__ = "./models/bart/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy-loaded variables
tokenizer = None
model = None

def get_bart_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("🔄 Loading BART summarization model...")
        tokenizer = BartTokenizer.from_pretrained(__save_dir__)
        model = BartForConditionalGeneration.from_pretrained(__save_dir__).to(device)
        print("✅ BART model loaded.")
    return tokenizer, model

def unload_bart_model():
    global tokenizer, model
    if model:
        print("🚮 Unloading BART model...")
        model.cpu()  # Move to CPU before deleting to free VRAM
        del tokenizer
        del model
        tokenizer = None
        model = None
        torch.cuda.empty_cache()
        print("✅ BART model unloaded.")
