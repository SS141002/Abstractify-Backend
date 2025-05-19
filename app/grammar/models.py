from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

save_path = "./models/coedit"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy-loaded variables
gram_tokenizer = None
gram_model = None

def get_grammar_model():
    global gram_tokenizer, gram_model
    if gram_tokenizer is None or gram_model is None:
        print("🔄 Loading Grammar model...")
        gram_tokenizer = AutoTokenizer.from_pretrained(save_path)
        gram_model = AutoModelForSeq2SeqLM.from_pretrained(save_path)
        gram_model.to(device)
        print("✅ Grammar model loaded.")
    return gram_tokenizer, gram_model

def unload_grammar_model():
    global gram_tokenizer, gram_model
    if gram_model:
        print("🚮 Unloading Grammar model...")
        gram_model.cpu()
        del gram_tokenizer
        del gram_model
        gram_tokenizer = None
        gram_model = None
        torch.cuda.empty_cache()
        print("✅ Grammar model unloaded.")
