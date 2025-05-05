from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

save_path = "./models/coedit"


gram_tokenizer = AutoTokenizer.from_pretrained(save_path)
gram_model = AutoModelForSeq2SeqLM.from_pretrained(save_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
gram_model.to(device)

