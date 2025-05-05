from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

save_dir_trocr = "./models/trocr/"

trocr_processor = TrOCRProcessor.from_pretrained(save_dir_trocr)
trocr_model = VisionEncoderDecoderModel.from_pretrained(save_dir_trocr)

device = "cuda" if torch.cuda.is_available() else "cpu"
trocr_model.to(device)