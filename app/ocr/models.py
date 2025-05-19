from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

save_dir_trocr = "./models/trocr/"

device = "cuda" if torch.cuda.is_available() else "cpu"

trocr_processor = None
trocr_model = None

def get_trocr_model():
    global trocr_processor, trocr_model
    if trocr_processor is None or trocr_model is None:
        print("🔄 Loading TrOCR model...")
        trocr_processor = TrOCRProcessor.from_pretrained(save_dir_trocr)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(save_dir_trocr)
        trocr_model.to(device)
        print("✅ TrOCR model loaded.")
    return trocr_processor, trocr_model

def unload_trocr_model():
    global trocr_processor, trocr_model
    if trocr_model:
        print("🚮 Unloading TrOCR model...")
        trocr_model.cpu()  # Move model to CPU before deleting
        del trocr_processor
        del trocr_model
        trocr_processor = None
        trocr_model = None
        torch.cuda.empty_cache()
        print("✅ Model unloaded.")
