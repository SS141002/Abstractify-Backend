from transformers import TrOCRProcessor, VisionEncoderDecoderModel

save_dir_trocr = "./models/trocr-base-handwritten/"

trocr_processor = TrOCRProcessor.from_pretrained(save_dir_trocr)
trocr_model = VisionEncoderDecoderModel.from_pretrained(save_dir_trocr)
