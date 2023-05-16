from transformers import AutoProcessor, AutoModel
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_processor.save_pretrained("./offline_model/auto_processor/")
clip = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
clip.save_pretrained("./offline_model/auto_model/")
