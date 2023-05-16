# from PIL import Image
#
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from sentence_transformers import SentenceTransformer
#
# from config import *
#
# from transformers import AutoProcessor
#
# clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
#
#
# class MyDataset(Dataset):
#     def __init__(self, df, processor=clip_processor):
#         self.df = df
#         self.processor = processor
#
#     def __len__(self): return len(self.df)
#
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         image = Image.open(row['filepath'])
#         image = self.processor(image)
#         prompt = row['prompt']
#         return image, prompt
#
#
# class MyCollator:
#     def __init__(self):
#         self.st_model = SentenceTransformer(ST_MODEL_PATH, device='cpu')
#
#     def __call__(self, batch):
#         images, prompts = zip(*batch)
#         images = torch.stack(images)
#         prompt_embeddings = self.st_model.encode(prompts, show_progress_bar=False, convert_to_tensor=True)
#         return images, prompt_embeddings
#
#
# def dataloaders(train_df, vaild_df):
#     trn_dataset = MyDataset(train_df)
#     val_dataset = MyDataset(vaild_df)
#     collator = MyCollator()
#     train_dataloader = DataLoader(
#         dataset=trn_dataset,
#         shuffle=True,
#         batch_size=BATCH_SIZE,
#         pin_memory=True,
#         num_workers=6,
#         drop_last=False,
#         collate_fn=collator
#     )
#     valid_dataloader = DataLoader(
#         dataset=val_dataset,
#         shuffle=False,
#         batch_size=BATCH_SIZE,
#         pin_memory=True,
#         num_workers=6,
#         drop_last=False,
#         collate_fn=collator
#     )
#     return train_dataloader, valid_dataloader
#
#
# train_df = pd.read_csv(TRAIN_CSV_PATH)
# valid_df = pd.read_csv(VALID_CSV_PATH)
# train_dataloader, valid_dataloader = dataloaders(train_df, valid_df)
