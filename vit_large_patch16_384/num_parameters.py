import torch
import timm

model = timm.create_model('vit_base_patch16_224', pretrained=False)

total_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.numel())
        total_params += param.numel()

print("Total number of trainable parameters:", total_params)
