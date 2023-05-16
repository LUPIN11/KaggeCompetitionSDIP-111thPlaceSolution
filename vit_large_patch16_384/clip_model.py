# from torch import nn
# from transformers import AutoModel
#
# from config import *
#
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         clip = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
#         self.vision = clip.vision_model
#         self.fc = nn.Linear(1024, 384)
#
#     def forward(self, x):
#         out = self.vision(x)['pooler_output']
#         return self.fc(out)
#
#
# def load_pretrained_model(model):
#     trainable_model_weights = False
#     for name, child in model.named_children():
#         if name == 'vision':
#             for pn, p in child.named_parameters():
#                 if str(18) in pn:  # 如果数据更多就减小这里的数字
#                     trainable_model_weights = True
#                 p.requires_grad = trainable_model_weights
#                 if p.requires_grad:
#                     print(f"{pn} is set to be trainable.")
#     return model
#
#
# model = Model()
# model = load_pretrained_model(model)
# if CONTINUE:
#     model.load_state_dict(torch.load(MODEL_LOAD_PATH))
# model.to(DEVICE)
