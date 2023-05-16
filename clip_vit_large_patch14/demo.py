import torch
import clip
from PIL import Image
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
# model.save('/root/autodl-tmp/kaggle/PromptPredict/encoder/model.pth')
# preprocess.save(preprocess, '/root/autodl-tmp/kaggle/PromptPredict/encoder/preprocess.pth')
# model = torch.load('/root/autodl-tmp/kaggle/PromptPredict/encoder/model.pth')
# preprocess = torch.load('/root/autodl-tmp/kaggle/PromptPredict/encoder/preprocess.pth')

# model.save('')
# preprocess.save('/root/autodl-tmp/kaggle/PromptPredict/encoder/preprocess')
text = clip.tokenize(["i am a dog"]).to(device)
text_features = model.encode_text(text)[0]
norm = torch.norm(text_features)
print(norm)
# # text = clip.tokenize(["a man is lying along aside a drive road"]).to(device)
# text = clip.tokenize(["a kind of 'I'm a robot now, I'm not going to get killed by robots.' It's only when you consider that we're still robots, and we're still talking about free will, and you think about it, what would happen if we"]).to(device)
#
# # 加载图像
# image = Image.open("/root/autodl-tmp/kaggle/PromptPredict/images/0/100.png").convert('RGB')
# image_input = preprocess(image).unsqueeze(0).to(device) # 转换为模型的输入格式
# image_features = model.encode_image(image_input)
# print(image_features.shape)
# with torch.no_grad():
#     image_features = model.encode_image(image_input)
#     text_features = model.encode_text(text)
#     print(image_features.shape)
#     print(text_features.shape)
#
#     logits_per_image, logits_per_text = model(image_input, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:", probs)
# print(F.cosine_similarity(image_features, text_features))

