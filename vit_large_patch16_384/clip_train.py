# import os
# from tqdm import tqdm
#
# from torch import nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from timm.utils import AverageMeter
#
# from config import *
# from functions import cosine_similarity
# from clip_model import model
# from clip_dataloader import train_dataloader, valid_dataloader
#
# current_directory = os.getcwd()
# print("cwd:", current_directory)
# print("device:", DEVICE)
#
# # optimizer
# # optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# # optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
#
# # criterion
# criterion = nn.CosineEmbeddingLoss(margin=0.95)
#
# # scheduler
# # scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_dataloader),
# #                               eta_min=MIN_LR)
#
# # run
# best_score = -1.0
#
# try:
#     for epoch in range(NUM_EPOCHS):
#         # train
#         train_meters = {
#             'loss': AverageMeter(),
#             'cos': AverageMeter(),
#         }
#         model.train()
#         bar = tqdm(train_dataloader, leave=False, ncols=BAR_LENGTH)
#         for X, y in bar:
#             X, y = X.to(DEVICE), y.to(DEVICE)
#             optimizer.zero_grad()
#             X_out = model(X)
#             # backward
#             target = torch.ones(X.size(0)).to(DEVICE)
#             loss = criterion(X_out, y, target)
#             loss.backward()
#             # step
#             optimizer.step()
#             # scheduler.step()
#             # update
#             trn_loss = loss.item()
#             trn_cos = cosine_similarity(
#                 X_out.detach().cpu().numpy(),
#                 y.detach().cpu().numpy()
#             )
#             train_meters['loss'].update(trn_loss, n=X.size(0))
#             train_meters['cos'].update(trn_cos, n=X.size(0))
#             bar.set_postfix(trn_los=f'{train_meters["loss"].avg:.4f}', trn_cos=f'{train_meters["cos"].avg:.4f}')
#         print(
#             f"Epoch {epoch + 1:d} / train_loss={train_meters['loss'].avg:.4f}, train_cos={train_meters['cos'].avg:.4f}")
#
#         # valid
#         val_meters = {
#             'loss': AverageMeter(),
#             'cos': AverageMeter(),
#         }
#         model.eval()
#         bar = tqdm(valid_dataloader, leave=False, ncols=BAR_LENGTH)
#         for X, y in bar:
#             X, y = X.to(DEVICE), y.to(DEVICE)
#             with torch.no_grad():
#                 X_out = model(X)
#                 target = torch.ones(X.size(0)).to(DEVICE)
#                 loss = criterion(X_out, y, target)
#                 val_loss = loss.item()
#                 val_cos = cosine_similarity(
#                     X_out.detach().cpu().numpy(),
#                     y.detach().cpu().numpy()
#                 )
#             val_meters['loss'].update(val_loss, n=X.size(0))
#             val_meters['cos'].update(val_cos, n=X.size(0))
#             bar.set_postfix(val_los=f'{val_meters["loss"].avg:.4f}', val_cos=f'{val_meters["cos"].avg:.4f}')
#         print(f'Epoch {epoch + 1} / val_loss={val_meters["loss"].avg:.4f}, val_cos={val_meters["cos"].avg:.4f}')
#
#         # save best model
#         # if val_meters['cos'].avg > best_score:
#         #     best_score = val_meters['cos'].avg
#         #     torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/{SAVED_NAME}')
#         # save
#         torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/'+f"{MODEL_NAME}|{epoch}|{BATCH_SIZE}|{LR}|{val_meters['cos'].avg:.4f}.pth")
# finally:
#     torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/LAST.pth')
