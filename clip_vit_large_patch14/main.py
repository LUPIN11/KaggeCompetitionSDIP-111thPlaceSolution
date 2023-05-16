from tqdm import tqdm
import torch
import torch.optim as optim
from timm.utils import AverageMeter
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import *
from model import model
from dataloader import train_dataloader, valid_dataloader
from functions import *

if __name__ == "__main__":
    # best_cos = -1

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    optimizer.zero_grad()
    # criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        train_meters = {
            # 'loss': AverageMeter(),
            'cos': AverageMeter(),
        }
        val_meters = {
            'cos': AverageMeter(),
        }

        train_bar = tqdm(train_dataloader, leave=False)
        model.train()
        for images, ebs in train_bar:
            images, ebs = images.to(device), ebs.to(device)
            pred = model(images)

            loss = cosine_similarity_loss(pred, ebs)
            # loss = criterion(pred, ebs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # train_meters['loss'].update(loss.item(), n=images.size(0))
            # loss = cosine_similarity_loss(pred, ebs)
            train_meters['cos'].update(-loss, n=images.size(0))
            train_bar.set_postfix(trn_cos=f'{train_meters["cos"].avg:.4f}')
        print(f"Epoch {epoch + 1:d} / train_cos={train_meters['cos'].avg:.4f}")

        valid_bar = tqdm(valid_dataloader, leave=False)
        model.eval()
        with torch.no_grad():
            for images, ebs in valid_bar:
                images, ebs = images.to(device), ebs.to(device)
                pred = model(images)
                loss = cosine_similarity_loss(pred, ebs)
                val_meters['cos'].update(-loss, n=images.size(0))
                valid_bar.set_postfix(val_cos=f'{val_meters["cos"].avg:.4f}')
        print(f"Epoch {epoch + 1:d} / valid_cos={val_meters['cos'].avg:.4f}")


        # if val_meters["cos"].avg > best_cos:
        #     best_cos = val_meters['cos'].avg
        torch.save(model.state_dict(), f'./ckpt/{epoch}|{val_meters["cos"].avg:.4f}.pth')
        # print(f"current model saved")

        scheduler.step()