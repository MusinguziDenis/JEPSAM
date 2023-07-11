from dataloader import get_dataloaders

from losses import get_loss_func
from models import JEPSAM
import numpy as np

import pandas as pd

import os.path as osp
import time

import torch
import torch.optim as optim

from utils.parser import parse_args, load_config


from tqdm import tqdm

def train(
        cfg, 
        model, 
        dataloader, 
        optimizer, 
        criterion, 
        epoch
    ):

    curr_loss = 0.
    
    pbar = tqdm(dataloader, desc=f"Training [Epoch {epoch+1}/{cfg.TRAIN.MAX_EPOCH}]")
    
    for batch_idx, data in enumerate(pbar, 0):
        in_state, goal_state, ad, cmd, ad_lens, cmd_lens = data
        batch_size, _, _, _ = in_state.shape

        ad          = ad.to(cfg.TRAIN.GPU_DEVICE)
        cmd         = cmd.to(cfg.TRAIN.GPU_DEVICE)
        goal_state  = goal_state.to(cfg.TRAIN.GPU_DEVICE)

        # forward
        rec_per_img, goal_img_out, text_out, cmd_out = model(data)

        # loss computation
        try:
            text_out = torch.argmax(text_out, -1)[:, :ad.shape[-1]]

            cmd_len = min(cmd.shape[-1], cmd_out.shape[-1])
            cmd_out = torch.argmax(cmd_out, -1)[:, :cmd_len]
            cmd     = cmd[:, :, :cmd_len] 
            # print()    
            # print(text_out.shape, ad.shape)
            # print(cmd_out.shape, cmd.shape)
            
            loss_img, loss_text, loss_cmd = criterion(
                goal_img_out, 
                text_out, 
                cmd_out, 
                goal_state, 
                ad.squeeze(1), 
                cmd.squeeze(1)
            )
        except RuntimeError:
            print("cmd_out: ", cmd_out.shape)
            print("cmd: ", cmd.shape)
            print("Min len: ", cmd_len)

        mask_text = torch.zeros(text_out.size()).to(cfg.TRAIN.GPU_DEVICE)
        for i in range(batch_size):
            mask_text[i, :ad_lens[i]] = 1

        # mask_text = mask_text.view(-1).to(device)

        mask_cmd = torch.zeros(cmd_out.size()).to(cfg.TRAIN.GPU_DEVICE)

        for i in range(batch_size):
            mask_cmd[i, :ad_lens[i]] = 1

        # mask_cmd = mask_cmd.view(-1).to(cfg.TRAIN.GPU_DEVICE)

        masked_loss_text = torch.sum(
            loss_text * mask_text) / torch.sum(mask_text, 1)
        
        masked_loss_cmd = torch.sum(
            loss_cmd * mask_cmd) / torch.sum(mask_cmd, 1)

        loss_i = torch.mean(loss_img)
        loss_ad = torch.mean(masked_loss_text)
        loss_cmd = torch.mean(masked_loss_cmd)

        loss = loss_i + loss_ad + loss_cmd
        
        loss.backward()

        optimizer.step()

        # Output training stats
        if batch_idx % 50 == 0:
            pbar.set_postfix({
                "Step": f"{batch_idx}/{len(dataloader)}",
                "L_img": f"{loss_i:.4f}",
                "L_ad": f"{loss_ad:.4f}",
                "L_cmd": f"{loss_cmd:.4f}",
                "TrainLoss": f"{loss:.4f}"
            })
            pbar.update()

    pbar.close()

    return loss

def validate(
        cfg, 
        model, 
        dataloader, 
        criterion, 
        epoch
    ):
    pbar = tqdm(
        dataloader, 
        desc=f"Validating [Epoch {epoch+1}/{cfg.TRAIN.MAX_EPOCH}]",
        ncols=100)
    
    for batch_idx, data in enumerate(pbar, 0):
        in_state, goal_state, ad, cmd, ad_lens, cmd_lens = data
        batch_size, _, _, _ = in_state.shape

        ad          = ad.to(cfg.TRAIN.GPU_DEVICE)
        cmd         = cmd.to(cfg.TRAIN.GPU_DEVICE)
        goal_state  = goal_state.to(cfg.TRAIN.GPU_DEVICE)

        # forward
        rec_per_img, goal_img_out, text_out, cmd_out = model(data)

        # loss computation
        text_out = torch.argmax(text_out, -1)[:, :ad.shape[-1]]
        cmd_out = torch.argmax(cmd_out, -1)[:, :cmd.shape[-1]]
        # print()    
        # # print(text_out.shape, ad.shape)
        # print(cmd_out.shape, cmd.shape)

        loss_img, loss_text, loss_cmd = criterion(
            goal_img_out, 
            text_out, 
            cmd_out, 
            goal_state, 
            ad.squeeze(1), 
            cmd.squeeze(1)
        )

        mask_text = torch.zeros(text_out.size()).to(cfg.TRAIN.GPU_DEVICE)
        for i in range(batch_size):
            mask_text[i, :ad_lens[i]] = 1

        # mask_text = mask_text.view(-1).to(device)

        mask_cmd = torch.zeros(cmd_out.size()).to(cfg.TRAIN.GPU_DEVICE)

        for i in range(batch_size):
            mask_cmd[i, :ad_lens[i]] = 1

        # mask_cmd = mask_cmd.view(-1).to(cfg.TRAIN.GPU_DEVICE)

        masked_loss_text = torch.sum(
            loss_text * mask_text) / torch.sum(mask_text, 1)
        
        masked_loss_cmd = torch.sum(
            loss_cmd * mask_cmd) / torch.sum(mask_cmd, 1)

        loss = torch.mean(loss_img) + torch.mean(masked_loss_text) + \
            torch.mean(masked_loss_cmd)  # / batch_size

        break
    pbar.close()


    # Output training stats
    print("Validation results:")
    print('[Epoch %d/%d][Step %d/%d]\tAvg. Loss: %.5f\t'
            % (epoch+1, cfg.TRAIN.MAX_EPOCH, batch_idx, len(dataloader), loss))

    return loss


def run_AEJEPS(args, cfg):
    
    # init
    best_loss = np.inf

    # dataloader = construct_loader(cfg)
    tdf = pd.read_csv(
        osp.join(args.data_path, "updated_train.csv"))

    train_dl, val_dl = get_dataloaders(
        train_df=tdf,
        cfg=cfg
    )

    model = JEPSAM(cfg).to(cfg.TRAIN.GPU_DEVICE)
    loss_type = "cross_entropy"

    criterion = get_loss_func(loss_type)(reduction="none")

    if "adam" in cfg.MODEL.OPTIMIZER.lower():
        optimizer = getattr(optim, cfg.MODEL.OPTIMIZER)(
            params=model.parameters(), 
            lr=float(cfg.MODEL.LEARNING_RATE), 
            betas=(0.5, 0.999)
        )
    elif cfg.MODEL.OPTIMIZER.lower() == "sgd":
        optimizer = getattr(optim, cfg.MODEL.OPTIMIZER)(
            params=model.parameters(), 
            lr=float(cfg.MODEL.LEARNING_RATE), 
            nesterov=True,
            momentum=.9
        )

    ckpt_path = f"{cfg.MODEL.CHECKPOINT_DIR}/AEJEPS_{time.time()}.pth"

    # training loop
    print("Started Autoencoder JEPS training")
    for epoch in range(cfg.TRAIN.MAX_EPOCH):

        train_loss  = train(
            cfg=cfg,
            model=model, 
            dataloader=train_dl, 
            optimizer=optimizer, 
            criterion=criterion,
            epoch=epoch
        )

        val_loss    = validate(        
            cfg=cfg, 
            model=model, 
            dataloader=val_dl, 
            criterion=criterion, 
            epoch=epoch
        )

        if val_loss < best_loss:
            print(f"Loss improvement: from {best_loss:.5f}-->{val_loss:.5f} \nSaving checkpoint to ", ckpt_path)
            torch.save(model.state_dict(), ckpt_path)

            best_loss = val_loss


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()
