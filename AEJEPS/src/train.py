from tqdm import tqdm
import sys
from utils.parser import parse_args, load_config
import torch.optim as optim
import torch
from losses import get_loss_func
import os.path as osp
import pandas as pd
import numpy as np
from models import JEPSAM
from dataloader import get_dataloaders

import logging
logging.basicConfig(level="INFO")


def train(
    cfg,
    model,
    dataloader,
    optimizer,
    criterion,
    epoch
):

    curr_loss = 0.

    pbar = tqdm(
        dataloader, desc=f"Training [Epoch {epoch+1}/{cfg.TRAIN.MAX_EPOCH}]")

    model.train()

    for batch_idx, data in enumerate(pbar, 0):
        in_state, goal_state, ad, cmd, ad_lens, cmd_lens = data
        batch_size, _, _, _ = in_state.shape

        in_state = in_state.to(cfg.TRAIN.GPU_DEVICE)
        ad = ad.to(cfg.TRAIN.GPU_DEVICE)
        cmd = cmd.to(cfg.TRAIN.GPU_DEVICE)
        goal_state = goal_state.to(cfg.TRAIN.GPU_DEVICE)

        # forward pass
        rec_per_img, goal_img_out, ad_out, cmd_out = model(data)

        # loss computation
        try:
            # AD
            ad_len  = min(ad.shape[-1], ad_out.shape[-2])
            ad_     = ad[:, :, 1:].reshape(-1)  # skipping [EOS] token
            # reshape model outputs for CELoss too
            ad_out  = ad_out[:, :ad_len, :]
            ad_out_ = ad_out[:, 1:].reshape(-1, ad_out.shape[2])
            # print(f"ad_: {ad_.shape} - ad_out_: {ad_out_.shape}")

            # CMD
            cmd_len = min(cmd.shape[-1], cmd_out.shape[-2])
            cmd_     = cmd[:, :, 1:].reshape(-1)  # skipping [EOS] token
            # reshape model outputs for CELoss too
            cmd_out  = cmd_out[:, :cmd_len, :]
            cmd_out_ = cmd_out[:, 1:].reshape(-1, cmd_out.shape[2])            
            # print()
            # print(ad_out.shape, ad.shape)
            # print(cmd_out.shape, cmd.shape)

            loss_img, loss_ad, loss_cmd = criterion(
                rec_per_img,
                goal_img_out,
                ad_out_,
                cmd_out_,
                in_state,
                goal_state,
                ad_,
                cmd_,
                ignore_idx=cfg.DATASET.PAD,
                debug=False
            )
        except RuntimeError as e:
            print("ad_out: ", ad_out.shape)
            print("ad: ", ad.shape)
            print("Min len: ", ad_len)

            print("cmd_out: ", cmd_out.shape)
            print("cmd: ", cmd.shape)
            print("Min len: ", cmd_len)

            logging.error(e)

            sys.exit()
   

        loss        = loss_img + loss_ad + loss_cmd
        curr_loss   += loss.item()

        # backward pass
        loss.backward()
        optimizer.step()

        # Output training stats
        if batch_idx % 50 == 0:
            pbar.set_postfix({
                "Step": f"{batch_idx}/{len(dataloader)}",
                "L_img": f"{loss_img:.4f}",
                "L_ad": f"{loss_ad:.4f}",
                "L_cmd": f"{loss_cmd:.4f}",
                "TrainLoss": f"{loss:.4f}"
            })
            pbar.update()

    pbar.close()

    train_loss = curr_loss / len(dataloader)

    return train_loss


def validate(
    cfg,
    model,
    dataloader,
    criterion,
    epoch
):
    model.eval()

    pbar = tqdm(
        dataloader,
        desc=f"Validating [Epoch {epoch+1}/{cfg.TRAIN.MAX_EPOCH}]",
        ncols=100
        )

    val_loss = 0.0

    for batch_idx, data in enumerate(pbar, 0):
        in_state, goal_state, ad, cmd, ad_lens, cmd_lens = data
        batch_size, _, _, _ = in_state.shape

        # in_state = in_state.to(cfg.TRAIN.GPU_DEVICE)
        # ad = ad.to(cfg.TRAIN.GPU_DEVICE)
        # cmd = cmd.to(cfg.TRAIN.GPU_DEVICE)
        # goal_state = goal_state.to(cfg.TRAIN.GPU_DEVICE)

        # forward
        rec_per_img, goal_img_out, ad_out, cmd_out = model(
            data,
            mode="test"
        )

        # loss computation
        try:
            # AD
            ad_len  = min(ad.shape[-1], ad_out.shape[-2])
            # print(f"\nad: {ad.shape} - ad_out: {ad_out.shape} - min: {ad_len}")
            ad_     = ad[:, :, 1:].reshape(-1)  # skipping [EOS] token
            # reshape model outputs for CELoss too
            ad_out  = ad_out[:, :ad_len, :]
            ad_out_ = ad_out[:, 1:].reshape(-1, ad_out.shape[2])
            # print(f"\nad_: {ad_.shape} - ad_out_: {ad_out_.shape}")

            # CMD
            cmd_len = min(cmd.shape[-1], cmd_out.shape[-2])
            cmd_     = cmd[:, :, 1:].reshape(-1)  # skipping [EOS] token
            # reshape model outputs for CELoss too
            cmd_out  = cmd_out[:, :cmd_len, :]
            cmd_out_ = cmd_out[:, 1:].reshape(-1, cmd_out.shape[2])            
            # print()
            # print(ad_out.shape, ad.shape)
            # print(cmd_out.shape, cmd.shape)

            loss_img, loss_ad, loss_cmd = criterion(
                rec_per_img.cpu().detach(),
                goal_img_out.cpu().detach(),
                ad_out_.cpu().detach(),
                cmd_out_.cpu().detach(),
                in_state,
                goal_state,
                ad_,
                cmd_,
                ignore_idx=cfg.DATASET.PAD,
                debug=False
            )

            loss = loss_img + loss_ad + loss_cmd

            val_loss += loss

        except RuntimeError as ex:
            print()
            logging.error(ex)
            print("ad_out: ", ad_out.shape)
            print("ad: ", ad.shape)
            print("Min len: ", ad_len)

            print()
            print("cmd_out: ", cmd_out.shape)
            print("cmd: ", cmd.shape)
            print("Min len: ", cmd_len)

            sys.exit()
        
        
    pbar.close()
    val_loss = val_loss / len(dataloader)
    # Output training stats
    print("Validation results:")
    print('[Epoch %d/%d]\tAvg. Loss: %.5f\t'
            % (epoch+1, cfg.TRAIN.MAX_EPOCH, val_loss))        
                
    return val_loss


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
    loss_type = cfg.MODEL.LOSS_TYPE

    criterion = get_loss_func(loss_name=loss_type)()

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

    # ckpt_path = f"{cfg.MODEL.CHECKPOINT_DIR}/AEJEPS_{time.time()}.pth"
    ckpt_path = f"{cfg.MODEL.CHECKPOINT_DIR}jepsam_best.bin"

    # training loop
    logging.info("Training strating now...")
    for epoch in range(cfg.TRAIN.MAX_EPOCH):

        train_loss = train(
            cfg=cfg,
            model=model,
            dataloader=train_dl,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch
        )

        val_loss = validate(
            cfg=cfg,
            model=model,
            dataloader=val_dl,
            criterion=criterion,
            epoch=epoch
        )


        if val_loss < best_loss:
            print(
                f"Loss improvement: from {best_loss:.5f}-->{val_loss:.5f} \nSaving checkpoint to ", ckpt_path)
            
            torch.save({
                "mode_state_dict": model.state_dict(),
                "best_score": val_loss
            }, ckpt_path)

            # step tf rate
            model.decoder._step_tf_rate()

            best_loss = val_loss

    logging.info("Completed training")

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()
