from dataloader import get_dataloaders

from losses import get_loss_func
from models import AutoencoderJEPS

import pandas as pd

import os.path as osp
import time

import torch
import torch.optim as optim

from utils.parser import parse_args, load_config


def train(args, cfg):
    pass


def validate(args, cfg):
    pass


def run_AEJEPS(args, cfg):
    num_epochs = cfg.TRAIN.MAX_EPOCH
    # dataloader = construct_loader(cfg)
    tdf = pd.read_csv(
        osp.join(args.data_path, "updated_train.csv"))

    train_dl, val_dl = get_dataloaders(
        train_df=tdf,
        args=args,
        cfg=cfg
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoencoderJEPS(cfg).to(device)
    loss_type = "aejeps_loss"

    criterion = get_loss_func(loss_type)(reduction="none")

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.MODEL.LEARNING_RATE, betas=(0.5, 0.999))

    print("Started Autoencoder JEPS training")
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for batch_idx, data in enumerate(train_dl, 0):
            per_img, goal_img, text, commands, lengths_text, lengths_cmd = data
            per_img = per_img.to(device)
            goal_img = goal_img.to(device)
            text = text.to(device)
            commands = commands.to(device)

            batch_size = per_img.shape[0]

            train_text, text_target = text[:, :-1], text[:, 1:]
            train_cmd, cmd_target = commands[:, :-1, :], commands[:, 1:, :]

            goal_img_out, text_out, cmd_out = model(
                per_img, goal_img, train_text, train_cmd, lengths_text, lengths_cmd)

            if loss_type == "aejeps_loss":
                text_out = torch.argmax(text_out, 2)
                cmd_out = torch.argmax(cmd_out, 2)
                cmd_target = torch.argmax(cmd_target, 2)
            loss_img, loss_text, loss_cmd = criterion(
                goal_img_out, text_out, cmd_out, goal_img, text_target, cmd_target)

            mask_text = torch.zeros(text_out.size()).to(device)
            for i in range(batch_size):
                mask_text[i, :lengths_text[i]] = 1

            # mask_text = mask_text.view(-1).to(device)

            mask_cmd = torch.zeros(cmd_out.size()).to(device)

            for i in range(batch_size):
                mask_cmd[i, :lengths_text[i]] = 1

            # mask_cmd = mask_cmd.view(-1).to(device)

            masked_loss_text = torch.sum(
                loss_text * mask_text) / torch.sum(mask_text, 1)
            masked_loss_cmd = torch.sum(
                loss_cmd * mask_cmd) / torch.sum(mask_cmd, 1)

            loss = torch.mean(loss_img) + torch.mean(masked_loss_text) + \
                torch.mean(masked_loss_cmd)  # / batch_size
            loss.backward()

            optimizer.step()

            # Output training stats
            if batch_idx % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f\t'
                      % (epoch, num_epochs, batch_idx, len(train_dl), loss))

    ckpt_path = f"{cfg.MODEL.CHECKPOINT_DIR}/AEJEPS_{time.time()}.pth"
    print("Saving checkpoint to ", ckpt_path)
    torch.save(model, ckpt_path)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()
