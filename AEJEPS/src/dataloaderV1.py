
import logging
import numpy as np
import os
import os.path as osp
import pandas as pd
from pprint import pprint
from skimage.io import imread
import sys
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from utils.parser import parse_args, load_config

import vocabulary as vocab

# Logger config
logging.basicConfig(level="INFO")


def tokenize_by_space(full_string):
    return full_string.split()


class AllModalitiesDataset(Dataset):
    def __init__(
        self,
        csv,
        dataset_directory,
        df: pd.DataFrame = None
    ):

        if df is not None:
            self.dataset_points = df.copy()
        else:
            self.dataset_points = pd.read_csv(csv)

        self.dataset_directory = dataset_directory
        self.action_description_vocabulary = vocab.ACTION_DESCRIPTION
        self.motor_commands_vocabulary = vocab.MOTOR_COMMANDS

    def __len__(self):
        return len(self.dataset_points)

    def __getitem__(self, idx):
        data_point = self.dataset_points.sample_ID[idx]
        # in state
        in_state = self.dataset_points.in_state[idx]
        input_state_image = imread(os.path.join(
            self.dataset_directory, str(data_point), str(in_state)
        ))
        # out state
        goal_state = self.dataset_points.goal_state[idx]
        goal_state_image = imread(os.path.join(
            self.dataset_directory, str(data_point), str(goal_state)
        ))
        # action desc
        action_description = tokenize_by_space(
            self.dataset_points.action_description[idx]
        )
        action_description = [self.action_description_vocabulary.index(
            token) for token in action_description]
        action_description = torch.tensor(action_description)

        # motor cmd
        motor_command = tokenize_by_space(
            self.dataset_points.motor_cmd[idx])
        motor_command = [self.motor_commands_vocabulary.index(
            token) for token in motor_command]
        motor_command = torch.tensor(motor_command)

        return input_state_image, goal_state_image, action_description, motor_command

    def collate_fn(self, batch):
        batch_input_state = [input_state for input_state, _, _, _ in batch]
        batch_goal_state = [goal_state for _, goal_state, _, _ in batch]
        batch_action_description = [ad for _, _, ad, _ in batch]
        batch_motor_commands = [motor_cmd for _, _, _, motor_cmd in batch]

        batch_action_description = pad_sequence(
            batch_action_description, padding_value=-1)

        batch_motor_commands = pad_sequence(
            batch_motor_commands, padding_value=-1)

        return torch.as_tensor(batch_input_state), torch.as_tensor(batch_goal_state), torch.as_tensor(batch_action_description), torch.as_tensor(batch_motor_commands)


def get_dataloaders(args, cfg, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> tuple:

    csv_path = osp.join(args.data_path, "updated_train.csv")
    dataset_path = osp.join(args.data_path, "JEPS_data")

    val_pct = cfg.DATASET.VALIDATION_PCT

    if val_df is None:
        # 80/20 train/test split
        random_indices = np.random.rand(len(train_df)) < (1-val_pct)

        # Split the DataFrame into train and test sets
        train = train_df[random_indices].reset_index(drop=True)
        val = train_df[~random_indices].reset_index(drop=True)

    # datasets
    train_ds = AllModalitiesDataset(
        df=train,
        csv=csv_path,
        dataset_directory=dataset_path
    )
    val_ds = AllModalitiesDataset(
        df=val,
        csv=csv_path,
        dataset_directory=dataset_path
    )

    logging.info(
        f"Prepared {len(train_ds)} training samples and {len(val_ds)} validation samples ")

    # rand_idx = np.random.randint(low=0, high=len(train_ds))
    # ex = train_ds[rand_idx]

    # print("Dataset size: ", len(train_ds))
    # print("="*100)
    # print(">> ID\t: ", rand_idx)
    # print(">> InState\t: ", ex[0].shape)
    # print(">> GoalState\t: ", ex[1].shape)
    # print(">> Desc\t:")
    # pprint(ex[2])
    # print(">> Cmd\t:")
    # pprint(ex[3])
    # print("="*100)

    # data loaders
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        collate_fn=train_ds.collate_fn
    )

    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        collate_fn=val_ds.collate_fn

    )

    return (train_dl, val_dl)


if __name__ == "__main__":
    args = parse_args()

    csv_path = osp.join(args.data_path, "updated_train.csv")
    dataset_path = osp.join(args.data_path, "JEPS_data")

    # test dataset I/O
    # ds = AllModalitiesDataset(csv=csv_path, dataset_directory=dataset_path)

    # rand_idx = np.random.randint(low=0, high=len(ds))
    # ex = ds[rand_idx]
    # print("Dataset size: ", len(ds))
    # print("="*100)
    # print(">> InState\t: ", ex[0].shape)
    # print(">> GoalState\t: ", ex[1].shape)
    # print(">> Desc\t:")
    # pprint(ex[2])
    # print(">> Cmd\t:")
    # pprint(ex[3])
    # print("="*100)

    # test dataloader I/O
    logging.info("Checking data loading pipeline")
    cfg = load_config()
    tdf = pd.read_csv(osp.join(csv_path))

    train_dl, val_dl = get_dataloaders(
        train_df=tdf,
        args=args,
        cfg=cfg
    )
    logging.info("\n>> train data loader")
    print(f"# train batches\t: {len(train_dl)}")
    for data in train_dl:
        in_state, goal_state, ad, cmd = data[0], data[1], data[2], data[3]
        print("In\t\t:", in_state.shape)
        print("Goal\t\t:", goal_state.shape)
        print("Action desc\t:", ad.shape)
        print("CMD\t\t:", cmd.shape)

        break

    logging.info("\n\n>> val data loader")
    print(f"# validation batches\t: {len(val_dl)}")
    for data in val_dl:
        in_state, goal_state, ad, cmd = data[0], data[1], data[2], data[3]
        print("In\t\t:", in_state.shape)
        print("Goal\t\t:", goal_state.shape)
        print("Action desc\t:", ad.shape)
        print("CMD\t\t:", cmd.shape)

        break
