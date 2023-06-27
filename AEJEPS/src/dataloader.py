from addict import Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2

import logging
import numpy as np
import os
import os.path as osp
import pandas as pd

from PIL import Image
from pprint import pprint

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from typing import Union, Any, List
from utils.parser import parse_args, load_config

import vocabulary as vocab

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.basicConfig(level="INFO")

class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def tokenize_by_space(
        self, 
        full_string:str
    )->list:
        return full_string.split()


    def idx2token(
        self, 
        i:int
    )->str:
        return self.vocab.REVERSE_TOKENS_MAPPING[i]

    def token2idx(
        self, 
        t:str
    )->int:
        return self.vocab.TOKENS_MAPPING[t]

    def encode(self, input:str):
        tokens = self.tokenize_by_space(input)
        tokens = [self.vocab.SPECIAL_TOKENS[0]] + tokens + [self.vocab.SPECIAL_TOKENS[-1]]
        token_ids = torch.tensor([self.token2idx(t) for t in tokens])
        # enc = {
        #     "tokens": tokens,
        #     "input_ids": torch.tensor(token_ids)
        # }
        return token_ids
    
    def batch_encode(self, input:Union[List, np.ndarray]):
        batch_size = len(input)
        batch_token_ids = [self.encode(inp) for inp in input]
        return batch_token_ids
    
    def decode(self, token_ids:torch.tensor)->list:
        tokens = [self.idx2token(i) for i in token_ids.tolist()]
        return tokens
    
    def batch_decode(self, input:Union[List, np.ndarray]):

        batch_size = len(input)

        batch_tokens = [self.decode(inp) for inp in input]

        return batch_tokens

class SimpleJEPSAMDataset(Dataset):
    def __init__(
        self,
        cfg:Dict,
        vocab=vocab,
        csv:str = None,
        df: pd.DataFrame = None,
        apply_transforms: bool = True,
        task: str = "train"    
    ):
        super().__init__()

        self.vocab = vocab
        self.dataset_directory = osp.join(cfg.DATASET.PATH, "JEPS_data")
        self.tokenizer = SimpleTokenizer(vocab)
        self.cfg = cfg
        
        if df is not None:
            self.dataset_points = df.copy()
        else:
            self.dataset_points = pd.read_csv(csv)
        
        # image transforms
        self.apply_transforms = apply_transforms
        if task == "train":
            tfms = [
                getattr(A, tfms)(**params) for tfms, params in self.cfg.DATASET.TRAIN_TFMS.items()
            ]
        else:
            tfms = [
                getattr(A, tfms)(**params) for tfms, params in self.cfg.DATASET.TEST_TFMS.items()
            ]

        tfms.append(A.Normalize())
        tfms.append(ToTensorV2())
        self.transforms = A.Compose(tfms) if self.apply_transforms else None

    def __len__(self):
        return len(self.dataset_points)

    def __getitem__(self, idx):
        
        data_point = self.dataset_points.iloc[idx]
        
        # visual inputs
        ## in state
        in_state = np.array(Image.open(os.path.join(
            self.dataset_directory, str(data_point.sample_ID), str(data_point.in_state)
        )))

        goal_state = np.array(Image.open(os.path.join(
            self.dataset_directory, str(data_point.sample_ID), str(data_point.goal_state)
        )))
        # apply image treansforms
        if self.apply_transforms:
            # apply transforms
            in_state = self.transforms(image=in_state)["image"]
            goal_state = self.transforms(image=goal_state)["image"]        
        
        # Language modalities
        ## action desc
        action_description = self.tokenizer.encode(input=data_point.action_description)

        ## motor cmd
        motor_command = self.tokenizer.encode(input=data_point.motor_cmd)
        
        
        sample = {
            "sample_id": data_point.sample_ID,
            "in_state": in_state,
            "goal_state": goal_state,
            "action_desc": {
                "raw"   : data_point.action_description,
                "ids"   : action_description.long(),
                "length": data_point.len_action_desc
            },
            "motor_cmd": {
                "raw"   : data_point.motor_cmd,
                "ids"   : motor_command.long(),
                "length": data_point.len_motor_cmd
            }
        }

        return sample

    def collate_fn(self, batch):
        
        # imgs
        batch_input_state = [b["in_state"] for b in batch]
        batch_input_state_stack = torch.stack(batch_input_state)
        
        batch_goal_state = [b["goal_state"] for b in batch]
        batch_goal_state_stack = torch.stack(batch_goal_state)

        # ad
        batch_action_desc = [b["action_desc"]["ids"] for b in batch]
        # print(batch_action_desc)
        batch_action_description = pad_sequence(
            batch_action_desc, 
            batch_first=True, 
            padding_value=self.vocab.TOKENS_MAPPING["[PAD]"]
        ).unsqueeze(1)
        # print(batch_action_description)
        
        batch_action_desc_lens = torch.as_tensor([b["action_desc"]["length"] for b in batch])
        # batch_action_desc_lens_stack = torch.tensor(batch_action_desc_lens)
        # print(batch_action_desc_lens_stack)
        
        #cmd
        batch_motor_commands = [b["motor_cmd"]["ids"] for b in batch]
        batch_motor_commands = pad_sequence(
            batch_motor_commands, 
            batch_first=True, 
            padding_value=self.vocab.TOKENS_MAPPING["[PAD]"]
        ).unsqueeze(1)
        batch_motor_commands_lens = torch.as_tensor([b["motor_cmd"]["length"] for b in batch])
        
        out = (
            batch_input_state_stack, 
            batch_goal_state_stack, 
            batch_action_description, 
            batch_motor_commands, 
            batch_action_desc_lens, 
            batch_motor_commands_lens
        )
        return out
                    

class JEPSAMDataset(Dataset):
    def __init__(
        self,
        cfg=None,
        df: pd.DataFrame = None,
        config_file_path: str = None,
        data_path: str = None,
        apply_transforms: bool = True,
        task: str = "train"
    ):
        super().__init__()

        try:
            args = parse_args()
        except:
            args = None

        if cfg:
            self.cfg = cfg
            self.data_path = cfg.DATASET.PATH
        else:
            if config_file_path is not None:
                self.cfg = load_config(config_file_path=config_file_path)
            else:
                self.cfg = load_config(config_file_path=args.cfg_path)

            # get data path from args
            if data_path is not None:
                self.data_path = data_path
            else:
                if args:
                    self.data_path = args.data_path

        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(
                osp.join(self.data_path, "updated_train.csv"))

        self.images_folder = osp.join(self.data_path, "JEPS_data")

        # image transforms
        self.apply_transforms = apply_transforms
        if task == "train":
            tfms = [
                getattr(A, tfms)(**params) for tfms, params in self.cfg.DATASET.TRAIN_TFMS.items()
            ]
        else:
            tfms = [
                getattr(A, tfms)(**params) for tfms, params in self.cfg.DATASET.TEST_TFMS.items()
            ]

        tfms.append(A.Normalize())
        tfms.append(ToTensorV2())
        self.transforms = A.Compose(tfms) if self.apply_transforms else None

        # self.tokenizer = Tokenizer.from_file(cfg.DATASET.TOKENIZER_PATH)
        try:
            self.tokenizer = PreTrainedTokenizerFast(
                # You can load from the tokenizer file, alternatively
                tokenizer_file=self.cfg.DATASET.TOKENIZER_PATH,
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]",
            )
        except:
            self.tokenizer = PreTrainedTokenizerFast(
                # You can load from the tokenizer file, alternatively
                tokenizer_file=osp.join(self.data_path, "jeps_tokenizer.json"),
                unk_token="[UNK]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                mask_token="[MASK]",
            )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        s = self.df.iloc[index]

        # visual inputs
        in_state = np.array(Image.open(
            osp.join(self.images_folder, str(s.sample_ID), str(s.in_state))
        ))
        goal_state = np.array(Image.open(
            osp.join(self.images_folder, str(s.sample_ID), str(s.goal_state))
        ))

        if self.apply_transforms:
            # apply transforms
            in_state = self.transforms(image=in_state)["image"]
            goal_state = self.transforms(image=goal_state)["image"]

        # text inputs
        # action_enc = self.tokenizer.encode(sequence=s.action_description)
        # cmd_enc = self.tokenizer.encode(sequence=s.motor_cmd)
        action_enc = self.tokenizer(
            text=s.action_description,
            padding="max_length",
            truncation=False,
            max_length=self.cfg.DATASET.ACTION_DESC_MAX_LEN,
            return_tensors="pt"
        )

        cmd_enc = self.tokenizer(
            text=s.motor_cmd,
            padding="max_length",
            truncation=False,
            max_length=self.cfg.DATASET.CMD_MAX_LEN,
            return_tensors="pt"
        )

        sample = {
            "sample_id": s.sample_ID,
            # "in_state": torch.from_numpy(in_state),
            # "goal_state": torch.from_numpy(goal_state),
            "in_state": in_state,
            "goal_state": goal_state,
            "action_desc": {
                "raw"   : s.action_description,
                "ids"   : action_enc.input_ids.long(),
                "length": s.len_action_desc
            },
            "motor_cmd": {
                "raw"   : s.motor_cmd,
                "ids"   : cmd_enc.input_ids.long(),
                "length": s.len_motor_cmd
            }
        }

        return sample


def get_dataloaders(
        cfg, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame = None,
        dataset_module:Dataset=SimpleJEPSAMDataset,
        ) -> tuple:

    val_pct = cfg.DATASET.VALIDATION_PCT

    if val_df is None:
        # 80/20 train/test split
        random_indices = np.random.rand(len(train_df)) < (1-val_pct)

        # Split the DataFrame into train and test sets
        train = train_df[random_indices]
        val = train_df[~random_indices]

    # datasets
    train_ds        =   dataset_module(df=train, cfg=cfg)
    val_ds          =   dataset_module(df=val, cfg=cfg)

    logging.info(
        f"Prepared {len(train_ds)} training samples and {len(val_ds)} validation samples ")
    # data loaders
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        collate_fn=train_ds.collate_fn if dataset_module == SimpleJEPSAMDataset else None
    )

    val_dl = DataLoader(
        dataset=val_ds,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        collate_fn=val_ds.collate_fn if dataset_module == SimpleJEPSAMDataset else None

    )

    return (train_dl, val_dl)


if __name__ == "__main__":

    # ds = JEPSAMDataset()

    # rand_idx = np.random.randint(low=0, high=len(ds))
    # ex = ds[rand_idx]
    # print("Dataset size: ", len(ds))
    # print("="*100)
    # print("ID\t: ", ex["sample_id"])
    # print(">> InState\t: ", ex["in_state"].shape)
    # print(">> GoalState\t: ", ex["goal_state"].shape)
    # print(">> Desc\t:")
    # pprint(ex["action_desc"])
    # print(">> Cmd\t:")
    # pprint(ex["motor_cmd"])
    # print("="*100)

    logging.info("Checking data loading pipeline")
    args = parse_args()
    cfg = load_config()
    tdf = pd.read_csv(
        osp.join(args.data_path, "updated_train.csv"))

    train_dl, val_dl = get_dataloaders(
        train_df=tdf,
        cfg=cfg
    )

    try:
        logging.info("\n>> train data loader")
        for data in train_dl:
            s_id, in_state, goal_state, ad, cmd = data['sample_id'], data[
                'in_state'], data['goal_state'], data['action_desc'], data["motor_cmd"]
            print("In\t\t\t:", in_state.shape)
            print("Goal\t\t\t:", goal_state.shape)
            print("Action desc\t\t:", ad["ids"].shape)
            print("Action desc (len)\t:", ad["length"].shape)

            print("CMD\t\t\t:", cmd["ids"].shape)
            print("CMD(len)\t\t:", cmd["length"].shape)
            break

        logging.info("\n\n>> val data loader")
        for data in val_dl:
            s_id, in_state, goal_state, ad, cmd = data['sample_id'], data[
                'in_state'], data['goal_state'], data['action_desc'], data["motor_cmd"]
            print("In\t\t:", in_state.shape)
            print("Goal\t\t:", goal_state.shape)
            print("Action desc\t:", ad["ids"].shape)
            print("Action desc (len)\t:", ad["length"].shape)

            print("CMD\t\t:", cmd["ids"].shape)
            print("CMD(len)\t\t:", cmd["length"].shape)
            break
    except Exception as e:
        logging.error(e)
        logging.info("\n>> train data loader")
        print(f"# train batches\t\t: {len(train_dl)}")
        for data in train_dl:
            in_state, goal_state, ad, cmd, ad_lens, cmd_lens = data[0], data[1], data[2], data[3], data[4], data[5]
            print("In\t\t\t:", in_state.shape)
            print("Goal\t\t\t:", goal_state.shape)
            print("Action desc\t\t:", ad.shape)
            print("Action desc (len)\t:", ad_lens.shape)
            print("CMD\t\t\t:", cmd.shape)
            print("CMD(len)\t\t:", cmd_lens.shape)
            print()
            # break

        # logging.info("\n>> val data loader")
        # print(f"# validation batches\t: {len(val_dl)}")
        # for data in val_dl:
        #     in_state, goal_state, ad, cmd, ad_lens, cmd_lens = data[0], data[1], data[2], data[3], data[4], data[5]
        #     print("In\t\t\t:", in_state.shape)
        #     print("Goal\t\t\t:", goal_state.shape)
        #     print("Action desc\t\t:", ad.shape)
        #     print("Action desc (len)\t:", ad_lens.shape)
        #     print("CMD\t\t\t:", cmd.shape)
        #     print("CMD(len)\t\t:", cmd_lens.shape)
        #     break