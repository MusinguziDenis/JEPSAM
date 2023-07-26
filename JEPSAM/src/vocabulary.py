###############################
# Author: Arisema M. & Cédric M.
# Date: Summer 2023
# 
# Comments: the mapping will 
# only consider the unique 
# tokens since both lists
# contain identical tokens
###############################



import logging
logging.basicConfig(level="INFO")

import os
import os.path as osp

from pprint import pprint
import sys

from tqdm import tqdm

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
# sys.path.append(project_root)

# Action description tokens
ACTION_DESCRIPTION = [
    'forwards', 
    ':POT', 
    ':BLUE-METAL-PLATE', 
    ':SPOON', 
    'move', 
    ':SHOE', 
    'in', 
    ':FORK', 
    'put', 
    ':BOTTLE', 
    ':GLASSES', 
    ':SPATULA', 
    'to', 
    'the', 
    ':KNIFE', 
    'top', 
    ':CEREAL', 
    ':WEISSWURST', 
    ':BREAKFAST-CEREAL', 
    ':GLOVE', 
    ':BUTTERMILK', 
    ':RED-METAL-PLATE', 
    'behind', 
    'right', 
    ':MUG', 
    ':CAP', 
    ':BOWL', 
    'of', 
    'on', 
    ':MONDAMIN', 
    'shift', 
    'left', 
    'front', 
    ':CUBE', 
    ':MILK', 
    ':PLATE', 
    ':CUP', 
    'backwards'
    ]

# motor commands tokens
MOTOR_COMMANDS = [
    'POSE-4', 
    'POSE-8', 
    ':POT', 
    ':BLUE-METAL-PLATE', 
    'POSE-10', 
    'RED', 
    ':SPOON', 
    ':SHOE', 
    ':FORK', 
    ':BOTTLE', 
    ':GLASSES', 
    ':SPATULA', 
    'POSE-11', 
    'POSE-15', 
    'POSE-5', 
    'POSE-12', 
    'POSE-3', 
    'POSE-14', 
    'NIL', 
    'POSE-9', 
    ':KNIFE', 
    'POSE-2', 
    ':CEREAL', 
    ':WEISSWURST', 
    ':BREAKFAST-CEREAL', 
    ':GLOVE', 
    ':CUBE', 
    'GREEN', 
    ':BUTTERMILK', 
    ':RED-METAL-PLATE', 
    "#'*leftward-transformation*", 
    "#'*forward-transformation*", 
    "#'*backward-transformation*", 
    'BLUE', 
    ':MUG', 
    ':CAP', 
    ':BOWL', 
    ':MONDAMIN', 
    'POSE-6', 
    "#'*rightward-transformation*", 
    'POSE-13', 
    'POSE-7', 
    ':MILK', 
    'POSE-1', 
    ':PLATE', 
    ':CUP'
    ]

SPECIAL_TOKENS                          = ["[SOS]", "[PAD]", "[UNK]", "[EOS]"] 
ALL_TOKENS                              = set(
                                                SPECIAL_TOKENS+ 
                                                ACTION_DESCRIPTION+ 
                                                MOTOR_COMMANDS
                                            )


def load_vocab():
    from config import Config

    level           = Config.DATASET["TOKEN_LEVEL"]
    file_path       = osp.join(Config.DATASET["PATH"], f"vocab_{level}.txt")

    try:
        with open(file_path, "r") as v:
            ALL_TOKENS = v.readlines()

        ALL_TOKENS = [tok.strip() for tok in ALL_TOKENS]

        TOKENS_MAPPING = {t:i for i, t in enumerate(ALL_TOKENS)}
        REVERSE_TOKENS_MAPPING = {i:t for i,t in enumerate(ALL_TOKENS)}  

        return TOKENS_MAPPING, REVERSE_TOKENS_MAPPING
    except Exception as e:
        logging.error(e)


def create_vocab():
    from config import Config
    level           = Config.DATASET["TOKEN_LEVEL"]

    logging.info(f"Vocab file does not exist...creating one")
    file_path = osp.join(Config.DATASET["PATH"], f"vocab_{level}.txt")
    
    if level=="W":
        with open(file_path, "w") as v:
            for t in tqdm(sorted(ALL_TOKENS), desc="Creating W-level vocab"):
                v.write(t+'\n')
    else:
        ctoks = [" "]
        for t in tqdm(sorted(ALL_TOKENS), desc="Creating C-level vocab"):
            for c in t:
                if c not in ctoks:
                    ctoks.append(c)        
        # save tokens
        with open(file_path, "w") as v:
            for c in sorted(ctoks):
                v.write(c+'\n')



if __name__ == '__main__':

    # print(len(ALL_TOKENS))

    print("#tokens in total\t: ", len(ACTION_DESCRIPTION)+len(MOTOR_COMMANDS))
    print("# action tokens\t\t: ", len(ACTION_DESCRIPTION))
    print("# motor tokens\t\t: ", len(MOTOR_COMMANDS))

    try:
        logging.info("Try block")
        TOKENS_MAPPING, REVERSE_TOKENS_MAPPING  = load_vocab() 
    except Exception as e:
        print("Except block")
        logging.error(e)
        create_vocab()


