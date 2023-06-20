from addict import Dict
import argparse
import yaml


def load_config(config_file_path: str = "configs/aejeps_cfg.yaml") -> Dict:
    """
    
    Parameters
    ----------
    config_file_path : str
        The path to the configuration file to be loaded

    Returns
    -------
    A Dict object with the settings accessible as attributes of this object.
    """
    with open(config_file_path) as cfg_file:
        cfg_dict = yaml.safe_load(cfg_file)

        # print(cfg_dict)
    return Dict(cfg_dict)


def parse_args():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description=("JEPS memory implementations.")
    )

    parser.add_argument(
        "--cfg",
        dest="cfg_path",
        help="Path to the config file",
        default="configs/aejeps_cfg.yaml",
        type=str,
    )

    parser.add_argument(
        "--data_path",
        dest="data_path",
        help="Path to data folder",
        default=cfg.DATASET.PATH,
        type=str,
    )

    return parser.parse_args()


if __name__ == '__main__':
    cfg_dict = load_config()
