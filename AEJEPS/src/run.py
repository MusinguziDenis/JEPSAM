from utils.parser import parse_args, load_config

from train import run_AEJEPS


def main():
    args = parse_args()
    cfg = load_config()

    if cfg.RUN.MODE == "train_aejeps":
        run_AEJEPS(args=args, cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.RUN.MODE} training not implemented")


if __name__ == '__main__':
    main()
