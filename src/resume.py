import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.wandb_utils import download_checkpoint_from_wandb
from src.train import train
from src.utils import RankedLogger, extras

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    extras(cfg)

    log.info("Downloading checkpoint from W&B...")
    ckpt_path = download_checkpoint_from_wandb(
        run_path=cfg.wandb_resume.run_path,
        checkpoint_name=cfg.wandb_resume.checkpoint_name,
        download_dir=cfg.wandb_resume.download_dir
    )

    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.ckpt_path = ckpt_path

    train(cfg)

if __name__ == "__main__":
    main()