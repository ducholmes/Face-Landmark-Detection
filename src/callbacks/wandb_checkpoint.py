import wandb
import shutil
from lightning import Callback
from pathlib import Path


class WandbCheckpointCallback(Callback):
    def __init__(self, upload_last_only: bool = True):
        super().__init__()
        self.upload_last_only = upload_last_only

    def on_validation_end(self, trainer, pl_module):
        if wandb.run is None:
            return

        if trainer.sanity_checking:
            return

        checkpoint_dir = Path(trainer.default_root_dir) / "checkpoints"

        if self.upload_last_only:
            last_ckpt = checkpoint_dir / "last.ckpt"
            if last_ckpt.exists():
                self._upload(last_ckpt, trainer)
            else:
                print(f"last.ckpt not found in {checkpoint_dir}")
        else:
            for ckpt_file in checkpoint_dir.glob("*.ckpt"):
                self._upload(ckpt_file, trainer)

    def _upload(self, ckpt_path: Path, trainer):
        dest = Path(trainer.default_root_dir) / ckpt_path.name
        shutil.copy(str(ckpt_path), str(dest))
        wandb.save(
            str(dest),
            base_path=trainer.default_root_dir,
            policy="now"
        )
        print(f"Uploaded {ckpt_path.name} to W&B Files tab")