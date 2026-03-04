import wandb
from pathlib import Path

def download_checkpoint_from_wandb(
    run_path: str,
    checkpoint_name: str,
    download_dir: str = "checkpoints/resume"
) -> str:
    api = wandb.Api()
    run = api.run(run_path)

    checkpoint_file = None
    for file in run.files():
        if checkpoint_name in file.name:
            checkpoint_file = file
            break

    if checkpoint_file is None:
        raise FileNotFoundError(
            f"Không tìm thấy {checkpoint_name} trong run {run_path}"
        )

    Path(download_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_file.download(root=download_dir, replace=True)

    local_path = str(list(Path(download_dir).glob(f"**/{checkpoint_name}"))[0])
    print(f"Downloaded checkpoint to: {local_path}")

    return local_path

def download_checkpoints_from_files(
    run_paths: list,
    checkpoint_name: str = "ema.ckpt",
    download_dir: str = "checkpoints/ensemble"
) -> list:
    api = wandb.Api()
    local_paths = []

    for run_path in run_paths:
        run = api.run(run_path)

        checkpoint_file = None
        for file in run.files():
            if checkpoint_name in file.name:
                checkpoint_file = file
                break

        if checkpoint_file is None:
            raise FileNotFoundError(
                f"Không tìm thấy {checkpoint_name} trong run {run_path}"
            )

        run_id = run_path.split("/")[-1]
        run_download_dir = f"{download_dir}/{run_id}"
        Path(run_download_dir).mkdir(parents=True, exist_ok=True)

        checkpoint_file.download(root=run_download_dir, replace=True)

        ckpt_file = list(Path(run_download_dir).glob(f"**/{checkpoint_name}"))[0]
        local_paths.append(str(ckpt_file))
        print(f"Downloaded {run_path} → {ckpt_file}")

    return local_paths