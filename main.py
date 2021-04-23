from pathlib import Path
from model import HatefulMemesModel
from pathlib import Path
import zipfile
import pandas as pd


if __name__ == '__main__':
    data_dir = Path.cwd() / 'meme_data'
    img_tar_path = data_dir / "img.zip"
    img_path = data_dir / 'img'
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    test_path = data_dir / "test.jsonl"

    # Extract images if there is need
    if not img_path.exists():
        with zipfile.ZipFile(img_tar_path, 'r') as zip_ref:
            zip_ref.extractall(img_path)

    hparams = {
        # Required hparams
        "train_path": train_path,
        "dev_path": dev_path,
        "img_dir": data_dir,

        # Optional hparams
        "embedding_dim": 150,
        "language_feature_dim": 300,
        "vision_feature_dim": 300,
        "fusion_output_size": 256,
        "output_path": "model-outputs",
        "dev_limit": None,
        "lr": 0.00005,
        "max_epochs": 10,
        "n_gpu": 1,
        "batch_size": 4,
        # allows us to "simulate" having larger batches
        "accumulate_grad_batches": 16,
        "early_stop_patience": 3,
    }

    hateful_memes_model = HatefulMemesModel(hparams=hparams)
    hateful_memes_model.fit()