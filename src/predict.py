import argparse
import os
from pathlib import Path

# OpenMP fix for Windows/Anaconda.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from dataset import CCPDDataset
from labels import CLASSES, show_pred
from model import ImageToSequenceModel


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_one(path):
    path = Path(path)
    ds = CCPDDataset(path.parent)

    for i, item in enumerate(ds.img_paths):
        if item.resolve() == path.resolve():
            image, _, _ = ds[i]
            return image.unsqueeze(0).to(DEV)

    raise FileNotFoundError(f"Image not found: {path}")

def load(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    model = ImageToSequenceModel(vocab=ckpt.get("vocab_size", len(CLASSES) + 3)).to(DEV)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def pred(path, ckpt_path):
    model, ckpt = load(ckpt_path)
    image = get_one(path)

    ids = model.greedy_dec(image, max_length=9)[0].tolist()
    text = show_pred(ids)

    print("Checkpoint epoch:", ckpt.get("epoch", "?"))
    print("Image:", path)
    print("Pred IDs:", ids)
    print("Plate:", text)


def main():
    parser = argparse.ArgumentParser(description="Predict one CCPD plate")
    parser.add_argument("image", help="CCPD image path")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Checkpoint path",
    )
    args = parser.parse_args()
    pred(args.image, args.checkpoint)


if __name__ == "__main__":
    main()
