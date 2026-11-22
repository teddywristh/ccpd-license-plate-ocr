import os

# Tránh lỗi xung đột nhiều bản OpenMP trong môi trường Windows/Anaconda.
# Đặt trước khi import torch hoặc OpenCV.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import logging

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset import CCPDDataset, seq2seq_collate
from labels import CLASSES, PAD_ID
from model import ImageToSequenceModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "data/CCPD2020/ccpd_green/train"
VAL_DIR = "data/CCPD2020/ccpd_green/val"

BATCH_SIZE = 32 if DEVICE.type == "cuda" else 8
EPOCHS = 30
LR = 1e-3

Path("checkpoints").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("checkpoints/training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def run_epoch(model, loader, loss_fn, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    loss_sum = 0.0
    n_img = 0

    for imgs, dec_in, tar, _ in loader:
        imgs = imgs.to(DEVICE)
        dec_in = dec_in.to(DEVICE)
        tar = tar.to(DEVICE)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(imgs, dec_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tar.reshape(-1))

            if train_mode:
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        n_img += bs

    return loss_sum / n_img


def main():
    train_dataset = CCPDDataset(TRAIN_DIR)
    val_dataset = CCPDDataset(VAL_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=seq2seq_collate, pin_memory=(DEVICE.type == "cuda"))

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=seq2seq_collate, pin_memory=(DEVICE.type == "cuda"))

    vocab_size = len(CLASSES) + 3
    model = ImageToSequenceModel(vocab=vocab_size).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")

    logger.info("Device: %s", DEVICE)
    logger.info("Train images: %s", len(train_dataset))
    logger.info("Val images: %s", len(val_dataset))
    logger.info("Classes: %s", vocab_size)

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn)

        logger.info(
            "Epoch %02d/%d | train loss: %.4f | val loss: %.4f",
            epoch,
            EPOCHS,
            train_loss,
            val_loss,
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "val_loss": val_loss,
                },
                "checkpoints/best_model.pt",
            )
            logger.info("Saved best model.")


if __name__ == "__main__":
    main()
