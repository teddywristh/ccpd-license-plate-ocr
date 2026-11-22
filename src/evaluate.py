import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import CCPDDataset, seq2seq_collate
from labels import EOS_ID, PAD_ID, show_pred
from model import ImageToSequenceModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "data/CCPD2020/ccpd_green/test"
CHECKPOINT_PATH = "checkpoints/best_model.pt"
BATCH_SIZE = 8

def strip_special_tokens(ids):
    out = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id in (PAD_ID, EOS_ID):
            break
        out.append(token_id)
    return out

def edit_dist(a, b):
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]

def main():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    model = ImageToSequenceModel(vocab=ckpt["vocab_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = CCPDDataset(TEST_DIR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=seq2seq_collate)

    exact_correct = 0
    character_correct = 0
    character_total = 0
    edit_total = 0
    target_character_total = 0
    all_probabilities = []
    all_one_hot_targets = []
    prediction_rows = []

    with torch.inference_mode():
        for imgs, dec_in, tar, paths in loader:
            imgs = imgs.to(DEVICE)
            dec_in = dec_in.to(DEVICE)
            tar = tar.to(DEVICE)

            logits = model(imgs, dec_in)
            probs = F.softmax(logits, dim=-1)

            ok = (tar != PAD_ID) & (tar != EOS_ID)
            all_probabilities.append(probs[ok].cpu().numpy())
            all_one_hot_targets.append(
                F.one_hot(
                    tar[ok],
                    num_classes=ckpt["vocab_size"],
                ).float().cpu().numpy()
            )

            pred = model.greedy_dec(imgs, max_length=9)

            for i, path in enumerate(paths):
                true = strip_special_tokens(tar[i].cpu().tolist())
                out = strip_special_tokens(pred[i].cpu().tolist())
                exact_correct += int(out == true)
                character_correct += sum(
                    out[j] == true[j]
                    for j in range(min(len(true), len(out)))
                )
                character_total += len(true)
                edit_total += edit_dist(true, out)
                target_character_total += len(true)
                prediction_rows.append({"image": path, "target": show_pred(true), "prediction": show_pred(out), "exact_match": out == true})

    n = len(ds)
    exact_acc = exact_correct / n
    char_acc = character_correct / character_total
    cer = edit_total / target_character_total
    y_p = np.concatenate(all_probabilities).ravel()
    y_t = np.concatenate(all_one_hot_targets).ravel()
    err = y_t - y_p
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    sse = np.sum(err ** 2)
    sst = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - sse / sst if sst else float("nan")
    nse = r2

    print(f"Checkpoint epoch: {ckpt['epoch']}")
    print(f"Test images: {n}")
    print(f"Exact Plate Accuracy: {exact_acc:.4f}")
    print(f"Character Accuracy:   {char_acc:.4f}")
    print(f"CER:                  {cer:.4f}")
    print(f"MAE (one-hot probs):  {mae:.4f}")
    print(f"RMSE (one-hot probs): {rmse:.4f}")
    print(f"R² (one-hot probs):   {r2:.4f}")
    print(f"NSE (one-hot probs):  {nse:.4f}")

    metrics_path = Path("outputs/evaluation_metrics.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value", "note"])
        writer.writerow(["Exact Plate Accuracy", exact_acc, "plate"])
        writer.writerow(["Character Accuracy", char_acc, "char"])
        writer.writerow(["CER", cer, "edit"])
        writer.writerow(["MAE", mae, "prob-onehot"])
        writer.writerow(["RMSE", rmse, "prob-onehot"])
        writer.writerow(["R2", r2, "prob-onehot"])
        writer.writerow(["NSE", nse, "prob-onehot"])
    print("Saved metrics:", metrics_path)

    output_path = Path("outputs/test_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "target", "prediction", "exact_match"])
        writer.writeheader()
        writer.writerows(prediction_rows)
    print("Saved predictions:", output_path)

if __name__ == "__main__":
    main()
