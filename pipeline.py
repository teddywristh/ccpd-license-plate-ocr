import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

PROVINCES = [
    "Anhui", "Shanghai", "Tianjin", "Chongqing", "Hebei", "Shanxi",
    "Inner Mongolia", "Liaoning", "Jilin", "Heilongjiang", "Jiangsu",
    "Zhejiang", "Beijing", "Fujian", "Jiangxi", "Shandong", "Henan",
    "Hubei", "Hunan", "Guangdong", "Guangxi", "Hainan", "Sichuan",
    "Guizhou", "Yunnan", "Tibet", "Shaanxi", "Gansu", "Qinghai",
    "Ningxia", "Xinjiang", "Police", "Student", "No-character",
]
PROV_TOKENS = [f"P{i:02d}" for i in range(31)] + ["POLICE", "STUDENT", "NONE"]
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
ALPHA = LETTERS + ["NONE"]
ADS = LETTERS + list("0123456789") + ["NONE"]
CLASSES = PROV_TOKENS + LETTERS + list("0123456789") + ["NONE"]
PAD, BOS, EOS = 0, 1, 2
T2I = {x: i + 3 for i, x in enumerate(CLASSES)}
I2T = {i: x for x, i in T2I.items()}

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "data/CCPD2020/ccpd_green/train"
VAL_DIR = "data/CCPD2020/ccpd_green/val"
TEST_DIR = "data/CCPD2020/ccpd_green/test"
CKPT = "checkpoints/best_model.pt"

def enc_label(path):
    f = Path(path).stem.split("-")
    if len(f) != 7: raise ValueError(f"Invalid CCPD filename: {path}")
    ids = [int(x) for x in f[4].split("_")]
    if len(ids) != 8: raise ValueError(f"Invalid label: {path}")
    tok = [PROV_TOKENS[ids[0]], ALPHA[ids[1]], *[ADS[i] for i in ids[2:]]]
    return [T2I[x] for x in tok if x != "NONE"]

def show_pred(ids):
    tok = [I2T[i] for i in ids if i not in (PAD, BOS, EOS)]
    if tok and tok[0].startswith("P") and tok[0][1:].isdigit(): return PROVINCES[int(tok[0][1:])] + " " + "".join(tok[1:])
    return "".join(tok)

def par_bbox(path):
    f = Path(path).stem.split("-")
    if len(f) != 7: raise ValueError(f"Invalid CCPD filename: {path}")
    a, b = f[2].split("_")
    x1, y1 = map(int, a.split("&")); x2, y2 = map(int, b.split("&"))
    return x1, y1, x2, y2

class Plates(Dataset):
    def __init__(self, root, h=48, w=192):
        self.root, self.h, self.w = Path(root), h, w
        self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.paths: raise FileNotFoundError(f"No images: {self.root.resolve()}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]; img = cv2.imread(str(path))

        if img is None: raise ValueError(f"Cannot read image: {path}")
        x1, y1, x2, y2 = par_bbox(path); ih, iw = img.shape[:2]
        x1, x2 = max(0, min(x1, iw)), max(0, min(x2, iw)); y1, y2 = max(0, min(y1, ih)), max(0, min(y2, ih))

        crop = img[y1:y2, x1:x2]

        if crop.size == 0: raise ValueError(f"Empty crop: {path}")
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY); ch, cw = crop.shape
        s = min(self.w / cw, self.h / ch); nw, nh = max(1, int(cw * s)), max(1, int(ch * s))

        crop = cv2.resize(crop, (nw, nh)); canvas = np.zeros((self.h, self.w), dtype=np.uint8)
        left, top = (self.w - nw) // 2, (self.h - nh) // 2; canvas[top:top + nh, left:left + nw] = crop

        x = torch.from_numpy(canvas.astype(np.float32) / 255).unsqueeze(0)
        return x, enc_label(path), str(path)
    
def collate(batch):
    imgs, labs, paths = zip(*batch); imgs = torch.stack(imgs); n = max(len(x) for x in labs) + 1
    dins, tars = [], []
    for lab in labs:
        di, tar = [BOS] + lab, lab + [EOS]; pad = n - len(di)
        dins.append(di + [PAD] * pad); tars.append(tar + [PAD] * pad)
    return imgs, torch.tensor(dins), torch.tensor(tars), paths

class Enc(nn.Module):
    def __init__(self, dim=256):
        super().__init__(); self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)))
        self.projection = nn.Linear(256 * 3, dim)

    def forward(self, imgs):
        x = self.cnn(imgs); bs, c, h, w = x.shape; x = x.permute(0, 3, 1, 2).contiguous().view(bs, w, c * h)
        return torch.tanh(self.projection(x))
    
class Attn(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__(); self.encoder_projection = nn.Linear(enc_dim, attn_dim); 
        self.decoder_projection = nn.Linear(dec_dim, attn_dim); 
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, enc, h):
        score = self.score(torch.tanh(self.encoder_projection(enc) + self.decoder_projection(h).unsqueeze(1))).squeeze(-1); 
        a = F.softmax(score, dim=1)
        return torch.bmm(a.unsqueeze(1), enc).squeeze(1), a
    
class Net(nn.Module):
    def __init__(self, vocab):
        super().__init__(); self.encoder = Enc(); 
        self.embedding = nn.Embedding(vocab, 128, padding_idx=PAD); 
        self.attention = Attn(256, 256, 128)
        self.init_hidden = nn.Linear(256, 256); 
        self.init_cell = nn.Linear(256, 256); 
        self.lstm_cell = nn.LSTMCell(128 + 256, 256); 
        self.output_layer = nn.Linear(256 + 256, vocab)

    def forward(self, imgs, dins):
        enc = self.encoder(imgs); 
        pool = enc.mean(1); 
        h = torch.tanh(self.init_hidden(pool)); 
        c = torch.tanh(self.init_cell(pool)); 
        outs = []
        for i in range(dins.size(1)):
            e = self.embedding(dins[:, i]); 
            ctx, _ = self.attention(enc, h); 
            h, c = self.lstm_cell(torch.cat([e, ctx], 1), (h, c)); 
            outs.append(self.output_layer(torch.cat([h, ctx], 1)))
        return torch.stack(outs, 1)
    
    @torch.no_grad()
    def dec(self, imgs, max_len=9):
        enc = self.encoder(imgs); pool = enc.mean(1); 
        h = torch.tanh(self.init_hidden(pool)); 
        c = torch.tanh(self.init_cell(pool)); 
        bs = imgs.size(0); 
        cur = torch.full((bs,), BOS, dtype=torch.long, device=imgs.device); 
        done = torch.zeros(bs, dtype=torch.bool, device=imgs.device); out = []

        for _ in range(max_len):
            e = self.embedding(cur); ctx, _ = self.attention(enc, h); h, c = self.lstm_cell(torch.cat([e, ctx], 1), (h, c)); nxt = self.output_layer(torch.cat([h, ctx], 1)).argmax(1); nxt = torch.where(done, torch.full_like(nxt, EOS), nxt); out.append(nxt); done |= nxt == EOS; cur = nxt
            if done.all(): break
        return torch.stack(out, 1)
    
def train(epochs=30, bs=8):
    Path("checkpoints").mkdir(exist_ok=True); 
    tr, va = Plates(TRAIN_DIR), Plates(VAL_DIR); 
    tl = DataLoader(tr, bs, shuffle=True, collate_fn=collate); 
    vl = DataLoader(va, bs, collate_fn=collate); net = Net(len(CLASSES) + 3).to(DEV); 
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD); 
    opt = torch.optim.Adam(net.parameters(), 1e-3); 
    best = float("inf")
    print("Device:", DEV, "Train:", len(tr), "Val:", len(va))

    for ep in range(1, epochs + 1):
        net.train(); total = 0
        for imgs, dins, tars, _ in tl:
            imgs, dins, tars = imgs.to(DEV), dins.to(DEV), tars.to(DEV); 
            opt.zero_grad(); 
            loss = loss_fn(net(imgs, dins).flatten(0, 1), tars.flatten()); 
            loss.backward(); 
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5); 
            opt.step(); 
            total += loss.item() * imgs.size(0)

        net.eval(); val = 0
        with torch.no_grad():
            for imgs, dins, tars, _ in vl:
                imgs, dins, tars = imgs.to(DEV), dins.to(DEV), tars.to(DEV); 
                val += loss_fn(net(imgs, dins).flatten(0, 1), tars.flatten()).item() * imgs.size(0)
        tr_loss, va_loss = total / len(tr), val / len(va); 
        print(f"Epoch {ep:02d}/{epochs} | train: {tr_loss:.4f} | val: {va_loss:.4f}")
        if va_loss < best: best = va_loss; 
        torch.save({"epoch": ep, "model_state_dict": net.state_dict(), "vocab_size": len(CLASSES) + 3, "val_loss": va_loss}, CKPT); 
        print("Saved best")

def load_net():
    ckpt = torch.load(CKPT, map_location=DEV, weights_only=False); net = Net(ckpt["vocab_size"]).to(DEV); net.load_state_dict(ckpt["model_state_dict"]); net.eval(); return net, ckpt
def strip(ids):
    out = []
    for i in ids:
        if int(i) in (PAD, EOS): break
        out.append(int(i))
    return out
def dist(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        now = [i]
        for j, y in enumerate(b, 1): now.append(min(row[j] + 1, now[j - 1] + 1, row[j - 1] + (x != y)))
        row = now
    return row[-1]
def evaluate():
    net, ckpt = load_net(); ds = Plates(TEST_DIR); ld = DataLoader(ds, 8, collate_fn=collate); 
    exact = chars = nchar = edits = 0; 
    probs, onehots, rows = [], [], []
    with torch.no_grad():
        for imgs, dins, tars, paths in ld:
            imgs, dins, tars = imgs.to(DEV), dins.to(DEV), tars.to(DEV); logit = net(imgs, dins); 
            p = F.softmax(logit, -1); ok = (tars != PAD) & (tars != EOS); 
            probs.append(p[ok].cpu().numpy()); 
            onehots.append(F.one_hot(tars[ok], ckpt["vocab_size"]).float().cpu().numpy()); 
            pred = net.dec(imgs)
            for i, path in enumerate(paths):
                true, out = strip(tars[i].cpu()), strip(pred[i].cpu()); 
                exact += int(true == out); 
                chars += sum(a == b for a, b in zip(true, out)); 
                nchar += len(true); edits += dist(true, out); 
                rows.append([path, show_pred(true), show_pred(out), true == out])
    yp, yt = np.concatenate(probs).ravel(), np.concatenate(onehots).ravel(); 
    err = yt - yp; 
    sse = np.sum(err ** 2); 
    sst = np.sum((yt - yt.mean()) ** 2); 
    r2 = 1 - sse / sst; print("Epoch:", ckpt["epoch"], "Test:", len(ds)); 

    print("Exact:", exact / len(ds), "Char:", chars / nchar, "CER:", edits / nchar); 
    print("MAE:", np.mean(np.abs(err)), "RMSE:", np.sqrt(np.mean(err ** 2)), "R2:", r2, "NSE:", r2)

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/test_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["image", "target", "prediction", "exact_match"]); w.writerows(rows)
def pred_one(path):
    net, ckpt = load_net(); 
    ds = Plates(Path(path).parent); 
    i = next(i for i, p in enumerate(ds.paths) if p.resolve() == Path(path).resolve()); 
    img, _, _ = ds[i]; out = net.dec(img.unsqueeze(0).to(DEV))[0].tolist(); 
    
    print("Epoch:", ckpt["epoch"], "Image:", path, "Plate:", show_pred(out))
def main():
    p = argparse.ArgumentParser(); 
    p.add_argument("--mode", choices=["train", "eval", "predict"], default="eval"); 
    p.add_argument("--image"); 
    p.add_argument("--epochs", type=int, default=30); a = p.parse_args()
    
    if a.mode == "train": train(a.epochs)
    elif a.mode == "eval": evaluate()
    elif a.image: pred_one(a.image)
    else: p.error("--image is required for predict")
if __name__ == "__main__": main()
