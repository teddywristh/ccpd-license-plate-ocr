from pathlib import Path
from labels import PAD_ID, BOS_ID, EOS_ID
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from labels import enc_label, par_bbox

class CCPDDataset(Dataset):
    def __init__(self, img_root, h=48, w=192):
        self.img_root = Path(img_root)
        self.h = h
        self.w = w
        self.img_paths = sorted(path for path in self.img_root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.img_paths:
            raise FileNotFoundError(f"Not found {self.img_root.resolve()}")
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, index):
        path = self.img_paths[index]
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot read img: {path}")

        x1, y1, x2, y2 = par_bbox(path)
        ih, iw = img.shape[:2]
        x1 = max(0, min(x1, iw))
        x2 = max(0, min(x2, iw))
        y1 = max(0, min(y1, ih))
        y2 = max(0, min(y2, ih))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"Blank: {path}")
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_h, crop_w = crop.shape
        scale = min(self.w / crop_w, self.h / crop_h)

        new_w = max(1, int(crop_w * scale))
        new_h = max(1, int(crop_h * scale))
        crop = cv2.resize(crop, (new_w, new_h))

        canvas = np.zeros((self.h, self.w), dtype=np.uint8)

        left = (self.w - new_w)//2
        top = (self.h - new_h)//2
        canvas[top:top + new_h, left:left + new_w] = crop

        x = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)
        label = enc_label(path)

        return x, label, str(path)
def seq2seq_collate(batch):
    imgs, labels, paths = zip(*batch)
    imgs = torch.stack(imgs)
    max_label_length = max(len(label) for label in labels)
    seq_len = max_label_length + 1
    dec_ins = []
    tars = []

    for label in labels:
        dec_in = [BOS_ID] + label
        tar = label + [EOS_ID]
        pad_len = seq_len - len(dec_in)

        dec_in += [PAD_ID] * pad_len
        tar += [PAD_ID] * pad_len

        dec_ins.append(dec_in)
        tars.append(tar)
    return imgs, torch.tensor(dec_ins, dtype=torch.long), torch.tensor(tars, dtype=torch.long), paths
