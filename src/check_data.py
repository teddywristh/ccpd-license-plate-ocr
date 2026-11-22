from torch.utils.data import DataLoader
from dataset import CCPDDataset, seq2seq_collate
from labels import CLASSES, show_pred
from model import ImageToSequenceModel

TRAIN_DIR = "data/CCPD2020/ccpd_green/train"

train_dataset = CCPDDataset(TRAIN_DIR)
print("Train images:", len(train_dataset))

img, label, path = train_dataset[0]

print("Image:", path)
print("Tensor:", img.shape)
print("Label len:", len(label))
print("Label IDs:", label)
print("Label:", show_pred(label))
# batch
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=seq2seq_collate)
imgs, dec_in, tar, paths = next(iter(train_loader))

print("Batch size:", imgs.shape)
print("Dec input size:", dec_in.shape)
print("Target size:", tar.shape)
print("Dec sample:", dec_in[0].tolist())
print("Target sample:", tar[0].tolist())

model = ImageToSequenceModel(vocab=len(CLASSES) + 3)
logits = model(imgs, dec_in)

print("Output size:", logits.shape)
