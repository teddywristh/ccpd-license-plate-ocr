import torch
import torch.nn as nn
import torch.nn.functional as F
from labels import BOS_ID, EOS_ID, PAD_ID

class CNNEncoder(nn.Module):
    def __init__(self, enc_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.projection = nn.Linear(256 * 3, enc_dim)
    def fw(self, imgs):
        features = self.cnn(imgs)
        bs, channels, height, width = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()
        features = features.view(bs, width, channels * height)
        return torch.tanh(self.proj(features))
    @property
    def proj(self): return self.projection
    def forward(self, imgs):
        return self.fw(imgs)
class AdditiveAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.encoder_projection = nn.Linear(enc_dim, attn_dim)
        self.decoder_projection = nn.Linear(dec_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)
    @property
    def enc_proj(self): return self.encoder_projection
    @property
    def dec_proj(self): return self.decoder_projection
    def fw(self, enc, h_dec):
        scores = self.score(torch.tanh(self.enc_proj(enc) + self.dec_proj(h_dec).unsqueeze(1))).squeeze(-1)
        attn = F.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), enc).squeeze(1)
        return context, attn
    def forward(self, enc, h_dec):
        return self.fw(enc, h_dec)
class ImageToSequenceModel(nn.Module):
    def __init__(
        self,
        vocab,
        enc_dim=256,
        emb_dim=128,
        dec_dim=256,
        attn_dim=128,
    ):
        super().__init__()
        self.encoder = CNNEncoder(enc_dim=enc_dim)
        self.embedding = nn.Embedding(
            num_embeddings=vocab,
            embedding_dim=emb_dim,
            padding_idx=PAD_ID,
        )
        self.attention = AdditiveAttention(
            enc_dim=enc_dim,
            dec_dim=dec_dim,
            attn_dim=attn_dim,
        )
        self.init_hidden = nn.Linear(enc_dim, dec_dim)
        self.init_cell = nn.Linear(enc_dim, dec_dim)
        self.lstm_cell = nn.LSTMCell(
            input_size=emb_dim + enc_dim,
            hidden_size=dec_dim,
        )
        self.output_layer = nn.Linear(dec_dim + enc_dim, vocab)
    @property
    def init_h(self): return self.init_hidden
    @property
    def init_c(self): return self.init_cell
    @property
    def lstm(self): return self.lstm_cell
    @property
    def out(self): return self.output_layer
    def fw(self, imgs, dec_in):
        enc = self.encoder(imgs)
        pooled = enc.mean(dim=1)
        hidden = torch.tanh(self.init_h(pooled))
        cell = torch.tanh(self.init_c(pooled))
        outs = []

        for step in range(dec_in.size(1)):
            embedded = self.embedding(dec_in[:, step])
            context, _ = self.attention(enc, hidden)
            hidden, cell = self.lstm(torch.cat([embedded, context], dim=1), (hidden, cell))
            logits = self.out(torch.cat([hidden, context], dim=1))
            outs.append(logits)

        return torch.stack(outs, dim=1)

    def forward(self, imgs, dec_in):
        return self.fw(imgs, dec_in)
    @torch.no_grad()
    def greedy_dec(self, imgs, max_length=9):
        self.eval()

        enc = self.encoder(imgs)
        pooled = enc.mean(dim=1)
        hidden = torch.tanh(self.init_h(pooled))
        cell = torch.tanh(self.init_c(pooled))

        bs = imgs.size(0)
        cur = torch.full((bs,), BOS_ID, dtype=torch.long, device=imgs.device)
        finished = torch.zeros(bs, dtype=torch.bool, device=imgs.device)
        preds = []

        for _ in range(max_length):
            embedded = self.embedding(cur)
            context, _ = self.attention(enc, hidden)
            hidden, cell = self.lstm(torch.cat([embedded, context], dim=1), (hidden, cell))
            logits = self.out(torch.cat([hidden, context], dim=1))
            nxt = logits.argmax(dim=1)
            nxt = torch.where(finished, torch.full_like(nxt, EOS_ID), nxt)
            preds.append(nxt)
            finished = finished | (nxt == EOS_ID)
            cur = nxt

            if finished.all():
                break

        return torch.stack(preds, dim=1)

    def greedy_decode(self, imgs, max_length=9): return self.greedy_dec(imgs, max_length)
