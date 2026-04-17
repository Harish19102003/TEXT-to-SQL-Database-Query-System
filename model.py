import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torch.utils.data import Dataset
import math
from dataset import dataset, pad_idx, text_vocab, sql_vocab
from config import d_model, n_heads, n_layers, d_ff, dropout, max_seq_len

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embedding(nn.Module):
    def __init__(self,input_dim:int,d_model: int,):
        super().__init__()
        self.embedding = nn.Embedding(input_dim,d_model)
        self.scale     = math.sqrt(d_model)
    def forward(self,x):
        return self.embedding(x) * self.scale
    
# ── 1. Positional Encoding ──────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Adds position information to embeddings.
    Transformer has no recurrence so it needs to know token order explicitly.

    sin/cos waves of different frequencies:
      pos 0: [sin(0), cos(0), sin(0), cos(0), ...]
      pos 1: [sin(1), cos(1), sin(0.1), cos(0.1), ...]
      pos 2: [sin(2), cos(2), sin(0.2), cos(0.2), ...]
    Each position gets a unique fingerprint.
    """
    def __init__(self, d_model, dropout, max_len=200):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pe: [max_len, d_model]
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # div_term: [d_model/2] — different frequency per dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        pe = pe.unsqueeze(0)   # [1, max_len, d_model] — broadcast over batch
        self.register_buffer('pe', pe)   # not a parameter, but saved with model

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.shape[1], :] #type: ignore
        return self.dropout(x)
    
def make_masks( src, trg, pad_idx):
        """
        Create all masks needed for transformer forward pass.

        src_key_padding_mask: tells encoder to ignore <pad> tokens
        trg_mask:             prevents decoder from seeing future tokens
        trg_key_padding_mask: tells decoder to ignore <pad> in target
        """
        # True where token is <pad> — these positions get -inf in attention
        src_key_padding_mask = (src == pad_idx)
        # [batch, src_len]

        trg_key_padding_mask = (trg == pad_idx)
        # [batch, trg_len]

        trg_len = trg.shape[1]
        # causal mask — upper triangle is True (blocked)
        # position i can only attend to positions 0..i
        trg_mask = torch.triu(
            torch.ones(trg_len, trg_len, device=trg.device),
            diagonal=1
        ).bool()
        # [trg_len, trg_len]
        #  False True  True  ...
        #  False False True  ...
        #  False False False ...

        return src_key_padding_mask, trg_mask, trg_key_padding_mask

class Transformer(pl.LightningModule):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 max_len: int = 200,
                 dataset=dataset,
                 pad_idx: int = 0):
        super().__init__()
        self.dataset = dataset
        self.pad_idx = pad_idx
        self.src_embedding = nn.Embedding(input_dim, d_model)
        self.trg_embedding = nn.Embedding(output_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)

        # FIX 2: removed dead self.encoder / self.decoder — only keep self.transformer
        self.transformer = nn.Transformer(d_model, n_heads, n_layers, n_layers,
                                          d_ff, dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, output_dim)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, src, trg):
        # FIX 1: removed internal trg[:, :-1] slice — caller owns the slicing
        src_pad_mask, trg_mask, trg_pad_mask = make_masks(src, trg, self.pad_idx)
        src = self.pos_enc(self.src_embedding(src))
        trg = self.pos_enc(self.trg_embedding(trg))
        output = self.transformer(src, trg,
                                  tgt_mask=trg_mask,
                                  src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=trg_pad_mask)
        return self.fc_out(output)

    def configure_optimizers(self): # type: ignore
        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val_loss",
                "interval":  "epoch",
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx):
        src, trg = batch
        # trg[:, :-1] fed as decoder input, trg[:, 1:] as target — slicing lives here
        output = self(src, trg[:, :-1])
        output = output.reshape(-1, output.shape[-1])
        target = trg[:, 1:].reshape(-1)

        loss  = self.criterion(output, target)
        mask  = target != self.pad_idx
        acc   = (output.argmax(dim=1)[mask] == target[mask]).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc',  acc,  prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        output = self(src, trg[:, :-1])
        output = output.reshape(-1, output.shape[-1])
        target = trg[:, 1:].reshape(-1)

        loss  = self.criterion(output, target)
        mask  = target != self.pad_idx
        acc   = (output.argmax(dim=1)[mask] == target[mask]).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc',  acc,  prog_bar=True)

    def _make_tgt_mask(self, size, device):
        # FIX 3: float additive mask instead of boolean — works correctly on all PyTorch versions
        return torch.triu(
            torch.full((size, size), float('-inf'), device=device), diagonal=1
        )

    def predict_step(self, batch, batch_idx, max_len=50):
        src, _     = batch
        batch_size = src.shape[0]
        end_idx    = self.dataset.sql_vocab.stoi["<end>"]
        start_idx  = self.dataset.sql_vocab.stoi["<start>"]

        src_pad_mask = (src == self.pad_idx)

        with torch.no_grad():
            src_emb = self.pos_enc(self.src_embedding(src))
            memory  = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        trg      = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            trg_len      = trg.shape[1]
            trg_mask     = self._make_tgt_mask(trg_len, src.device)   # FIX 3
            trg_pad_mask = (trg == self.pad_idx)

            with torch.no_grad():
                trg_emb = self.pos_enc(self.trg_embedding(trg))
                output  = self.transformer.decoder(trg_emb, memory,
                                                   tgt_mask=trg_mask,
                                                   tgt_key_padding_mask=trg_pad_mask,
                                                   memory_key_padding_mask=src_pad_mask)
                output  = self.fc_out(output)

            next_token = output[:, -1, :].argmax(dim=1, keepdim=True)
            trg        = torch.cat([trg, next_token], dim=1)

            finished |= (next_token.squeeze(1) == end_idx)
            if finished.all():
                break

        results = []
        for i in range(batch_size):
            tokens = trg[i, 1:].tolist()
            if end_idx in tokens:
                tokens = tokens[:tokens.index(end_idx)]
            sql = " ".join(
                self.dataset.sql_vocab.itos[t]
                for t in tokens
                if t not in [self.pad_idx,
                              self.dataset.sql_vocab.stoi["<start>"],
                              self.dataset.sql_vocab.stoi["<unk>"]]
            )
            results.append(sql)
        return results

    def translate(self, sentence: str, max_len=50):
        was_training = self.training
        self.eval()


        tokens       = self.dataset.text_vocab.encode(sentence).unsqueeze(0).to(self.device)
        src_pad_mask = (tokens == self.pad_idx)
        end_idx      = self.dataset.sql_vocab.stoi["<end>"]
        start_idx    = self.dataset.sql_vocab.stoi["<start>"]

        with torch.no_grad():
            src_emb = self.pos_enc(self.src_embedding(tokens))
            memory  = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        trg    = torch.tensor([[start_idx]], device=self.device)
        result = []

        for _ in range(max_len):
            trg_len      = trg.shape[1]
            trg_mask     = self._make_tgt_mask(trg_len, self.device)  # FIX 3
            trg_pad_mask = (trg == self.pad_idx)

            with torch.no_grad():
                trg_emb = self.pos_enc(self.trg_embedding(trg))
                output  = self.transformer.decoder(trg_emb, memory,
                                                   tgt_mask=trg_mask,
                                                   tgt_key_padding_mask=trg_pad_mask,
                                                   memory_key_padding_mask=src_pad_mask)
                output  = self.fc_out(output)

            next_token = output[:, -1, :].argmax(dim=1)
            token      = self.dataset.sql_vocab.itos[next_token.item()]

            if token == "<end>":
                break
            result.append(token)

            trg = torch.cat([trg, next_token.unsqueeze(0)], dim=1)

        if was_training:
            self.train()
        return " ".join(result)
    
model = Transformer(
    input_dim=len(text_vocab),
    output_dim=len(sql_vocab),
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    d_ff=d_ff,
    dropout=dropout,
    max_len=max_seq_len,
    dataset=dataset,
    pad_idx=pad_idx
)