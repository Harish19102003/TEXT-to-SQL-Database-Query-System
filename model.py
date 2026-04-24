import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import math
from dataset import dataset, pad_idx, parse_schema, text_vocab, sql_vocab
from config import d_model, n_heads, n_layers, d_ff, dropout, max_seq_len

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.set_float32_matmul_precision( 'high')

class PositionalEncoding(nn.Module):
    """Adds sinusoidal position information to embeddings."""
    def __init__(self, d_model, dropout, max_len=200):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]  # type: ignore
        return self.dropout(x)


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
        self.dataset   = dataset
        self.pad_idx   = pad_idx
        self.d_model   = d_model
        
        self.src_embedding = nn.Embedding(input_dim, d_model)
        self.trg_embedding = nn.Embedding(output_dim, d_model)
        self.pos_enc       = PositionalEncoding(d_model, dropout, max_len)
        self.scale         = math.sqrt(d_model)

        self.transformer = nn.Transformer(
            d_model, n_heads, n_layers, n_layers,
            d_ff, dropout, batch_first=True
        )
        self.fc_out    = nn.Linear(d_model, output_dim)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.save_hyperparameters()

    def make_masks(self, src, trg):
        """Create padding masks and causal mask."""
        src_pad = (src == self.pad_idx)
        trg_pad = (trg == self.pad_idx)
        trg_mask = torch.triu(
            torch.ones(trg.shape[1], trg.shape[1], device=trg.device), diagonal=1
        ).bool()
        return src_pad, trg_mask, trg_pad

    def forward(self, src, trg):
        src_pad, trg_mask, trg_pad = self.make_masks(src, trg)
        
        # embed and scale
        src = self.pos_enc(self.src_embedding(src) * self.scale)
        trg = self.pos_enc(self.trg_embedding(trg) * self.scale)
        
        output = self.transformer(
            src, trg,
            tgt_mask                = trg_mask,
            src_key_padding_mask    = src_pad,
            tgt_key_padding_mask    = trg_pad,
            memory_key_padding_mask = src_pad
        )
        return self.fc_out(output)

    def configure_optimizers(self):  # type: ignore
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
        output   = self(src, trg[:, :-1])
        output   = output.reshape(-1, output.shape[-1])
        target   = trg[:, 1:].reshape(-1)

        loss = self.criterion(output, target)
        mask = target != self.pad_idx
        acc  = (output.argmax(dim=1)[mask] == target[mask]).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc',  acc,  prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        output   = self(src, trg[:, :-1])
        output   = output.reshape(-1, output.shape[-1])
        target   = trg[:, 1:].reshape(-1)

        loss = self.criterion(output, target)
        mask = target != self.pad_idx
        acc  = (output.argmax(dim=1)[mask] == target[mask]).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc',  acc,  prog_bar=True)

    def _make_tgt_mask(self, size, device):
        """Float additive mask for causal attention."""
        return torch.triu(
            torch.full((size, size), float('-inf'), device=device), diagonal=1
        )

    def predict_step(self, batch, batch_idx, max_len=100):
        src, _     = batch
        batch_size = src.shape[0]
        end_idx    = self.dataset.sql_vocab.stoi["<end>"]
        start_idx  = self.dataset.sql_vocab.stoi["<start>"]
        src_pad    = (src == self.pad_idx)

        with torch.no_grad():
            src_emb = self.pos_enc(self.src_embedding(src) * self.scale)
            memory  = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad)

        trg      = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            trg_mask = self._make_tgt_mask(trg.shape[1], src.device)
            trg_pad  = (trg == self.pad_idx)

            with torch.no_grad():
                trg_emb = self.pos_enc(self.trg_embedding(trg) * self.scale)
                output  = self.transformer.decoder(
                    trg_emb, memory,
                    tgt_mask                = trg_mask,
                    tgt_key_padding_mask    = trg_pad,
                    memory_key_padding_mask = src_pad
                )
                output = self.fc_out(output)

            next_tok = output[:, -1, :].argmax(dim=1, keepdim=True)
            trg      = torch.cat([trg, next_tok], dim=1)
            finished |= (next_tok.squeeze(1) == end_idx)
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

    def translate(self, sentence: str, schema: str = None, max_len=50): # type: ignore
        was_training = self.training
        self.eval()

        if schema:
            sentence = f"{sentence} | {parse_schema(schema)}"

        tokens  = self.dataset.text_vocab.encode(sentence).unsqueeze(0).to(self.device)
        src_pad = (tokens == self.pad_idx)
        end_idx = self.dataset.sql_vocab.stoi["<end>"]
        start_idx = self.dataset.sql_vocab.stoi["<start>"]

        with torch.no_grad():
            src_emb = self.pos_enc(self.src_embedding(tokens) * self.scale)
            memory  = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad)

        trg    = torch.tensor([[start_idx]], device=self.device)
        result = []

        for _ in range(max_len):
            trg_mask = self._make_tgt_mask(trg.shape[1], self.device)
            trg_pad  = (trg == self.pad_idx)

            with torch.no_grad():
                trg_emb = self.pos_enc(self.trg_embedding(trg) * self.scale)
                output  = self.transformer.decoder(
                    trg_emb, memory,
                    tgt_mask                = trg_mask,
                    tgt_key_padding_mask    = trg_pad,
                    memory_key_padding_mask = src_pad
                )
                output = self.fc_out(output)

            next_tok = output[:, -1, :].argmax(dim=1)
            token    = self.dataset.sql_vocab.itos[next_tok.item()]

            if token == "<end>":
                break
            result.append(token)
            trg = torch.cat([trg, next_tok.unsqueeze(0)], dim=1)

        if was_training:
            self.train()
        return " ".join(result)


model = Transformer(
    input_dim  = len(text_vocab),
    output_dim = len(sql_vocab),
    d_model    = d_model,
    n_heads    = n_heads,
    n_layers   = n_layers,
    d_ff       = d_ff,
    dropout    = dropout,
    max_len    = max_seq_len,
    dataset    = dataset,
    pad_idx    = pad_idx
).to(device)