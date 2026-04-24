import torch
import pytorch_lightning as pl
from pathlib import Path
import os
from torchtext.data.metrics import bleu_score
from model import model, device, Transformer
from dataset import test_loader, sql_vocab, text_vocab, dataset, pad_idx
from config import d_model, n_heads, n_layers, d_ff, dropout, max_seq_len
import warnings

if device.type == "cuda":
    torch.set_float32_matmul_precision( 'high')

def load_model(output_file, model=model):
    checkpoint = torch.load(output_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model


def clean(tokens):
    """Remove special tokens from token list."""
    return [
        t for t in tokens
        if t not in ["<pad>", "<start>", "<end>", "<unk>"]
    ]


def main():
    warnings.filterwarnings("ignore")
    output_file = Path("checkpoints/text_to_sql.ckpt")

    if not os.path.exists(output_file):
        print("No trained model found")
        return

    model = Transformer.load_from_checkpoint(output_file,
                                             input_dim  = len(text_vocab),
    output_dim = len(sql_vocab),
    d_model    = d_model,
    n_heads    = n_heads,
    n_layers   = n_layers,
    d_ff       = d_ff,
    dropout    = dropout,
    max_len    = max_seq_len,
    dataset    = dataset,
    pad_idx    = pad_idx).eval()

    # ── bulk predictions ──
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',devices=1)
    preds = trainer.predict(model, test_loader)
    preds = [seq for batch in preds for seq in batch]  # type: ignore
    # preds: list of strings e.g. "select count ( * ) from students"

    # ── tokenize predictions ──
    pred_tokens = [
        clean(sql_vocab.tokenizer(seq))
        for seq in preds
    ]

    # ── tokenize references using itos directly ──
    ref_tokens = [
        clean([
            sql_vocab.itos[idx.item()]
            for idx in trg
        ])
        for _, trg in test_loader.dataset
    ]

    # ── compute corpus BLEU ──
    score = bleu_score(pred_tokens, [[ref] for ref in ref_tokens])
    print(f"BLEU: {score:.4f} ({score*100:.2f}%)")


if __name__ == "__main__":
    main()