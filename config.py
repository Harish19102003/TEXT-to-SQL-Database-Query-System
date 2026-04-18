# Transformer hyperparameters
d_model     = 256           # embedding size — must be divisible by n_heads
n_heads     = 8             # attention heads
n_layers    = 3             # encoder and decoder layers each
d_ff        = 2 * d_model   # feedforward size (2x d_model)
dropout     = 0.1           # transformers use lower dropout
max_seq_len = 500           # max length for positional encoding

# kept for compatibility — not used in transformer
enc_emb_dim = d_model
dec_emb_dim = d_model

epochs     = 50
grad_clip  = 1.0
batch_size = 64