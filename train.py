import torch
import os 
import time
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from config import epochs, grad_clip
from dataset import train_loader, val_loader
from model import model
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.set_float32_matmul_precision( 'high')

output_dir = Path("checkpoints")
early_stop = EarlyStopping(
        monitor  = 'val_loss',
        patience = 5,        # stop if val_loss doesn't improve for 5 epochs
        mode     = 'min'
    )

checkpoint = ModelCheckpoint(
    dirpath= output_dir,
    monitor   = 'val_loss',
    save_top_k = 1,          # only keep best model
    mode      = 'min',
    filename  = 'text_to_sql',
    auto_insert_metric_name=False
    )

lr_monitor = LearningRateMonitor(logging_interval='step')
logger = TensorBoardLogger("tb_logs", name="text_to_sql")
trainer = pl.Trainer(max_epochs=epochs, gradient_clip_val=grad_clip,accelerator='gpu' if torch.cuda.is_available() else 'cpu',devices=1, callbacks= [early_stop, checkpoint,lr_monitor], logger=logger)

def main():
    warnings.filterwarnings("ignore")
    if not os.path.exists(output_dir):
        output_dir.mkdir()
    start = time.time() // 60
    trainer.fit(model, train_loader, val_loader)
    end = time.time() // 60
    print(f"Training time: {end - start}")

if __name__ == "__main__":
    main()