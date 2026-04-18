# TEXT-to-SQL Database Query System

This project implements **Sequence to Sequence(Seq2Seq)** using **PyTorch Lightning** to generate **SQL QUERY**.

##  Project Structure

```
TEXT-to-SQL Database Query/
├── data                   # Data
├── README.md              # Documentation
├── requirements.txt       # Dependencies
├── .gitignore             # Ignore cache/checkpoint/log files
├── checkpoints/           # Model checkpoints
├── config.py              # Model Hyperparameters
├── dataset.py             # Dataset loading
├── model.py               # seq2seq model 
├── train.py               # Main training script
├── utils.py               # Model loading and evaluation
```

---

##  Setup

### Installation :

```bash
git clone https://github.com/Harish19102003/TEXT-to-SQL-Database-Query-System.git
cd TEXT-to-SQL-Database-Query-System
pip install -r requirements.txt
```

### Train :
```bash
python -m train.py
```

## Evaluation :
Using BLEU Score
```bash
python -m utils.py
```

### TensorBoard :
```bash
tensorboard --logdir tb_logs
```
## Results :

| Metric | Score |
|-------|------|
| BLEU Score | 63.6% |
| Train Accuracy | 89.7% |
| Validation Accuracy | 89.0% |