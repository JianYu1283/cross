"""Comment 3: GraphCodeBERT full-469 with softmax probs, for PR-AUC / ROC-AUC.

Loads from a local safetensors snapshot (data/models/graphcodebert) so the
torch<2.6 `torch.load` gate is never hit and no network access is required.
Re-runs the 5x5 CV on the full 469-negative set (69 positives), saving
per-instance class-1 softmax probabilities, and reports pooled F1, precision,
recall, MCC, balanced accuracy, PR-AUC, ROC-AUC.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score,
                             matthews_corrcoef, average_precision_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(__file__))
from codebert_baseline import SolDataset
from transformer_common import common_items

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "data", "models", "graphcodebert")


class TransformerClassifier(nn.Module):
    def __init__(self, model_dir, n_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_classes)

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state  # (B, L, H)
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = common_items(subsample_neg=0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    ds = SolDataset(items, tokenizer, 512)
    labels = ds.labels.numpy()
    n_pos = int((labels == 1).sum()); n_neg = int((labels == 0).sum())
    print(f"dataset: {len(ds)} (pos={n_pos}, neg={n_neg}) on {device}", flush=True)

    per_fold_f1 = []
    probs_all, preds_all, truths_all = [], [], []
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        idx = np.arange(len(ds))
        for fold, (tr, te) in enumerate(skf.split(idx, labels)):
            model = TransformerClassifier(MODEL_DIR).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
            tr_loader = DataLoader(Subset(ds, tr), batch_size=8, shuffle=True)
            model.train()
            for _ in range(3):
                for ids, mask, lab in tr_loader:
                    ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
                    opt.zero_grad()
                    loss = nn.functional.cross_entropy(model(ids, mask), lab)
                    loss.backward(); opt.step()
            model.eval()
            fold_probs, fold_preds, fold_truths = [], [], []
            with torch.no_grad():
                for ids, mask, lab in DataLoader(Subset(ds, te), batch_size=8):
                    ids, mask = ids.to(device), mask.to(device)
                    logits = model(ids, mask)
                    probs = torch.softmax(logits, 1)[:, 1]
                    fold_probs.extend(probs.cpu().tolist())
                    fold_preds.extend(logits.argmax(1).cpu().tolist())
                    fold_truths.extend(lab.tolist())
            per_fold_f1.append(float(f1_score(fold_truths, fold_preds, zero_division=0)))
            probs_all.extend(fold_probs); preds_all.extend(fold_preds); truths_all.extend(fold_truths)
            print(f"  seed={seed} fold={fold} f1={per_fold_f1[-1]:.4f}", flush=True)

    P = np.array(probs_all); Y = np.array(preds_all); T = np.array(truths_all)
    rec_pos = recall_score(T, Y, zero_division=0)
    rec_neg = recall_score(1 - T, 1 - Y, zero_division=0)
    out = {
        "model": "microsoft/graphcodebert-base (local safetensors)",
        "n": len(ds), "n_pos": n_pos, "n_neg": n_neg,
        "per_fold_f1": per_fold_f1,
        "f1_mean": float(np.mean(per_fold_f1)),
        "f1_std": float(np.std(per_fold_f1)),
        "f1_pooled": float(f1_score(T, Y, zero_division=0)),
        "precision": float(precision_score(T, Y, zero_division=0)),
        "recall": float(rec_pos),
        "acc": float(accuracy_score(T, Y)),
        "mcc": float(matthews_corrcoef(T, Y)),
        "balanced_acc": float((rec_pos + rec_neg) / 2),
        "pr_auc": float(average_precision_score(T, P)),
        "roc_auc": float(roc_auc_score(T, P)),
    }
    outp = os.path.join(ROOT, "results", "full469_graphcodebert_metrics.json")
    json.dump(out, open(outp, "w"))
    print(f"RESULT graphcodebert full469: pooled F1={out['f1_pooled']:.3f} "
          f"MCC={out['mcc']:.3f} BA={out['balanced_acc']:.3f} "
          f"PR-AUC={out['pr_auc']:.3f} ROC-AUC={out['roc_auc']:.3f}", flush=True)
    print("saved", outp, flush=True)


if __name__ == "__main__":
    main()
