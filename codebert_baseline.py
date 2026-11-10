"""Transformer baseline: fine-tune a code transformer on raw Solidity source.

Default model is microsoft/graphcodebert-base (override with --model). This is
a transformer baseline to test whether the low-level-call signal is captured
by a source-level transformer (which sees `msg.sender.call.value()` directly)
without any explicit graph construction. Reports mean F1 over 5-fold
stratified CV.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import collect_dataset  # noqa: E402


class SolDataset(Dataset):
    def __init__(self, items, tokenizer, max_len):
        self.ids, self.masks, self.labels = [], [], []
        for fpath, label in items:
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
            except OSError:
                continue
            tok = tokenizer(code, max_length=max_len, truncation=True,
                            padding="max_length", return_tensors="pt")
            self.ids.append(tok["input_ids"][0])
            self.masks.append(tok["attention_mask"][0])
            self.labels.append(label)
        self.ids = torch.stack(self.ids)
        self.masks = torch.stack(self.masks)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.masks[i], self.labels[i]


class TransformerClassifier(nn.Module):
    def __init__(self, model_name="microsoft/graphcodebert-base", n_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_classes)

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state  # (B, L, H)
        # mean pool over non-padding tokens
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)


def train_eval(items, tokenizer, model_name, epochs, lr, batch, max_len, device, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    ds = SolDataset(items, tokenizer, max_len)
    idx = np.arange(len(ds))
    labels_np = ds.labels.numpy()
    cnt = {int(c): (labels_np == c).sum() for c in set(labels_np.tolist())}
    n_folds = max(2, min(5, min(cnt.values())))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in skf.split(idx, labels_np):
        model = TransformerClassifier(model_name).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        tr_loader = DataLoader(torch.utils.data.Subset(ds, tr), batch_size=batch, shuffle=True)
        model.train()
        for _ in range(epochs):
            for ids, mask, lab in tr_loader:
                ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
                opt.zero_grad()
                logits = model(ids, mask)
                loss = nn.functional.cross_entropy(logits, lab)
                loss.backward()
                opt.step()
        model.eval()
        preds, truths = [], []
        te_loader = DataLoader(torch.utils.data.Subset(ds, te), batch_size=batch)
        with torch.no_grad():
            for ids, mask, lab in te_loader:
                ids, mask = ids.to(device), mask.to(device)
                logits = model(ids, mask)
                preds.extend(logits.argmax(1).cpu().tolist())
                truths.extend(lab.tolist())
        accs.append(accuracy_score(truths, preds))
        f1s.append(f1_score(truths, preds, zero_division=0))
    return float(np.mean(accs)), float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-dirs", nargs="+", required=True)
    ap.add_argument("--neg-dirs", nargs="+", required=True)
    ap.add_argument("--model", default="microsoft/graphcodebert-base")
    ap.add_argument("--limit", type=int, default=130)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = collect_dataset(args.pos_dirs, args.neg_dirs, args.limit, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    acc, f1 = train_eval(items, tokenizer, args.model, args.epochs, 2e-5, args.batch, args.max_len, device, seed=args.seed)
    print(f"RESULT model={args.model} acc={acc:.4f} f1={f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
