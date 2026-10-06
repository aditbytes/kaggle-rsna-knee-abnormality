"""Train one fold of my 2.5D knee reader on report-derived soft labels.

Inputs: the 256 px cache from src/preprocess.py (one .npy per series + *_meta.csv) and a CSV of
soft labels per study. The 58 gold-labelled studies are never trained on; every epoch reports
macro AUC on them (image truth) and on the fold's held-out soft labels (report proxy).

    python src/train.py --fold 0 --folds 4 --backbone tf_efficientnetv2_s.in21k_ft_in1k
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from knee import LABELS, KneeNet, StudyDataset, build_study_table


def find(pattern):
    """Matches of a file pattern up to 3 folders deep under the input roots, shallowest first.
    Roots: $KNEE_INPUT, Kaggle's inputs, the local data/ and outputs/ folders. A fixed depth
    (not **) avoids crawling the ~819k DICOM files under train_series/."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [r for r in [os.environ.get("KNEE_INPUT"), "/kaggle/input", f"{here}/../data", f"{here}/../outputs"] if r]
    for root in roots:
        for depth in range(4):
            hits = sorted(glob.glob(os.path.join(root, *["*"] * depth, pattern)))
            if hits:
                return hits if pattern.startswith("cache") else hits[:1]
    raise FileNotFoundError(pattern)


def load_cache():
    """Returns (series -> npy path, series -> slice count) across all cache shards."""
    paths, n_slices = {}, {}
    for meta in find("cache256_train_meta.csv"):
        d = meta.replace("_meta.csv", "")
        m = pd.read_csv(meta)
        m = m[m["error"].isna() & (m["n"] >= 3)] if "error" in m else m[m["n"] >= 3]
        for sr, n in zip(m.series, m.n):
            paths[sr], n_slices[sr] = f"{d}/{sr}.npy", int(n)
    return paths, n_slices


def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def predict(model, loader, device, res):
    model.eval()
    out = []
    for x, mask, pos, _ in loader:
        x, mask, pos = x.to(device), mask.to(device), pos.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(x, mask, pos, res=res).float().sigmoid().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, len(LABELS)), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--backbone", default="tf_efficientnetv2_s.in21k_ft_in1k")
    ap.add_argument("--labels", default="llm_labels_v4_blend.csv")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--k", type=int, default=8, help="3-slice windows per series in training")
    ap.add_argument("--k_eval", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=2, help="studies per step; 4 runs out of memory on a 16 GB T4")
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (effective batch = bs * accum)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many training studies")
    ap.add_argument("--out", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")
    args = ap.parse_args()
    torch.manual_seed(args.fold)
    np.random.seed(args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train_csv = pd.read_csv(find("train.csv")[0])
    series = pd.read_csv(find("train_series.csv")[0])
    soft = pd.read_csv(find(args.labels)[0]).set_index("StudyInstanceUID")[LABELS]
    paths, n_slices = load_cache()
    table = build_study_table(series, n_slices)

    gold = train_csv.dropna(subset=LABELS).set_index("StudyInstanceUID")[LABELS]
    studies = sorted(s for s in table if s in soft.index and s not in gold.index)
    rng = np.random.RandomState(42)  # same folds for every fold index
    fold_of = dict(zip(studies, rng.permutation(len(studies)) % args.folds))
    tr = [s for s in studies if fold_of[s] != args.fold]
    va = [s for s in studies if fold_of[s] == args.fold]
    gd = [s for s in gold.index if s in table]
    if args.limit:
        tr, va = tr[: args.limit], va[: max(4, args.limit // 4)]
    print(f"device {device} | train {len(tr)} | val {len(va)} | gold {len(gd)} | series cached {len(paths)}", flush=True)

    loader = lambda s: np.load(paths[s], mmap_mode="r")
    ds = lambda st, y, k, train: StudyDataset(st, table, None, y, k=k, train=train, loader=loader)
    dl = lambda d, shuffle: torch.utils.data.DataLoader(
        d, batch_size=args.bs, shuffle=shuffle, num_workers=args.workers, drop_last=shuffle,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_dl = dl(ds(tr, soft.loc[tr].values.astype(np.float32), args.k, True), True)
    val_dl = dl(ds(va, None, args.k_eval, False), False)
    gold_dl = dl(ds(gd, None, args.k_eval, False), False)

    model = KneeNet(args.backbone, pretrained=True, img_size=args.res).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    enc = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr}, {"params": rest, "lr": args.lr * 3}], weight_decay=args.wd)
    steps = args.epochs * len(train_dl) // args.accum
    warm = max(1, len(train_dl) // (2 * args.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / steps))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.backbone.split('.')[0]}_r{args.res}_f{args.fold}"
    log = []
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, mask, pos, y) in enumerate(train_dl, 1):
            x, mask, pos, y = x.to(device), mask.to(device), pos.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x, mask, pos, res=args.res, train=True, flip=True)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss / args.accum).backward()
            if i % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            tot, n = tot + loss.item() * len(y), n + len(y)
        pv, pg = predict(model, val_dl, device, args.res), predict(model, gold_dl, device, args.res)
        row = {"epoch": ep, "loss": tot / max(n, 1), "minutes": (time.time() - t0) / 60,
               "val_auc_vs_reports": macro_auc((soft.loc[va].values > 0.5).astype(int), pv),
               "gold_auc": macro_auc(gold.loc[gd].values.astype(int), pg)}
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)

    torch.save({"model": model.state_dict(), "args": {"backbone": args.backbone, "res": args.res}, "log": log},
               f"{args.out}/{tag}.pt")
    pd.DataFrame(pv, index=pd.Index(va, name="StudyInstanceUID"), columns=LABELS).to_csv(f"{args.out}/oof_{tag}.csv")
    pd.DataFrame(pg, index=pd.Index(gd, name="StudyInstanceUID"), columns=LABELS).to_csv(f"{args.out}/gold_{tag}.csv")
    json.dump({"args": vars(args), "log": log}, open(f"{args.out}/log_{tag}.json", "w"), indent=1)
    print("saved", tag)


if __name__ == "__main__":
    main()
