from __future__ import annotations

import argparse
import random
from pathlib import Path


AMD_MAP = {
    "normal": 0,
    "no amd": 0,
    "no.amd.diagnosis": 0,
    "not.in.icd.table": 0,
    "early amd": 1,
    "early.dry": 1,
    "intermediate amd": 1,
    "intermediate.dry": 1,
    "late amd": 1,
    "advanced.atrophic.dry.with.subfoveal.involvement": 1,
    "advanced.atrophic.dry.without.subfoveal.involvement": 1,
    "wet.amd.active.choroidal.neovascularization": 1,
    "wet.amd.inactive.choroidal.neovascularization": 1,
    "wet.amd.inactive.scar": 1,
}


DR_MAP = {
    "normal": 0,
    "no dr": 0,
    "no.dr.diagnosis": 0,
    "not.in.icd.table": 0,
    "non-vision threatening dr": 0,
    "non vision threatening dr": 0,
    "mild.npdr": 0,
    "moderate.npdr": 0,
    "vision threatening dr": 1,
    "vision-threatening dr": 1,
    "severe.npdr": 1,
    "pdr": 1,
}


def metadata_path(root: Path, task: str) -> Path:
    source = {"amd": "AMD", "dr": "DR"}[task]
    candidates = [
        root / "HarvardFairVision30k" / source / "ReadMe" / f"data_summary_{task}.csv",
        root / f"data_summary_{task}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing metadata CSV for {task}: {candidates}")


def image_path(root: Path, filename: str, split: str, source: str) -> Path:
    split_folder = {"train": "Training", "val": "Validation", "test": "Test"}[split]
    candidates = [
        root / split_folder / filename,
        root / split_folder / source / filename,
        root / source / split_folder / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing image for {filename}: {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny overfit test for FairVision labels/images. If this cannot overfit, the pipeline is broken."
    )
    parser.add_argument("--root", type=Path, default=Path("Datasets/FairVision"))
    parser.add_argument("--task", choices=("amd", "dr"), required=True)
    parser.add_argument("--modality", choices=("slo", "oct_center"), default="oct_center")
    parser.add_argument("--n-pos", type=int, default=32)
    parser.add_argument("--n-neg", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    label_map = AMD_MAP if args.task == "amd" else DR_MAP
    source = "AMD" if args.task == "amd" else "DR"
    label_col = args.task
    frame = pd.read_csv(metadata_path(args.root, args.task))
    frame = frame[frame["use"].astype(str).str.lower().isin({"training", "train"})].copy()
    frame["binary_label"] = frame[label_col].astype(str).str.strip().str.lower().map(label_map)
    frame = frame[frame["binary_label"].isin([0, 1])].copy()

    pos = frame[frame["binary_label"] == 1].sample(n=args.n_pos, random_state=args.seed, replace=len(frame[frame["binary_label"] == 1]) < args.n_pos)
    neg = frame[frame["binary_label"] == 0].sample(n=args.n_neg, random_state=args.seed, replace=len(frame[frame["binary_label"] == 0]) < args.n_neg)
    sample = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    class SmallSet(Dataset):
        def __len__(self):
            return len(sample)

        def __getitem__(self, index):
            row = sample.iloc[index]
            path = image_path(args.root, str(row["filename"]), "train", source)
            with np.load(path, allow_pickle=True) as data:
                if args.modality == "slo":
                    arr = data["slo_fundus"]
                else:
                    volume = data["oct_bscans"]
                    arr = volume[volume.shape[0] // 2]
            arr = np.asarray(arr)
            if arr.max() <= 1.0:
                arr = (arr * 255).astype("uint8")
            else:
                arr = np.clip(arr, 0, 255).astype("uint8")
            image = Image.fromarray(arr).convert("RGB")
            return transform(image), torch.tensor(float(row["binary_label"]), dtype=torch.float32)

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 1),
    ).to(device)

    loader = DataLoader(SmallSet(), batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"task={args.task} modality={args.modality} n={len(sample)} pos={int(sample['binary_label'].sum())} neg={int((sample['binary_label'] == 0).sum())}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            ys = []
            ps = []
            with torch.no_grad():
                for images, labels in DataLoader(SmallSet(), batch_size=args.batch_size, shuffle=False, num_workers=0):
                    probs = torch.sigmoid(model(images.to(device))).view(-1).cpu().numpy()
                    ps.extend(probs.tolist())
                    ys.extend(labels.numpy().astype(int).tolist())
            preds = [int(p >= 0.5) for p in ps]
            acc = sum(int(a == b) for a, b in zip(preds, ys)) / len(ys)
            try:
                auc = roc_auc_score(ys, ps)
            except Exception:
                auc = float("nan")
            print(f"epoch={epoch} loss={sum(losses)/len(losses):.4f} train_acc={acc:.3f} train_auc={auc:.3f}")


if __name__ == "__main__":
    main()
