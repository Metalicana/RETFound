from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TorchFeatureProbe:
    model: object
    mean: object
    std: object
    input_dim: int
    hidden_dim: int
    dropout: float
    num_classes: int
    best_epoch: int
    best_val_loss: float


def build_mlp_probe(torch, input_dim: int, hidden_dim: int, dropout: float, num_classes: int):
    if hidden_dim <= 0:
        return torch.nn.Linear(input_dim, num_classes)
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.LayerNorm(hidden_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, num_classes),
    )


def class_weights(np, torch, y_train, num_classes: int, device):
    counts = np.bincount(y_train.astype("int64"), minlength=num_classes).astype("float32")
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def standardize_features(np, x_train, *others):
    x_train = np.nan_to_num(x_train.astype("float32"), copy=False)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std).astype("float32")
    mean = mean.astype("float32")
    standardized = [(x_train - mean) / std]
    for item in others:
        item = np.nan_to_num(item.astype("float32"), copy=False)
        standardized.append((item - mean) / std)
    return mean, std, standardized


def train_torch_probe(
    np,
    torch,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    device,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    y_train = y_train.astype("int64")
    y_val = y_val.astype("int64")
    num_classes = int(max(y_train.max(), y_val.max()) + 1)
    mean, std, (x_train_std, x_val_std) = standardize_features(np, x_train, x_val)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(x_train_std, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    x_val_t = torch.tensor(x_val_std, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    model = build_mlp_probe(torch, x_train.shape[1], hidden_dim, dropout, num_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(np, torch, y_train, num_classes, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(labels.numel())
            seen += int(labels.numel())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            train_loss = running_loss / max(seen, 1)
            print(
                f"torch_probe_epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} best_epoch={best_epoch}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return TorchFeatureProbe(
        model=model,
        mean=mean,
        std=std,
        input_dim=int(x_train.shape[1]),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        num_classes=int(num_classes),
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val_loss),
    )


def torch_probe_checkpoint(probe: TorchFeatureProbe, extra: dict):
    checkpoint = {
        "probe_kind": "torch_mlp",
        "state_dict": {key: value.detach().cpu() for key, value in probe.model.state_dict().items()},
        "mean": probe.mean,
        "std": probe.std,
        "input_dim": probe.input_dim,
        "hidden_dim": probe.hidden_dim,
        "dropout": probe.dropout,
        "num_classes": probe.num_classes,
        "best_epoch": probe.best_epoch,
        "best_val_loss": probe.best_val_loss,
    }
    checkpoint.update(extra)
    return checkpoint


def torch_probs_for_features(np, torch, probe: TorchFeatureProbe, features_by_index: dict[int, object], device):
    model = probe.model.to(device)
    model.eval()
    out = {}
    with torch.no_grad():
        for index, feature in features_by_index.items():
            feature = np.nan_to_num(feature.astype("float32"), copy=False)
            feature = (feature.reshape(1, -1) - probe.mean) / probe.std
            tensor = torch.tensor(feature, dtype=torch.float32, device=device)
            probs = torch.softmax(model(tensor), dim=1).detach().cpu().numpy()[0]
            out[index] = float(1.0 - probs[0]) if len(probs) > 2 else float(probs[1])
    return out
