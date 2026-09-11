"""
Chạy các baselines cho MIT-BIH ECG (PHIÊN BẢN TỐI ƯU).

Cấu hình:
- LR + DT: chạy nhanh trên CPU (~1-2 phút mỗi cái)
- LSTM/BiLSTM: hidden=96, epochs=45, early stop patience=4
- num_workers=0 + pin_memory=False để tránh treo trên Windows

Cách chạy:
    python run_baselines.py
"""

import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from pathlib import Path


NUM_CLASSES = 5


def load_data():
    train = np.load("data/processed/splits/inter_train.npz")
    val   = np.load("data/processed/splits/inter_val.npz")
    test  = np.load("data/processed/splits/inter_test.npz")
    return (train["X"], train["y"]), (val["X"], val["y"]), (test["X"], test["y"])


from sklearn.metrics import classification_report

def compute_metrics(y_true, y_pred, y_prob):
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision_macro":  precision_score(y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0),
        "recall_macro":     recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0),
        "f1_macro":         f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0),
        "f1_weighted":      f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="weighted", zero_division=0),
    }

    # Per-class metrics for classes N, S, V, F (labels 0, 1, 2, 3)
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    
    metrics["f1_N"] = f1_per_class[0]
    metrics["f1_S"] = f1_per_class[1]
    metrics["f1_V"] = f1_per_class[2]
    metrics["f1_F"] = f1_per_class[3]
    
    metrics["rec_N"] = rec_per_class[0]
    metrics["rec_S"] = rec_per_class[1]
    metrics["rec_V"] = rec_per_class[2]
    metrics["rec_F"] = rec_per_class[3]

    try:
        # AUC: restrict to AAMI 4-class (labels 0-3) and only classes present in y_true.
        valid_idx = np.isin(y_true, [0, 1, 2, 3])
        y_true_4c = y_true[valid_idx]
        y_prob_4c = y_prob[valid_idx]

        present_labels = sorted(set(y_true_4c.tolist()) & {0, 1, 2, 3})
        if len(present_labels) >= 2:
            # Ensure y_prob has at least 4 columns
            if y_prob_4c.shape[1] < 4:
                y_prob_4c_padded = np.zeros((y_prob_4c.shape[0], 4))
                y_prob_4c_padded[:, :y_prob_4c.shape[1]] = y_prob_4c
                y_prob_4c = y_prob_4c_padded
            else:
                y_prob_4c = y_prob_4c[:, :4]
                
            # Re-normalize so rows sum to 1 for OvR
            row_sums = y_prob_4c.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1e-9, row_sums)
            y_prob_4c = y_prob_4c / row_sums
            
            metrics["auc_ovr"] = roc_auc_score(
                y_true_4c, y_prob_4c,
                multi_class="ovr",
                average="macro",
                labels=present_labels,
            )

            # Per-class AUC (binary OvR for each AAMI class).
            # Return float('nan') — NOT 0.0 — when a class has no positive samples,
            # so "not computable" is never confused with "zero discriminability".
            class_names = {0: "N", 1: "S", 2: "V", 3: "F"}
            for cls_id, cls_name in class_names.items():
                key = f"auc_{cls_name}"
                if cls_id not in present_labels:
                    metrics[key] = float("nan")
                    continue
                try:
                    y_bin = (y_true_4c == cls_id).astype(int)
                    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                        metrics[key] = float("nan")
                    else:
                        metrics[key] = roc_auc_score(y_bin, y_prob_4c[:, cls_id])
                except Exception:
                    metrics[key] = float("nan")
        else:
            metrics["auc_ovr"] = 0.0
            for cls_name in ["N", "S", "V", "F"]:
                metrics[f"auc_{cls_name}"] = float("nan")
    except Exception as e:
        print(f"  Warning: AUC calculation failed: {e}")
        metrics["auc_ovr"] = 0.0
        for cls_name in ["N", "S", "V", "F"]:
            metrics[f"auc_{cls_name}"] = float("nan")
    return metrics




def flatten_sequences(X):
    return X.reshape(X.shape[0], -1)


# ─── Sklearn baselines ───
def run_logistic_regression(train, test):
    X_tr, y_tr = train
    X_te, y_te = test
    X_tr_flat = flatten_sequences(X_tr)
    X_te_flat = flatten_sequences(X_te)

    t0 = time.time()
    model = LogisticRegression(max_iter=5000, random_state=42,
                                class_weight="balanced", n_jobs=-1,
                                solver="lbfgs")
    model.fit(X_tr_flat, y_tr)
    train_time = time.time() - t0

    y_pred = model.predict(X_te_flat)
    y_prob = model.predict_proba(X_te_flat)
    
    # Map to standard 5 columns using model.classes_ to prevent column mismatch if classes are missing
    classes = model.classes_
    y_prob_full = np.zeros((len(X_te_flat), 5))
    for idx, cls in enumerate(classes):
        if cls < 5:
            y_prob_full[:, cls] = y_prob[:, idx]
    y_prob = y_prob_full

    metrics = compute_metrics(y_te, y_pred, y_prob)
    print(f"\n[Logistic Regression Per-Class Test Report]")
    print(classification_report(y_te, y_pred, labels=[0, 1, 2, 3], target_names=["N", "S", "V", "F"], zero_division=0, digits=4))
    return {**metrics, "train_time_sec": train_time}


def run_decision_tree(train, test):
    X_tr, y_tr = train
    X_te, y_te = test
    X_tr_flat = flatten_sequences(X_tr)
    X_te_flat = flatten_sequences(X_te)

    t0 = time.time()
    model = DecisionTreeClassifier(max_depth=15, min_samples_leaf=10,
                                    class_weight="balanced", random_state=42)
    model.fit(X_tr_flat, y_tr)
    train_time = time.time() - t0

    y_pred = model.predict(X_te_flat)
    y_prob = model.predict_proba(X_te_flat)
    
    # Map to standard 5 columns using model.classes_ to prevent column mismatch if classes are missing
    classes = model.classes_
    y_prob_full = np.zeros((len(X_te_flat), 5))
    for idx, cls in enumerate(classes):
        if cls < 5:
            y_prob_full[:, cls] = y_prob[:, idx]
    y_prob = y_prob_full

    metrics = compute_metrics(y_te, y_pred, y_prob)
    print(f"\n[Decision Tree Per-Class Test Report]")
    print(classification_report(y_te, y_pred, labels=[0, 1, 2, 3], target_names=["N", "S", "V", "F"], zero_division=0, digits=4))
    return {**metrics, "train_time_sec": train_time}


# ─── LSTM baselines ───
class LSTMBaseline(nn.Module):
    """LSTM/BiLSTM baseline với CNN feature extractor (để fair so với ResMemECG)."""

    def __init__(self, input_size, hidden_size=96, bidirectional=False,
                 dropout=0.25, num_classes=5, cnn_out_channels=64):
        super().__init__()

        # CNN feature extractor (giống ResMemECG để fair)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
        )

        # LSTM hoặc BiLSTM trên features
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        # Attention pooling (giống ResMemECG để fair)
        self.attention = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.Tanh(),
            nn.Linear(out_dim // 2, 1),
        )

        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C) → CNN → (B, T', C')
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        # LSTM
        h_seq, _ = self.lstm(x)  # (B, T', H)

        # Attention pooling
        scores = self.attention(h_seq)              # (B, T', 1)
        weights = torch.softmax(scores, dim=1)
        h_pooled = (h_seq * weights).sum(dim=1)     # (B, H)

        # Classification
        h_pooled = self.layer_norm(h_pooled)
        h_pooled = self.dropout(h_pooled)
        return self.classifier(h_pooled)



# ─── ResNet1D baseline ───
class ResidualBlock1D(nn.Module):
    """Basic residual block cho 1D signal."""
    def __init__(self, channels, kernel_size=7, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class ResNet1D(nn.Module):
    """
    ResNet1D baseline — 1D residual network cho ECG classification.
    Kiến trúc gần với Kiranyaz et al. (2016) và Wang et al. (2017):
    stem Conv → 4 residual blocks (64 ch) → global avg pool → classifier.
    ~220K params, so sánh được với ResMemECG (~505K).
    """
    def __init__(self, input_size=1, num_classes=5, base_ch=64, dropout=0.25):
        super().__init__()
        # Stem: downample T=187 → T=46 (giống CNN của các model khác)
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, base_ch, kernel_size=15, padding=7, stride=2, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # 4 residual blocks cùng channel
        self.blocks = nn.Sequential(
            ResidualBlock1D(base_ch, kernel_size=7, dropout=dropout * 0.4),
            ResidualBlock1D(base_ch, kernel_size=7, dropout=dropout * 0.4),
            ResidualBlock1D(base_ch, kernel_size=7, dropout=dropout * 0.4),
            ResidualBlock1D(base_ch, kernel_size=7, dropout=dropout * 0.4),
        )
        # Global average pooling → flatten → classify
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch, base_ch // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(base_ch // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 1) → transpose → (B, 1, T)
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        return self.classifier(x)


# ─── InceptionTime baseline ───

class InceptionModule(nn.Module):
    """One Inception module: bottleneck + parallel multi-scale convs + pool branch."""
    def __init__(self, in_channels, n_filters=24, kernel_sizes=(9, 19, 39)):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        conv_in = n_filters if self.use_bottleneck else in_channels
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, n_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        self.out_channels = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(self.out_channels)

    def forward(self, x):
        x_in = self.bottleneck(x) if self.use_bottleneck else x
        branches = [conv(x_in) for conv in self.convs]
        branches.append(self.pool_conv(self.pool(x)))
        out = torch.cat(branches, dim=1)
        return torch.relu(self.bn(out))


class InceptionTime(nn.Module):
    """
    InceptionTime baseline (Ismail Fawaz et al., 2020, "InceptionTime: Finding
    AlexNet for Time Series Classification"). Stack of Inception modules with
    a residual shortcut every 3 modules, global average pooling, classifier head.
    """
    def __init__(self, input_size=1, num_classes=5, n_filters=24, depth=6, dropout=0.25):
        super().__init__()
        self.modules_list = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        in_ch = input_size
        shortcut_in_ch = input_size
        for d in range(depth):
            module = InceptionModule(in_ch, n_filters=n_filters)
            self.modules_list.append(module)
            in_ch = module.out_channels
            if (d + 1) % 3 == 0:
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(shortcut_in_ch, in_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(in_ch),
                ))
                shortcut_in_ch = in_ch

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(in_ch // 2, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, 1) → (B, 1, T)
        residual = x
        shortcut_idx = 0
        for d, module in enumerate(self.modules_list):
            x = module(x)
            if (d + 1) % 3 == 0:
                residual = self.shortcuts[shortcut_idx](residual)
                x = torch.relu(x + residual)
                residual = x
                shortcut_idx += 1
        return self.classifier(x)


# ─── TCN baseline ───

class TCNBlock(nn.Module):
    """Dilated causal residual block (Bai, Kolter & Koltun, 2018)."""
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # Causal: chop the right-side padding after each conv so position t
        # only sees inputs at position <= t.
        out = self.bn1(self.conv1(x))[..., :-self.pad]
        out = self.dropout(torch.relu(out))
        out = self.bn2(self.conv2(out))[..., :-self.pad]
        out = self.dropout(torch.relu(out))
        residual = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + residual)


class TCN(nn.Module):
    """Temporal Convolutional Network baseline (Bai, Kolter & Koltun, 2018)."""
    def __init__(self, input_size=1, num_classes=5, channels=(32, 32, 64, 64),
                 kernel_size=7, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size=kernel_size,
                                    dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(in_ch // 2, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, 1) → (B, 1, T)
        x = self.tcn(x)
        return self.classifier(x)


# ─── ECG-Transformer baseline ───

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ECGTransformer(nn.Module):
    """
    CNN + TransformerEncoder baseline — implements the `TransformerECG`
    anticipated by train_modern_baseline()'s docstring below. CNN frontend
    is identical to LSTMBaseline's for a fair architecture-only comparison.
    """
    def __init__(self, input_size=1, d_model=96, nhead=4, num_layers=3,
                 dim_feedforward=192, dropout=0.25, num_classes=5, cnn_out_channels=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
        )
        self.input_proj = nn.Linear(cnn_out_channels, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        h_seq = self.transformer(x)

        scores = self.attention(h_seq)
        weights = torch.softmax(scores, dim=1)
        h_pooled = (h_seq * weights).sum(dim=1)

        h_pooled = self.layer_norm(h_pooled)
        h_pooled = self.dropout(h_pooled)
        return self.classifier(h_pooled)




def train_modern_baseline(name, model, train, val, test, device,
                           class_weights=None, epochs=45):
    """
    Training loop dùng chung cho ResNet1D, InceptionTime, TCN, ECGTransformer.
    Giống với train_lstm_baseline về config: Adam lr=1e-3, patience=4,
    weighted CE, gradient clipping 0.5.
    """
    X_tr, y_tr = train
    X_va, y_va = val
    X_te, y_te = test

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(),
                      torch.from_numpy(y_tr).long()),
        batch_size=128, shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va).float(),
                      torch.from_numpy(y_va).long()),
        batch_size=128, shuffle=False, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te).float(),
                      torch.from_numpy(y_te).long()),
        batch_size=128, shuffle=False, num_workers=0, pin_memory=False,
    )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"best_{name.lower().replace('-', '_')}.pt"

    import sys
    force_retrain = "--force-retrain" in sys.argv

    if ckpt_path.exists() and not force_retrain:
        print(f"  Found pre-trained checkpoint: {ckpt_path}. Loading for evaluation...")
        best_state = torch.load(ckpt_path, map_location=device)
        train_time = 0.0
    else:
        criterion = (nn.CrossEntropyLoss(weight=class_weights.to(device))
                     if class_weights is not None else nn.CrossEntropyLoss())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_f1, best_state, patience_cnt = 0.0, None, 0
        t0 = time.time()
        last_epoch = 0

        for epoch in range(1, epochs + 1):
            last_epoch = epoch
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    all_logits.append(model(X.to(device)).cpu())
                    all_y.append(y)
            preds = torch.cat(all_logits).argmax(-1).numpy()
            y_true_val = torch.cat(all_y).numpy()
            val_f1 = f1_score(y_true_val, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)
            print(f"    epoch {epoch:2d} | val F1_macro = {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 4:
                    break

        train_time = time.time() - t0
        print(f"  [{name}] best val F1_macro = {best_f1:.4f} (stopped at epoch {last_epoch})")
        torch.save(best_state, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")


    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for X, y in test_loader:
            all_logits.append(model(X.to(device)).cpu())
            all_y.append(y)
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs = torch.softmax(logits, -1).numpy()
    preds = logits.argmax(-1).numpy()

    metrics = compute_metrics(y_true, preds, probs)
    print(f"\n[{name} Per-Class Test Report]")
    print(classification_report(y_true, preds, labels=[0, 1, 2, 3], target_names=["N", "S", "V", "F"], zero_division=0, digits=4))
    return {**metrics, "train_time_sec": train_time}


def train_lstm_baseline(name, train, val, test, bidirectional, device,
                        class_weights=None, epochs=45):
    X_tr, y_tr = train
    X_va, y_va = val
    X_te, y_te = test
    input_size = X_tr.shape[-1]

    train_ds = TensorDataset(torch.from_numpy(X_tr).float(),
                              torch.from_numpy(y_tr).long())
    val_ds   = TensorDataset(torch.from_numpy(X_va).float(),
                              torch.from_numpy(y_va).long())
    test_ds  = TensorDataset(torch.from_numpy(X_te).float(),
                              torch.from_numpy(y_te).long())

    # num_workers=0 để tránh treo trên Windows
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False,
                              num_workers=0, pin_memory=False)

    torch.manual_seed(42)
    model = LSTMBaseline(
        input_size=input_size, hidden_size=96,
        bidirectional=bidirectional, dropout=0.25,
        num_classes=NUM_CLASSES,
    ).to(device)

    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_{name.lower()}.pt"

    import sys
    force_retrain = "--force-retrain" in sys.argv

    if checkpoint_path.exists() and not force_retrain:
        print(f"  Found pre-trained checkpoint: {checkpoint_path}. Loading for evaluation...")
        best_state = torch.load(checkpoint_path, map_location=device)
        train_time = 0.0
    else:
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_f1 = 0.0
        best_state = None
        patience = 0
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            model.eval()
            all_logits, all_y = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(device)
                    all_logits.append(model(X).cpu())
                    all_y.append(y)
            logits = torch.cat(all_logits)
            y_true = torch.cat(all_y).numpy()
            preds = logits.argmax(-1).numpy()
            val_f1 = f1_score(y_true, preds, labels=[0, 1, 2, 3], average="macro", zero_division=0)

            print(f"    epoch {epoch:2d} | val F1_macro = {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 4:  # Early stop patience
                    break

        train_time = time.time() - t0
        print(f"  [{name}] best val F1_macro = {best_f1:.4f} (stopped at epoch {epoch})")
        torch.save(best_state, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")


    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            all_logits.append(model(X).cpu())
            all_y.append(y)
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs = torch.softmax(logits, -1).numpy()
    preds = logits.argmax(-1).numpy()

    metrics = compute_metrics(y_true, preds, probs)
    print(f"\n[{name} Per-Class Test Report]")
    print(classification_report(y_true, preds, labels=[0, 1, 2, 3], target_names=["N", "S", "V", "F"], zero_division=0, digits=4))
    return {**metrics, "train_time_sec": train_time}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("\n[Loading data]")
    train_data, val_data, test_data = load_data()
    print(f"  Train: {train_data[0].shape}, Test: {test_data[0].shape}")

    # Compute class weights from inter-patient training distribution (issue #6)
    # Do NOT use data/processed/class_weights.npy which is from intra-patient splits
    y_inter_tr = train_data[1]
    counts = np.bincount(y_inter_tr, minlength=5).astype(np.float64)
    counts = np.where(counts == 0, 1e-9, counts)  # avoid division by zero
    cw_arr = counts.sum() / (5.0 * counts)
    cw_arr = np.clip(cw_arr, 0.5, 50.0).astype(np.float32)
    cw = torch.from_numpy(cw_arr).float()
    print(f"  Class weights (inter-patient): {cw.numpy()}")

    all_results = {}

    print("\n[Logistic Regression]")
    all_results["logistic_regression"] = run_logistic_regression(train_data, test_data)
    print(f"  Test F1_macro: {all_results['logistic_regression']['f1_macro']:.4f}, "
          f"AUC: {all_results['logistic_regression']['auc_ovr']:.4f}")

    print("\n[Decision Tree]")
    all_results["decision_tree"] = run_decision_tree(train_data, test_data)
    print(f"  Test F1_macro: {all_results['decision_tree']['f1_macro']:.4f}, "
          f"AUC: {all_results['decision_tree']['auc_ovr']:.4f}")

    print("\n[LSTM (unidirectional)]")
    all_results["lstm"] = train_lstm_baseline(
        "LSTM", train_data, val_data, test_data,
        bidirectional=False, device=device, class_weights=cw,
        epochs=45,
    )
    print(f"  Test F1_macro: {all_results['lstm']['f1_macro']:.4f}, "
          f"AUC: {all_results['lstm']['auc_ovr']:.4f}")

    print("\n[BiLSTM]")
    all_results["bilstm"] = train_lstm_baseline(
        "BiLSTM", train_data, val_data, test_data,
        bidirectional=True, device=device, class_weights=cw,
        epochs=45,
    )
    print(f"  Test F1_macro: {all_results['bilstm']['f1_macro']:.4f}, "
          f"AUC: {all_results['bilstm']['auc_ovr']:.4f}")

    print("\n[ResNet1D]")
    torch.manual_seed(42)
    resnet_model = ResNet1D(
        input_size=train_data[0].shape[-1],
        num_classes=NUM_CLASSES,
        base_ch=64,
        dropout=0.25,
    )
    all_results["resnet1d"] = train_modern_baseline(
        "ResNet1D", resnet_model, train_data, val_data, test_data,
        device=device, class_weights=cw, epochs=45,
    )
    
    print(f"  Test F1_macro: {all_results['resnet1d']['f1_macro']:.4f}, "
          f"AUC: {all_results['resnet1d']['auc_ovr']:.4f}")

    print("\n[InceptionTime]")
    torch.manual_seed(42)
    inception_model = InceptionTime(
        input_size=train_data[0].shape[-1],
        num_classes=NUM_CLASSES,
        n_filters=24,
        depth=6,
        dropout=0.25,
    )
    all_results["inceptiontime"] = train_modern_baseline(
        "InceptionTime", inception_model, train_data, val_data, test_data,
        device=device, class_weights=cw, epochs=45,
    )
    print(f"  Test F1_macro: {all_results['inceptiontime']['f1_macro']:.4f}, "
          f"AUC: {all_results['inceptiontime']['auc_ovr']:.4f}")

    print("\n[TCN]")
    torch.manual_seed(42)
    tcn_model = TCN(
        input_size=train_data[0].shape[-1],
        num_classes=NUM_CLASSES,
        channels=(32, 32, 64, 64),
        kernel_size=7,
        dropout=0.2,
    )
    all_results["tcn"] = train_modern_baseline(
        "TCN", tcn_model, train_data, val_data, test_data,
        device=device, class_weights=cw, epochs=45,
    )
    print(f"  Test F1_macro: {all_results['tcn']['f1_macro']:.4f}, "
          f"AUC: {all_results['tcn']['auc_ovr']:.4f}")

    print("\n[ECGTransformer]")
    torch.manual_seed(42)
    ecg_transformer_model = ECGTransformer(
        input_size=train_data[0].shape[-1],
        num_classes=NUM_CLASSES,
        d_model=96,
        nhead=4,
        num_layers=3,
        dim_feedforward=192,
        dropout=0.25,
    )
    all_results["ecg_transformer"] = train_modern_baseline(
        "ECGTransformer", ecg_transformer_model, train_data, val_data, test_data,
        device=device, class_weights=cw, epochs=45,
    )
    print(f"  Test F1_macro: {all_results['ecg_transformer']['f1_macro']:.4f}, "
          f"AUC: {all_results['ecg_transformer']['auc_ovr']:.4f}")

    # ── ResMemECG Test Evaluation ──
    print("\n[ResMemECG Evaluation]")
    try:
        from report_results import load_hmr_bilstm
        hmr_ckpt = "results/checkpoints/inter_best_rlstm.pt"
        if Path(hmr_ckpt).exists():
            hmr_model, _ = load_hmr_bilstm(hmr_ckpt, device)
            hmr_model.eval()
            
            # Predict
            all_probs, all_preds = [], []
            X_t = torch.from_numpy(test_data[0]).to(device)
            with torch.no_grad():
                for i in range(0, len(X_t), 256):
                    b = X_t[i:i+256]
                    logits = hmr_model(b)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    all_probs.append(probs)
                    all_preds.append(preds)
            probs = np.concatenate(all_probs)
            preds = np.concatenate(all_preds)
            y_true = test_data[1]
            
            hmr_metrics = compute_metrics(y_true, preds, probs)
            all_results["hmr_bilstm"] = hmr_metrics
            
            print(f"\n[ResMemECG Per-Class Test Report]")
            print(classification_report(y_true, preds, labels=[0, 1, 2, 3], target_names=["N", "S", "V", "F"], zero_division=0, digits=4))
        else:
            print("  Warning: ResMemECG checkpoint not found at results/checkpoints/inter_best_rlstm.pt")
    except Exception as e:
        print(f"  Warning: failed to evaluate ResMemECG: {e}")

    Path("results/logs").mkdir(parents=True, exist_ok=True)

    out_path = Path("results/logs/baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(" BASELINE & ResMemECG MACRO COMPARISON (AAMI 4-Class: N, S, V, F) ")
    print("=" * 80)
    print(f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'F1_w':>8} {'AUC':>8}")
    print("-" * 80)
    order = ["logistic_regression", "decision_tree", "lstm", "bilstm", "resnet1d",
              "inceptiontime", "tcn", "ecg_transformer", "hmr_bilstm"]
    pretty = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "lstm": "LSTM",
        "bilstm": "BiLSTM",
        "resnet1d": "ResNet1D",
        "inceptiontime": "InceptionTime",
        "tcn": "TCN",
        "ecg_transformer": "ECG-Transformer",
        "hmr_bilstm": "ResMemECG",
    }
    for name in order:
        if name not in all_results:
            continue
        m = all_results[name]
        print(f"{pretty[name]:<25} "
              f"{m['accuracy']:>8.4f} "
              f"{m['precision_macro']:>8.4f} "
              f"{m['recall_macro']:>8.4f} "
              f"{m['f1_macro']:>8.4f} "
              f"{m['f1_weighted']:>8.4f} "
              f"{m['auc_ovr']:>8.4f}")
    print("=" * 80)

    print("\n" + "=" * 94)
    print(" BASELINE & ResMemECG PER-CLASS PERFORMANCE (4-Class F1 & Recall) ")
    print("=" * 94)
    print(f"{'Model':<25} {'F1-N':>7} {'F1-S':>7} {'F1-V':>7} {'F1-F':>7} | {'Rec-N':>7} {'Rec-S':>7} {'Rec-V':>7} {'Rec-F':>7}")
    print("-" * 94)
    for name in order:
        if name not in all_results:
            continue
        m = all_results[name]
        # Handle cases where results might have been loaded from old JSONs without these keys
        f1_N = m.get("f1_N", 0.0)
        f1_S = m.get("f1_S", 0.0)
        f1_V = m.get("f1_V", 0.0)
        f1_F = m.get("f1_F", 0.0)
        rec_N = m.get("rec_N", 0.0)
        rec_S = m.get("rec_S", 0.0)
        rec_V = m.get("rec_V", 0.0)
        rec_F = m.get("rec_F", 0.0)
        print(f"{pretty[name]:<25} "
              f"{f1_N:>7.4f} "
              f"{f1_S:>7.4f} "
              f"{f1_V:>7.4f} "
              f"{f1_F:>7.4f} | "
              f"{rec_N:>7.4f} "
              f"{rec_S:>7.4f} "
              f"{rec_V:>7.4f} "
              f"{rec_F:>7.4f}")
    print("=" * 94)

    # Per-class AUC summary table
    print("\n" + "=" * 78)
    print(" PER-CLASS AUC (OvR binary, AAMI 4-Class) ")
    print("=" * 78)
    print(f"{'Model':<25} {'AUC-N':>9} {'AUC-S':>9} {'AUC-V':>9} {'AUC-F':>9} {'AUC-macro':>10}")
    print("-" * 78)
    for name in order:
        if name not in all_results:
            continue
        m = all_results[name]
        def _fmt(v):
            return f"{v:>9.4f}" if (v is not None and not (isinstance(v, float) and np.isnan(v))) else f"{'nan':>9}"
        print(f"{pretty[name]:<25} "
              f"{_fmt(m.get('auc_N', float('nan')))} "
              f"{_fmt(m.get('auc_S', float('nan')))} "
              f"{_fmt(m.get('auc_V', float('nan')))} "
              f"{_fmt(m.get('auc_F', float('nan')))} "
              f"{_fmt(m.get('auc_ovr', float('nan')))}")
    print("=" * 78)
    print(f"\n[OK] Results saved to {out_path}")




if __name__ == "__main__":
    main()