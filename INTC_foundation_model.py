# ============================================================
# INTC Time-Series Foundation Model v2 — Margin-Aware
# 3-head transformer: classification + regression + sizing
# Designed for 150% margin (2.5x max leverage) trading
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# Shared constants (safe to import)
# ============================================================
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "ma_fast", "ma_slow", "macd", "signal_line",
    "rsi", "vwap", "bb_upper", "bb_lower",
    "recent_high", "recent_low",
]

INTERVAL_IDS = {"1m": 0, "5m": 1, "15m": 2, "30m": 3, "60m": 4, "1d": 5}

SEQ_LEN = 64
HORIZON = 5

# ============================================================
# Margin config — 150% margin = 2.5x max buying power
# ============================================================
MARGIN_CONFIG = {
    "margin_ratio": 1.5,           # Can borrow 150% of equity
    "max_leverage": 2.5,           # 100% equity + 150% margin = 250%
    "maintenance_margin": 0.25,    # 25% maintenance (Reg T)
    "margin_interest_annual": 0.08,  # 8% annual margin interest
    "stop_loss_base": 0.05,        # 5% stop-loss at 1x leverage
    "stop_loss_leveraged": 0.05,   # 5% stop-loss when leveraged (match base)
}


def compute_indicators(df):
    """Compute the 15 technical indicators used by the foundation model."""
    d = df.copy()

    # Moving averages
    d["ma_fast"] = d["close"].rolling(10).mean()
    d["ma_slow"] = d["close"].rolling(30).mean()

    # MACD
    d["ema12"] = d["close"].ewm(span=12).mean()
    d["ema26"] = d["close"].ewm(span=26).mean()
    d["macd"] = d["ema12"] - d["ema26"]
    d["signal_line"] = d["macd"].ewm(span=9).mean()

    # RSI
    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    d["rsi"] = 100 - (100 / (1 + rs))

    # VWAP
    d["cum_vol"] = d["volume"].cumsum()
    d["cum_pv"] = (d["close"] * d["volume"]).cumsum()
    d["vwap"] = d["cum_pv"] / d["cum_vol"]

    # Bollinger Bands
    d["bb_ma"] = d["close"].rolling(20).mean()
    d["bb_std"] = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_ma"] + 2 * d["bb_std"]
    d["bb_lower"] = d["bb_ma"] - 2 * d["bb_std"]

    # Breakout levels
    d["recent_high"] = d["close"].rolling(20).max()
    d["recent_low"] = d["close"].rolling(20).min()

    d.dropna(inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d


def compute_leverage(cls_probs, avg_ret, forecast_std, max_leverage=2.5):
    """
    Compute recommended leverage from model outputs.

    Uses classification confidence, return magnitude, and forecast
    uncertainty to determine optimal position sizing.

    Returns leverage in [0, max_leverage].
    """
    max_prob = max(cls_probs)

    # Base leverage from confidence
    if max_prob >= 0.70 and abs(avg_ret) > 0.0008:
        base = 2.5   # Very high confidence + strong signal
    elif max_prob >= 0.55 and abs(avg_ret) > 0.0004:
        base = 2.0   # High confidence
    elif max_prob >= 0.45 and abs(avg_ret) > 0.0001:
        base = 1.5   # Medium confidence
    else:
        base = 1.0   # Low confidence, no margin

    # Reduce leverage if forecast is uncertain (high std)
    if forecast_std > 0.005:
        base *= 0.80
    elif forecast_std > 0.003:
        base *= 0.90

    return min(base, max_leverage)


# ============================================================
# Model: INTCFoundationModel v2 — 3-head margin-aware
# ============================================================
class INTCFoundationModel(nn.Module):
    def __init__(self, input_dim=15, dim=128, heads=8, layers=6,
                 ff_dim=512, dropout=0.2, num_intervals=6,
                 max_seq=64, num_classes=3, horizon=5):
        super().__init__()

        # Feature embedding
        self.feat_embed = nn.Linear(input_dim, dim)

        # Interval embedding
        self.interval_embed = nn.Embedding(num_intervals, dim)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq, dim) * 0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

        # Head 1: Classification (BUY / HOLD / SELL)
        self.cls_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Head 2: Regression (predict % change of next `horizon` closes)
        self.reg_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
        )

        # Head 3: Sizing (optimal leverage 0..1, scaled to 0..max_leverage at inference)
        # Learns from risk-adjusted forward returns
        self.sizing_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x, interval_id=None):
        """
        x: (B, T, input_dim) features
        interval_id: (B,) int tensor
        Returns: cls_logits (B, 3), reg_pred (B, horizon), sizing (B, 1)
        """
        B, T, _ = x.shape

        h = self.feat_embed(x)  # (B, T, dim)

        # Add interval embedding (broadcast across time)
        if interval_id is not None:
            iv = self.interval_embed(interval_id)  # (B, dim)
            h = h + iv.unsqueeze(1)

        # Add positional encoding
        h = h + self.pos_embed[:, :T, :]

        h = self.encoder(h)
        h = self.norm(h)

        # Pool: take last token
        last = h[:, -1, :]  # (B, dim)

        cls_logits = self.cls_head(last)
        reg_pred = self.reg_head(last)
        sizing = self.sizing_head(last)   # (B, 1) in [0, 1]
        return cls_logits, reg_pred, sizing


# ============================================================
# Training code — only runs when executed directly
# ============================================================
if __name__ == "__main__":
    import yfinance as yf
    from sklearn.preprocessing import StandardScaler
    import joblib
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Margin config: {MARGIN_CONFIG}")

    # ========================================================
    # 1. Download multi-interval data
    # ========================================================
    INTERVALS = {
        "1m":  {"period": "7d"},
        "5m":  {"period": "60d"},
        "15m": {"period": "60d"},
        "30m": {"period": "60d"},
        "60m": {"period": "730d"},
        "1d":  {"period": "10y"},
    }

    all_dfs = {}
    for iv, params in INTERVALS.items():
        print(f"Downloading INTC {iv} ({params['period']})...")
        raw = yf.download("INTC", period=params["period"], interval=iv, progress=False)
        if raw is None or len(raw) == 0:
            print(f"  [WARN] No data for {iv}, skipping")
            continue

        # Normalize columns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = raw.columns.str.lower()

        raw.dropna(inplace=True)
        raw.reset_index(drop=True, inplace=True)

        if len(raw) < SEQ_LEN + HORIZON + 50:
            print(f"  [WARN] Only {len(raw)} bars for {iv}, skipping")
            continue

        # Compute indicators
        d = compute_indicators(raw)
        print(f"  Got {len(d)} bars after indicators")
        all_dfs[iv] = d

    if not all_dfs:
        print("ERROR: No data collected for any interval.")
        exit(1)

    # ========================================================
    # 2. Labeling: classification + regression + sizing target
    # ========================================================
    MAX_LEVERAGE = MARGIN_CONFIG["max_leverage"]

    def label_df(d):
        """Add classification labels, regression targets, and sizing targets."""
        d = d.copy()

        # --- Classification: multi-indicator vote ---
        d["buy_votes"] = 0
        d["sell_votes"] = 0

        d.loc[d["ma_fast"] > d["ma_slow"], "buy_votes"] += 1
        d.loc[d["ma_fast"] < d["ma_slow"], "sell_votes"] += 1

        d.loc[d["macd"] > d["signal_line"], "buy_votes"] += 1
        d.loc[d["macd"] < d["signal_line"], "sell_votes"] += 1

        d.loc[d["rsi"] < 30, "buy_votes"] += 1
        d.loc[d["rsi"] > 70, "sell_votes"] += 1

        d.loc[d["close"] < d["vwap"], "buy_votes"] += 1
        d.loc[d["close"] > d["vwap"], "sell_votes"] += 1

        d.loc[d["close"] < d["bb_lower"], "buy_votes"] += 1
        d.loc[d["close"] > d["bb_upper"], "sell_votes"] += 1

        d.loc[d["close"] > d["recent_high"], "buy_votes"] += 1
        d.loc[d["close"] < d["recent_low"], "sell_votes"] += 1

        score = d["buy_votes"] - d["sell_votes"]

        # 3-class: BUY=2, HOLD=1, SELL=0
        d["cls_label"] = 1  # default HOLD
        d.loc[score >= 1, "cls_label"] = 2   # BUY
        d.loc[score <= -1, "cls_label"] = 0  # SELL

        # --- Regression: % change of next 5 closes ---
        for h in range(1, HORIZON + 1):
            d[f"ret_{h}"] = d["close"].shift(-h) / d["close"] - 1

        # --- Sizing target: risk-adjusted optimal leverage ---
        # Forward returns
        ret_cols = [f"ret_{h}" for h in range(1, HORIZON + 1)]
        fwd_mean = d[ret_cols].mean(axis=1)
        fwd_std = d[ret_cols].std(axis=1).clip(lower=1e-8)
        fwd_sharpe = fwd_mean / fwd_std

        # Optimal sizing: high sharpe + positive return → high leverage
        # negative return → 0 leverage
        # Scale sharpe to [0, 1] range: sharpe of 2.0 → full leverage
        raw_sizing = (fwd_sharpe / 0.8).clip(0, 1)
        # Soft sigmoid gate: gradual suppression for negative returns
        soft_gate = 1.0 / (1.0 + np.exp(-fwd_mean * 500))
        raw_sizing = raw_sizing * soft_gate
        d["sizing_label"] = raw_sizing

        d.dropna(inplace=True)
        d.reset_index(drop=True, inplace=True)
        return d

    for iv in list(all_dfs.keys()):
        all_dfs[iv] = label_df(all_dfs[iv])
        sizing_stats = all_dfs[iv]["sizing_label"]
        print(f"  {iv}: {len(all_dfs[iv])} labeled | "
              f"Cls dist: {dict(all_dfs[iv]['cls_label'].value_counts().sort_index())} | "
              f"Sizing: mean={sizing_stats.mean():.3f} std={sizing_stats.std():.3f}")

    # ========================================================
    # 3. Build sequences per interval, then merge
    # ========================================================
    reg_cols = [f"ret_{h}" for h in range(1, HORIZON + 1)]

    def build_sequences(d, interval_id):
        """Build (X, cls_y, reg_y, sizing_y, iv_id) arrays."""
        features = d[FEATURE_COLS].values
        cls_labels = d["cls_label"].values
        reg_targets = d[reg_cols].values
        sizing_targets = d["sizing_label"].values

        X_list, cls_list, reg_list, sz_list, iv_list = [], [], [], [], []
        for i in range(len(features) - SEQ_LEN):
            X_list.append(features[i:i + SEQ_LEN])
            cls_list.append(cls_labels[i + SEQ_LEN - 1])
            reg_list.append(reg_targets[i + SEQ_LEN - 1])
            sz_list.append(sizing_targets[i + SEQ_LEN - 1])
            iv_list.append(interval_id)

        return (np.array(X_list), np.array(cls_list),
                np.array(reg_list), np.array(sz_list), np.array(iv_list))

    # Per-interval: temporal split 85/15, then merge
    train_X, train_cls, train_reg, train_sz, train_iv = [], [], [], [], []
    val_X, val_cls, val_reg, val_sz, val_iv = [], [], [], [], []

    for iv, d in all_dfs.items():
        iv_id = INTERVAL_IDS[iv]
        X, cy, ry, sy, iv_arr = build_sequences(d, iv_id)

        if len(X) == 0:
            continue

        split = int(len(X) * 0.85)
        train_X.append(X[:split])
        train_cls.append(cy[:split])
        train_reg.append(ry[:split])
        train_sz.append(sy[:split])
        train_iv.append(iv_arr[:split])

        val_X.append(X[split:])
        val_cls.append(cy[split:])
        val_reg.append(ry[split:])
        val_sz.append(sy[split:])
        val_iv.append(iv_arr[split:])

        print(f"  {iv}: {split} train / {len(X) - split} val sequences")

    train_X = np.concatenate(train_X)
    train_cls = np.concatenate(train_cls)
    train_reg = np.concatenate(train_reg)
    train_sz = np.concatenate(train_sz)
    train_iv = np.concatenate(train_iv)

    val_X = np.concatenate(val_X)
    val_cls = np.concatenate(val_cls)
    val_reg = np.concatenate(val_reg)
    val_sz = np.concatenate(val_sz)
    val_iv = np.concatenate(val_iv)

    print(f"\nTotal: {len(train_X)} train, {len(val_X)} val sequences")

    # ========================================================
    # 4. Scale features (fit on train only)
    # ========================================================
    scaler = StandardScaler()
    n_train, seq, feat = train_X.shape
    scaler.fit(train_X.reshape(-1, feat))

    train_X = scaler.transform(train_X.reshape(-1, feat)).reshape(n_train, seq, feat)
    n_val = len(val_X)
    val_X = scaler.transform(val_X.reshape(-1, feat)).reshape(n_val, seq, feat)

    # ========================================================
    # 5. Compute class weights
    # ========================================================
    class_counts = np.bincount(train_cls.astype(int), minlength=3)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 3
    print(f"Class counts: {class_counts}, weights: {np.round(class_weights, 3)}")

    # ========================================================
    # 6. Convert to tensors
    # ========================================================
    train_X_t = torch.tensor(train_X, dtype=torch.float32).to(device)
    train_cls_t = torch.tensor(train_cls, dtype=torch.long).to(device)
    train_reg_t = torch.tensor(train_reg, dtype=torch.float32).to(device)
    train_sz_t = torch.tensor(train_sz, dtype=torch.float32).to(device)
    train_iv_t = torch.tensor(train_iv, dtype=torch.long).to(device)

    val_X_t = torch.tensor(val_X, dtype=torch.float32).to(device)
    val_cls_t = torch.tensor(val_cls, dtype=torch.long).to(device)
    val_reg_t = torch.tensor(val_reg, dtype=torch.float32).to(device)
    val_sz_t = torch.tensor(val_sz, dtype=torch.float32).to(device)
    val_iv_t = torch.tensor(val_iv, dtype=torch.long).to(device)

    # ========================================================
    # 7. Create model, optimizer, losses
    # ========================================================
    model = INTCFoundationModel(input_dim=len(FEATURE_COLS)).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=10
    )

    cls_loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device),
        label_smoothing=0.05,
    )
    reg_loss_fn = nn.MSELoss()
    sizing_loss_fn = nn.MSELoss()

    # ========================================================
    # 8. Training loop — 3-head loss
    # ========================================================
    EPOCHS = 300
    BATCH_SIZE = 64
    PATIENCE = 25
    best_val_loss = float("inf")
    wait = 0

    # Loss weights: classification 0.35, regression 0.25, sizing 0.40
    W_CLS, W_REG, W_SZ = 0.35, 0.25, 0.40

    n_batches = max(1, len(train_X_t) // BATCH_SIZE)

    print(f"\nTraining with loss weights: cls={W_CLS} reg={W_REG} sizing={W_SZ}")
    print(f"Max leverage target: {MAX_LEVERAGE}x")
    print()

    for epoch in range(EPOCHS):
        model.train()

        # Shuffle training data
        perm = torch.randperm(len(train_X_t), device=device)
        train_X_t = train_X_t[perm]
        train_cls_t = train_cls_t[perm]
        train_reg_t = train_reg_t[perm]
        train_sz_t = train_sz_t[perm]
        train_iv_t = train_iv_t[perm]

        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_sz_loss = 0.0

        for b in range(n_batches):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(train_X_t))

            bx = train_X_t[start:end]
            b_cls = train_cls_t[start:end]
            b_reg = train_reg_t[start:end]
            b_sz = train_sz_t[start:end]
            b_iv = train_iv_t[start:end]

            cls_logits, reg_pred, sizing_pred = model(bx, b_iv)

            loss_cls = cls_loss_fn(cls_logits, b_cls)
            loss_reg = reg_loss_fn(reg_pred, b_reg)
            loss_sz = sizing_loss_fn(sizing_pred.squeeze(-1), b_sz)

            # Risk penalty: penalize high sizing when forward returns are negative
            # This teaches the model to be cautious with leverage
            fwd_mean = b_reg.mean(dim=1)
            risk_mask = (fwd_mean < 0).float()
            risk_penalty = (sizing_pred.squeeze(-1) * risk_mask).mean()

            loss = (W_CLS * loss_cls + W_REG * loss_reg + W_SZ * loss_sz
                    + 0.03 * risk_penalty)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_cls_loss += loss_cls.item()
            epoch_reg_loss += loss_reg.item()
            epoch_sz_loss += loss_sz.item()

        epoch_cls_loss /= n_batches
        epoch_reg_loss /= n_batches
        epoch_sz_loss /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            v_cls_logits, v_reg_pred, v_sizing = model(val_X_t, val_iv_t)
            v_cls_loss = cls_loss_fn(v_cls_logits, val_cls_t).item()
            v_reg_loss = reg_loss_fn(v_reg_pred, val_reg_t).item()
            v_sz_loss = sizing_loss_fn(v_sizing.squeeze(-1), val_sz_t).item()
            v_loss = W_CLS * v_cls_loss + W_REG * v_reg_loss + W_SZ * v_sz_loss

            v_acc = (v_cls_logits.argmax(1) == val_cls_t).float().mean().item()
            v_sz_mean = v_sizing.mean().item()

        scheduler.step(v_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train cls={epoch_cls_loss:.4f} reg={epoch_reg_loss:.4f} sz={epoch_sz_loss:.4f} | "
              f"Val cls={v_cls_loss:.4f} reg={v_reg_loss:.4f} sz={v_sz_loss:.4f} "
              f"acc={v_acc:.4f} sz_avg={v_sz_mean:.3f} | lr={lr:.2e}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            wait = 0
            save_dir = os.path.dirname(os.path.abspath(__file__))
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "INTC_foundation_model.pt"))
            print("  [*] Saved best model!")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  [!] Early stopping at epoch {epoch+1}")
                break

    # Save scaler
    save_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(scaler, os.path.join(save_dir, "INTC_foundation_scaler.pkl"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Saved: INTC_foundation_model.pt, INTC_foundation_scaler.pkl")
