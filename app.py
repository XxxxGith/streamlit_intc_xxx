# ============================================================
# INTC Margin v2 — Live Signal Dashboard (Streamlit Cloud)
# Self-contained: downloads data + runs model in real-time
# ============================================================

import streamlit as st
import os
import time
import numpy as np
import pandas as pd
import torch
import joblib
import altair as alt
import yfinance as yf
from datetime import datetime, timedelta

from INTC_foundation_model import (
    INTCFoundationModel, compute_indicators, compute_leverage,
    FEATURE_COLS, INTERVAL_IDS, SEQ_LEN, MARGIN_CONFIG,
)

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="INTC Live Trader — Margin v2",
    page_icon="📈",
    layout="wide",
)

# ============================================================
# Constants
# ============================================================
TICKER = "INTC"
FORECAST_THRESHOLD = 0.0005
MAX_LEVERAGE = MARGIN_CONFIG["max_leverage"]
INITIAL_CAPITAL = 10_000.0
_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Model loading (cached)
# ============================================================
@st.cache_resource
def load_model():
    device = torch.device("cpu")  # Cloud: use CPU
    model_path = os.path.join(_DIR, "INTC_foundation_model.pt")
    scaler_path = os.path.join(_DIR, "INTC_foundation_scaler.pkl")

    model = INTCFoundationModel(input_dim=len(FEATURE_COLS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler, device


def normalize(df_raw):
    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = df.columns.str.lower()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    if "datetime" in [c.lower() for c in df.columns]:
        dt_col = [c for c in df.columns if c.lower() == "datetime"][0]
    elif "date" in [c.lower() for c in df.columns]:
        dt_col = [c for c in df.columns if c.lower() == "date"][0]
    else:
        dt_col = df.columns[0]
    df.rename(columns={dt_col: "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def get_signal(model, scaler, device, df, interval_str):
    d = compute_indicators(df)
    if len(d) < SEQ_LEN:
        return None

    feature_data = d[FEATURE_COLS].tail(SEQ_LEN).values
    seq_scaled = scaler.transform(feature_data)
    X_t = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    iv_id = torch.tensor([INTERVAL_IDS.get(interval_str, 1)], dtype=torch.long).to(device)

    with torch.no_grad():
        cls_logits, reg_pred, sizing_out = model(X_t, iv_id)
        probs = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
        pred_cls = cls_logits.argmax(1).item()
        forecast = reg_pred[0].cpu().numpy()
        model_sizing = float(sizing_out.squeeze().cpu().numpy())

    cls_labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
    cls_label = cls_labels[pred_cls]
    avg_return = float(forecast.mean())
    forecast_std = float(forecast.std())

    if pred_cls == 2 and avg_return > 0:
        signal = 1
    elif pred_cls == 0 and avg_return < 0:
        signal = -1
    elif avg_return > FORECAST_THRESHOLD:
        signal = 1
    elif avg_return < -FORECAST_THRESHOLD:
        signal = -1
    else:
        signal = 0

    signal_name = {1: "BUY", -1: "SELL", 0: "HOLD"}[signal]

    recommended_lev = model_sizing * MAX_LEVERAGE
    heuristic_lev = compute_leverage(probs.tolist(), avg_return, forecast_std, MAX_LEVERAGE)
    target_leverage = 0.75 * recommended_lev + 0.25 * heuristic_lev
    target_leverage = max(1.0, min(target_leverage, MAX_LEVERAGE))

    return {
        "signal": signal_name,
        "cls_label": cls_label,
        "probs": probs,
        "forecast": forecast,
        "avg_return": avg_return,
        "forecast_std": forecast_std,
        "model_sizing": model_sizing,
        "target_leverage": target_leverage,
        "model_lev": recommended_lev,
        "heuristic_lev": heuristic_lev,
    }


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("Settings")
interval = st.sidebar.selectbox("Interval", ["5m", "15m", "30m", "60m"], index=0)
period = st.sidebar.selectbox("Lookback", ["5d", "1mo", "3mo"], index=0)
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=True)

st.sidebar.divider()
st.sidebar.markdown("**Margin v2 Config**")
st.sidebar.markdown(f"- Max Leverage: **{MAX_LEVERAGE}x**")
st.sidebar.markdown(f"- Margin Ratio: **150%**")
st.sidebar.markdown(f"- Stop-Loss: **5%**")
st.sidebar.markdown(f"- Interest: **8%/yr**")
st.sidebar.markdown(f"- Threshold: **{FORECAST_THRESHOLD*100:.2f}%**")

# ============================================================
# Load model + data
# ============================================================
model, scaler, device = load_model()

with st.spinner("Downloading market data..."):
    df_raw = yf.download(TICKER, period=period, interval=interval, progress=False, prepost=True)

if df_raw is None or len(df_raw) == 0:
    st.error("No market data available. Market may be closed.")
    st.stop()

df = normalize(df_raw)
latest_price = float(df["close"].iloc[-1])
latest_time = df["datetime"].iloc[-1]

# ============================================================
# Title
# ============================================================
st.title("INTC Live Signal Dashboard — Margin v2 (2.5x)")
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
           f"Data: {len(df)} bars @ {interval} | Latest bar: {latest_time}")

# ============================================================
# Tabs
# ============================================================
tab_signal, tab_paper, tab_log, tab_scanner, tab_system = st.tabs(["Current Signal", "Paper Trading", "Signal Log", "Multi-Bar Scanner", "System & Rules"])

# ============================================================
# TAB 1: Current Signal
# ============================================================
with tab_signal:
    result = get_signal(model, scaler, device, df, interval)

    if result is None:
        st.warning("Not enough data to generate signal.")
    else:
        # Big signal display
        sig = result["signal"]
        sig_colors = {"BUY": "green", "SELL": "red", "HOLD": "gray"}
        sig_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}

        col_sig, col_price = st.columns([1, 1])
        with col_sig:
            st.markdown(f"### {sig_emoji[sig]} Signal: **:{sig_colors[sig]}[{sig}]**")
            st.markdown(f"**Classification:** {result['cls_label']} | "
                        f"**Target Leverage:** {result['target_leverage']:.2f}x")
        with col_price:
            st.metric("INTC Price", f"${latest_price:.4f}")
            if result["avg_return"] != 0:
                st.metric("Forecast Avg Move", f"{result['avg_return']*100:+.4f}%")

        st.divider()

        # Detailed outputs
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Classification Probabilities**")
            prob_df = pd.DataFrame({
                "Class": ["SELL", "HOLD", "BUY"],
                "Probability": result["probs"],
            })
            bar_chart = alt.Chart(prob_df).mark_bar().encode(
                x=alt.X("Class:N", sort=["SELL", "HOLD", "BUY"]),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Class:N", scale=alt.Scale(
                    domain=["SELL", "HOLD", "BUY"],
                    range=["#cc0000", "#888888", "#00cc00"]
                ), legend=None),
            ).properties(height=200)
            st.altair_chart(bar_chart, use_container_width=True)

        with c2:
            st.markdown("**5-Step Return Forecast**")
            fc_df = pd.DataFrame({
                "Step": [f"t+{i+1}" for i in range(len(result["forecast"]))],
                "Return (%)": result["forecast"] * 100,
            })
            fc_chart = alt.Chart(fc_df).mark_bar().encode(
                x="Step:N",
                y="Return (%):Q",
                color=alt.condition(
                    alt.datum["Return (%)"] > 0,
                    alt.value("#00cc00"),
                    alt.value("#cc0000"),
                ),
            ).properties(height=200)
            st.altair_chart(fc_chart, use_container_width=True)

        with c3:
            st.markdown("**Leverage Breakdown**")
            st.markdown(f"""
            | Component | Value |
            |-----------|-------|
            | Model Sizing | {result['model_sizing']:.3f} |
            | Model Lev | {result['model_lev']:.2f}x |
            | Heuristic Lev | {result['heuristic_lev']:.2f}x |
            | **Final Lev** | **{result['target_leverage']:.2f}x** |
            | Blend | 75% model + 25% heuristic |
            """)

            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Avg Return | {result['avg_return']*100:+.4f}% |
            | Forecast Std | {result['forecast_std']*100:.4f}% |
            | Threshold | {FORECAST_THRESHOLD*100:.2f}% |
            """)

        st.divider()

        # Price chart
        st.subheader("Price Chart (last 100 bars)")
        chart_data = df[["datetime", "close"]].tail(100).copy()
        chart_data.columns = ["Time", "Price"]
        price_chart = alt.Chart(chart_data).mark_line(color="steelblue", strokeWidth=1.5).encode(
            x="Time:T",
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False)),
        ).properties(height=300)
        st.altair_chart(price_chart, use_container_width=True)

# ============================================================
# TAB 2: Paper Trading Simulation (Jan 1 2026 -> now)
# ============================================================
with tab_paper:
    st.subheader("Paper Trading Simulation — Jan 1, 2026 to Present")
    st.caption("Simulates $10,000 paper trading using Margin v2 (2.5x) on 5m bars since Jan 1, 2026")

    pt_interval = st.selectbox("Backtest Interval", ["5m", "15m", "60m"], index=0, key="pt_interval")

    # yfinance limits: 5m max 60 days, 15m max 60 days, 60m max 730 days
    if pt_interval in ["5m", "15m"]:
        pt_period = "60d"
        pt_note = "Note: yfinance limits 5m/15m data to last 60 days"
    else:
        pt_period = "1y"
        pt_note = "Using 1-year lookback for 60m interval"

    st.info(pt_note)

    @st.cache_data(ttl=300)
    def download_backtest_data(ticker, period, intv):
        raw = yf.download(ticker, period=period, interval=intv, progress=False, prepost=True)
        return raw

    with st.spinner("Downloading backtest data..."):
        pt_raw = download_backtest_data(TICKER, pt_period, pt_interval)

    if pt_raw is None or len(pt_raw) == 0:
        st.error("No data available for backtest.")
    else:
        pt_df = normalize(pt_raw)

        # Filter to 2026 only
        pt_df = pt_df[pt_df["datetime"] >= "2026-01-01"].reset_index(drop=True)

        if len(pt_df) < SEQ_LEN + 10:
            st.warning(f"Not enough 2026 data. Only {len(pt_df)} bars available.")
        else:
            d_bt = compute_indicators(pt_df)

            # Paper trading state
            cash = INITIAL_CAPITAL
            shares = 0.0
            entry_price = 0.0
            leverage_used = 0.0
            trades = []
            log_lines = []
            equity_curve = []
            margin_limit = INITIAL_CAPITAL * MARGIN_CONFIG["margin_ratio"]
            total_interest = 0.0
            stop_losses = 0
            interval_hours = {"5m": 5/60, "15m": 0.25, "60m": 1.0}
            hours = interval_hours.get(pt_interval, 5/60)

            log_lines.append("=" * 70)
            log_lines.append("  INTC MARGIN v2 — PAPER TRADING SIMULATION")
            log_lines.append("=" * 70)
            log_lines.append(f"Period: 2026-01-01 to present | Interval: {pt_interval}")
            log_lines.append(f"Capital: $10,000 | Max Leverage: {MAX_LEVERAGE}x | Margin: 150%")
            log_lines.append(f"Stop-Loss: 5% | Interest: 8%/yr | Threshold: {FORECAST_THRESHOLD*100:.2f}%")
            log_lines.append(f"Total bars: {len(d_bt)}")
            log_lines.append("-" * 70)

            progress = st.progress(0)
            log_placeholder = st.empty()
            n_bars = len(d_bt) - SEQ_LEN

            for i in range(n_bars):
                idx = SEQ_LEN + i
                window = d_bt.iloc[:idx + 1]
                price = float(window["close"].iloc[-1])
                bar_time = str(pt_df["datetime"].iloc[idx]) if idx < len(pt_df) else ""

                # Charge interest
                margin_used = max(0, -cash)
                if margin_used > 0:
                    interest = margin_used * MARGIN_CONFIG["margin_interest_annual"] / (252 * 6.5) * hours
                    cash -= interest
                    total_interest += interest

                # Stop-loss check
                if shares > 0:
                    pnl_pct = (price - entry_price) / entry_price
                    threshold = MARGIN_CONFIG["stop_loss_leveraged"] if leverage_used > 1.0 else MARGIN_CONFIG["stop_loss_base"]
                    if pnl_pct < -threshold:
                        proceeds = shares * price
                        pnl = proceeds - (shares * entry_price)
                        pnl_pct_val = pnl_pct * 100
                        cash += proceeds
                        trades.append({"time": bar_time, "entry": entry_price, "exit": price,
                                       "shares": shares, "pnl": pnl, "pnl_pct": pnl_pct_val,
                                       "leverage": leverage_used, "type": "STOP-LOSS"})
                        log_lines.append(f"[{bar_time}] *** STOP-LOSS *** {shares:.1f}sh @ ${price:.4f} | "
                                         f"PnL: ${pnl:+.2f} ({pnl_pct_val:+.2f}%) | Lev={leverage_used:.1f}x")
                        shares = 0.0
                        entry_price = 0.0
                        leverage_used = 0.0
                        stop_losses += 1
                        equity_curve.append({"bar": i, "time": bar_time, "equity": cash, "price": price})
                        continue

                # Run model
                feature_data = window[FEATURE_COLS].tail(SEQ_LEN).values
                seq_scaled = scaler.transform(feature_data)
                X_t = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                iv_id = torch.tensor([INTERVAL_IDS.get(pt_interval, 1)], dtype=torch.long).to(device)

                with torch.no_grad():
                    cls_logits, reg_pred, sizing_out = model(X_t, iv_id)
                    probs = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
                    pred_cls = cls_logits.argmax(1).item()
                    forecast = reg_pred[0].cpu().numpy()
                    model_sizing = float(sizing_out.squeeze().cpu().numpy())

                cls_labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
                cls_label = cls_labels[pred_cls]
                avg_ret = float(forecast.mean())
                forecast_std = float(forecast.std())

                if pred_cls == 2 and avg_ret > 0:
                    sig = 1
                elif pred_cls == 0 and avg_ret < 0:
                    sig = -1
                elif avg_ret > FORECAST_THRESHOLD:
                    sig = 1
                elif avg_ret < -FORECAST_THRESHOLD:
                    sig = -1
                else:
                    sig = 0

                rec_lev = model_sizing * MAX_LEVERAGE
                heur_lev = compute_leverage(probs.tolist(), avg_ret, forecast_std, MAX_LEVERAGE)
                target_lev = max(1.0, min(0.75 * rec_lev + 0.25 * heur_lev, MAX_LEVERAGE))

                sig_name = {1: "BUY", -1: "SELL", 0: "HOLD"}[sig]

                # Execute
                if sig == 1 and shares == 0:
                    eq = cash
                    buying_pow = min(eq * target_lev, eq + margin_limit)
                    shares = buying_pow / price
                    cost = shares * price
                    cash -= cost
                    entry_price = price
                    leverage_used = target_lev
                    log_lines.append(f"[{bar_time}] === BUY {shares:.1f}sh @ ${price:.4f} | "
                                     f"${cost:,.0f} | Lev={target_lev:.1f}x | "
                                     f"Class={cls_label}({probs[2]:.0%}) Fcst={avg_ret*100:+.3f}%")

                elif sig == -1 and shares > 0:
                    proceeds = shares * price
                    pnl = proceeds - (shares * entry_price)
                    pnl_pct = (price - entry_price) / entry_price * 100
                    cash += proceeds
                    result = "WIN" if pnl > 0 else "LOSS"
                    trades.append({"time": bar_time, "entry": entry_price, "exit": price,
                                   "shares": shares, "pnl": pnl, "pnl_pct": pnl_pct,
                                   "leverage": leverage_used, "type": "SIGNAL"})
                    log_lines.append(f"[{bar_time}] === SELL {shares:.1f}sh @ ${price:.4f} | "
                                     f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) [{result}] "
                                     f"Lev={leverage_used:.1f}x | Class={cls_label} Fcst={avg_ret*100:+.3f}%")
                    shares = 0.0
                    entry_price = 0.0
                    leverage_used = 0.0

                # Track equity
                eq_now = cash + shares * price
                equity_curve.append({"bar": i, "time": bar_time, "equity": eq_now, "price": price})

                # Update progress
                if i % max(1, n_bars // 100) == 0:
                    progress.progress(min(i / n_bars, 1.0))

            progress.progress(1.0)

            # Final equity
            final_price = float(d_bt["close"].iloc[-1])
            final_eq = cash + shares * final_price
            total_pnl = final_eq - INITIAL_CAPITAL
            total_pnl_pct = total_pnl / INITIAL_CAPITAL * 100
            wins = sum(1 for t in trades if t["pnl"] > 0)
            losses = sum(1 for t in trades if t["pnl"] <= 0)
            win_rate = wins / len(trades) * 100 if trades else 0

            # Buy & hold comparison
            first_price = float(d_bt["close"].iloc[SEQ_LEN])
            bh_return = (final_price - first_price) / first_price * 100

            log_lines.append("")
            log_lines.append("=" * 70)
            log_lines.append("  SIMULATION COMPLETE")
            log_lines.append("=" * 70)
            log_lines.append(f"Final Equity: ${final_eq:,.2f} | P&L: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
            log_lines.append(f"Trades: {len(trades)} | W/L: {wins}/{losses} ({win_rate:.0f}%)")
            log_lines.append(f"Stop-Losses: {stop_losses} | Interest: ${total_interest:.2f}")
            log_lines.append(f"Buy & Hold: {bh_return:+.2f}%")
            if shares > 0:
                log_lines.append(f"Open Position: {shares:.1f}sh @ ${entry_price:.4f} Lev={leverage_used:.1f}x")

            # Summary KPIs
            st.subheader("Results Summary")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Final Equity", f"${final_eq:,.2f}", f"{total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
            k2.metric("Trades", f"{len(trades)}", f"W:{wins} L:{losses}")
            k3.metric("Win Rate", f"{win_rate:.0f}%")
            k4.metric("Buy & Hold", f"{bh_return:+.2f}%")
            k5.metric("Interest Paid", f"${total_interest:.2f}", f"Stops: {stop_losses}")

            # Equity curve chart
            if equity_curve:
                st.subheader("Equity Curve")
                eq_df = pd.DataFrame(equity_curve)

                # Downsample for chart if too many points
                if len(eq_df) > 500:
                    step = len(eq_df) // 500
                    eq_df_chart = eq_df.iloc[::step]
                else:
                    eq_df_chart = eq_df

                eq_chart = alt.Chart(eq_df_chart).mark_line(color="gold", strokeWidth=2).encode(
                    x=alt.X("bar:Q", title="Bar #"),
                    y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False)),
                ).properties(height=300)

                bh_line = alt.Chart(eq_df_chart).mark_line(color="steelblue", strokeDash=[5,3]).encode(
                    x="bar:Q",
                    y=alt.Y("price:Q", title="INTC Price ($)", scale=alt.Scale(zero=False)),
                )

                st.altair_chart(eq_chart, use_container_width=True)

                st.subheader("INTC Price")
                st.altair_chart(bh_line, use_container_width=True)

            # Trade table
            if trades:
                st.subheader("Trade Log")
                t_df = pd.DataFrame(trades)
                t_df_disp = t_df[["time", "entry", "exit", "shares", "pnl", "pnl_pct", "leverage", "type"]].copy()
                t_df_disp.columns = ["Time", "Entry $", "Exit $", "Shares", "P&L $", "P&L %", "Leverage", "Type"]

                def color_pnl(val):
                    try:
                        v = float(val)
                        if v > 0: return "color: #00cc00"
                        elif v < 0: return "color: #cc0000"
                    except (ValueError, TypeError):
                        pass
                    return ""

                styled_t = t_df_disp.style
                for c in ["P&L $", "P&L %"]:
                    styled_t = styled_t.applymap(color_pnl, subset=[c])
                st.dataframe(styled_t, use_container_width=True, height=400)

            # Full log
            with st.expander("Full Trading Log"):
                st.code("\n".join(log_lines), language="text")

# ============================================================
# TAB 3: Signal Log (live model log on recent bars)
# ============================================================
with tab_log:
    st.subheader("Live Signal Log")
    st.caption("Model runs on each recent bar — simulates what the paper trader sees in real-time")

    n_log = st.slider("Number of bars to log", 10, 100, 30, key="log_slider")
    d_log = compute_indicators(df)

    if len(d_log) >= SEQ_LEN + n_log:
        log_lines = []
        log_data = []

        for i in range(n_log):
            idx = len(d_log) - n_log + i
            window = d_log.iloc[:idx + 1]
            if len(window) < SEQ_LEN:
                continue

            feature_data = window[FEATURE_COLS].tail(SEQ_LEN).values
            seq_scaled = scaler.transform(feature_data)
            X_t = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            iv_id = torch.tensor([INTERVAL_IDS.get(interval, 1)], dtype=torch.long).to(device)

            with torch.no_grad():
                cls_logits, reg_pred, sizing_out = model(X_t, iv_id)
                probs = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
                pred_cls = cls_logits.argmax(1).item()
                forecast = reg_pred[0].cpu().numpy()
                model_sizing = float(sizing_out.squeeze().cpu().numpy())

            cls_labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
            cls_label = cls_labels[pred_cls]
            avg_ret = float(forecast.mean())
            forecast_std = float(forecast.std())

            if pred_cls == 2 and avg_ret > 0:
                sig = "BUY"
            elif pred_cls == 0 and avg_ret < 0:
                sig = "SELL"
            elif avg_ret > FORECAST_THRESHOLD:
                sig = "BUY"
            elif avg_ret < -FORECAST_THRESHOLD:
                sig = "SELL"
            else:
                sig = "HOLD"

            rec_lev = model_sizing * MAX_LEVERAGE
            heur_lev = compute_leverage(probs.tolist(), avg_ret, forecast_std, MAX_LEVERAGE)
            target_lev = max(1.0, min(0.75 * rec_lev + 0.25 * heur_lev, MAX_LEVERAGE))

            price = float(window["close"].iloc[-1])
            bar_time = str(df["datetime"].iloc[-(n_log - i)])
            fc_str = " | ".join(f"{v*100:+.3f}%" for v in forecast)

            # Build log line
            sig_icon = {"BUY": ">>>BUY<<<", "SELL": ">>>SELL<<<", "HOLD": "---HOLD---"}[sig]
            line = (
                f"[{bar_time}] INTC=${price:.4f}\n"
                f"  SIGNAL: {sig_icon}\n"
                f"  Classification: {cls_label} (SELL={probs[0]:.3f} HOLD={probs[1]:.3f} BUY={probs[2]:.3f})\n"
                f"  Forecast: {fc_str}\n"
                f"  Avg move: {avg_ret*100:+.4f}% | Std: {forecast_std*100:.4f}% | Sizing: {model_sizing:.3f}\n"
                f"  Leverage: {target_lev:.2f}x (model: {rec_lev:.2f}x, heuristic: {heur_lev:.2f}x)\n"
            )

            # Check if classification disagrees with signal
            if cls_label != sig:
                line += f"  ** OVERRIDE: Class={cls_label} but Signal={sig} (regression forecast overrides)\n"

            log_lines.append(line)
            log_data.append({
                "Time": bar_time, "Price": price, "Class": cls_label,
                "Signal": sig, "Override": cls_label != sig,
                "BUY%": probs[2], "SELL%": probs[0],
                "Avg Move": avg_ret * 100, "Leverage": target_lev,
            })

        # Show raw log
        st.code("\n".join(log_lines), language="text")

        st.divider()

        # Classification vs Signal Analysis
        st.subheader("Classification vs Signal — When Do They Disagree?")

        st.markdown("""
        The model has **3 output heads** that can conflict with each other:

        | Head | What it does | Example |
        |------|-------------|---------|
        | **Classification** (Head 1) | Pattern recognition: "This chart looks like a BUY/SELL/HOLD setup" | BUY 82% |
        | **Regression** (Head 2) | Price forecast: "I predict the next 5 moves will be..." | -0.12%, -0.25%, -0.33%... |
        | **Sizing** (Head 3) | Position sizing: "How confident am I? How much to bet?" | 0.35 |

        **The Signal** is the **final decision** that combines Head 1 + Head 2:
        """)

        st.markdown("""
        ```
        IF class = BUY  AND forecast > 0     --> SIGNAL = BUY   (both agree: bullish)
        IF class = SELL AND forecast < 0     --> SIGNAL = SELL  (both agree: bearish)
        IF forecast > +0.05%                 --> SIGNAL = BUY   (strong forecast overrides)
        IF forecast < -0.05%                 --> SIGNAL = SELL  (strong forecast overrides)
        OTHERWISE                            --> SIGNAL = HOLD
        ```
        """)

        st.info(
            "**Key insight:** Classification sees *patterns* (\"this looks bullish\"), "
            "while Regression predicts *actual returns* (\"but price will drop\"). "
            "When they disagree, the regression forecast wins — this prevents buying into "
            "patterns that historically looked bullish but the model's forward prediction says otherwise."
        )

        # Show override statistics
        log_df = pd.DataFrame(log_data)
        n_overrides = log_df["Override"].sum()
        n_total = len(log_df)

        col_o1, col_o2, col_o3 = st.columns(3)
        col_o1.metric("Total Bars Analyzed", n_total)
        col_o2.metric("Overrides (Class != Signal)", int(n_overrides))
        col_o3.metric("Override Rate", f"{n_overrides/n_total*100:.0f}%" if n_total > 0 else "0%")

        if n_overrides > 0:
            st.markdown("**Override Details:**")
            override_df = log_df[log_df["Override"]][["Time", "Price", "Class", "Signal", "Avg Move", "BUY%", "SELL%"]]
            override_df.columns = ["Time", "Price", "Classification", "Final Signal", "Forecast %", "BUY Prob", "SELL Prob"]
            override_df = override_df.iloc[::-1].reset_index(drop=True)

            def color_cls(val):
                if val == "BUY": return "background-color: #0d6e0d; color: white"
                elif val == "SELL": return "background-color: #8b0000; color: white"
                return ""

            styled_o = override_df.style.applymap(color_cls, subset=["Classification", "Final Signal"])
            st.dataframe(styled_o, use_container_width=True)

            st.markdown("""
            **Example from above:** When Classification says BUY (82%) but the 5-step forecast
            is negative (-0.23%), the final Signal becomes SELL. The model sees a bullish *pattern*
            but predicts the price will actually *drop* — so it refuses to buy. This is the safety
            mechanism that prevents entering bad trades.
            """)

        # Signal vs Classification comparison chart
        st.subheader("Signal vs Classification Distribution")
        comp_data = []
        for _, row in log_df.iterrows():
            comp_data.append({"Type": "Classification", "Value": row["Class"]})
            comp_data.append({"Type": "Final Signal", "Value": row["Signal"]})
        comp_df = pd.DataFrame(comp_data)
        comp_chart = alt.Chart(comp_df).mark_bar().encode(
            x="Value:N",
            y="count():Q",
            color=alt.Color("Value:N", scale=alt.Scale(
                domain=["BUY", "SELL", "HOLD"],
                range=["#00cc00", "#cc0000", "#888888"]
            ), legend=None),
            column="Type:N",
        ).properties(width=250, height=200)
        st.altair_chart(comp_chart)

    else:
        st.warning(f"Not enough data for {n_log} bars.")

# ============================================================
# TAB 3: Multi-Bar Scanner
# ============================================================
with tab_scanner:
    st.subheader("Rolling Signal Scanner")
    st.caption("Runs the model on sliding windows to show signal history")

    n_scan = st.slider("Bars to scan back", 5, 50, 20)

    signals_list = []
    d_full = compute_indicators(df)

    if len(d_full) >= SEQ_LEN + n_scan:
        for i in range(n_scan):
            idx = len(d_full) - n_scan + i
            window = d_full.iloc[:idx + 1]

            if len(window) < SEQ_LEN:
                continue

            feature_data = window[FEATURE_COLS].tail(SEQ_LEN).values
            seq_scaled = scaler.transform(feature_data)
            X_t = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            iv_id = torch.tensor([INTERVAL_IDS.get(interval, 1)], dtype=torch.long).to(device)

            with torch.no_grad():
                cls_logits, reg_pred, sizing_out = model(X_t, iv_id)
                probs = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
                pred_cls = cls_logits.argmax(1).item()
                forecast = reg_pred[0].cpu().numpy()
                model_sizing = float(sizing_out.squeeze().cpu().numpy())

            avg_ret = float(forecast.mean())
            forecast_std = float(forecast.std())

            if pred_cls == 2 and avg_ret > 0:
                sig = "BUY"
            elif pred_cls == 0 and avg_ret < 0:
                sig = "SELL"
            elif avg_ret > FORECAST_THRESHOLD:
                sig = "BUY"
            elif avg_ret < -FORECAST_THRESHOLD:
                sig = "SELL"
            else:
                sig = "HOLD"

            cls_labels = {0: "SELL", 1: "HOLD", 2: "BUY"}
            rec_lev = model_sizing * MAX_LEVERAGE
            heur_lev = compute_leverage(probs.tolist(), avg_ret, forecast_std, MAX_LEVERAGE)
            target_lev = max(1.0, min(0.75 * rec_lev + 0.25 * heur_lev, MAX_LEVERAGE))

            price = float(window["close"].iloc[-1])
            bar_time = str(df["datetime"].iloc[-(n_scan - i)])

            signals_list.append({
                "Time": bar_time,
                "Price": f"${price:.4f}",
                "Signal": sig,
                "Class": cls_labels[pred_cls],
                "BUY%": f"{probs[2]:.1%}",
                "SELL%": f"{probs[0]:.1%}",
                "Avg Move": f"{avg_ret*100:+.3f}%",
                "Sizing": f"{model_sizing:.3f}",
                "Leverage": f"{target_lev:.2f}x",
            })

        if signals_list:
            scan_df = pd.DataFrame(signals_list)

            def color_sig(val):
                if val == "BUY":
                    return "background-color: #0d6e0d; color: white"
                elif val == "SELL":
                    return "background-color: #8b0000; color: white"
                return ""

            styled = scan_df.style.applymap(color_sig, subset=["Signal"])
            st.dataframe(styled, use_container_width=True, height=500)

            # Signal distribution
            st.subheader("Signal Distribution")
            dist = scan_df["Signal"].value_counts().reset_index()
            dist.columns = ["Signal", "Count"]
            dist_chart = alt.Chart(dist).mark_bar().encode(
                x="Signal:N",
                y="Count:Q",
                color=alt.Color("Signal:N", scale=alt.Scale(
                    domain=["BUY", "SELL", "HOLD"],
                    range=["#00cc00", "#cc0000", "#888888"]
                ), legend=None),
            ).properties(height=200)
            st.altair_chart(dist_chart, use_container_width=True)

            # Price with signals overlay
            st.subheader("Price with Signals")
            overlay_data = []
            for s in signals_list:
                overlay_data.append({
                    "Time": s["Time"],
                    "Price": float(s["Price"].replace("$", "")),
                    "Signal": s["Signal"],
                })
            overlay_df = pd.DataFrame(overlay_data)

            base = alt.Chart(overlay_df).encode(x="Time:N")
            line = base.mark_line(color="steelblue").encode(y=alt.Y("Price:Q", scale=alt.Scale(zero=False)))
            points = base.mark_circle(size=120).encode(
                y="Price:Q",
                color=alt.Color("Signal:N", scale=alt.Scale(
                    domain=["BUY", "SELL", "HOLD"],
                    range=["#00cc00", "#cc0000", "#888888"]
                )),
            )
            st.altair_chart(line + points, use_container_width=True)
    else:
        st.warning(f"Not enough data to scan {n_scan} bars. Need at least {SEQ_LEN + n_scan} bars.")

# ============================================================
# TAB 3: System & Rules
# ============================================================
with tab_system:
    st.subheader("System Architecture")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    DATA FLOW PIPELINE                          │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  yfinance API (prepost=True)                                   │
    │       │  INTC 5d/5m candles (pre-market + regular + after-hrs) │
    │       ▼                                                         │
    │  Raw OHLCV Data (Open, High, Low, Close, Volume)               │
    │       │                                                         │
    │       ▼                                                         │
    │  compute_indicators()  ──────────────────────────────────┐     │
    │       │  15 Features:                                     │     │
    │       │  • OHLCV (5)                                      │     │
    │       │  • MA fast/slow (2)                                │     │
    │       │  • MACD + signal line (2)                          │     │
    │       │  • RSI (1)                                         │     │
    │       │  • VWAP (1)                                        │     │
    │       │  • Bollinger upper/lower (2)                       │     │
    │       │  • Recent high/low (2)                             │     │
    │       ▼                                                         │
    │  StandardScaler (INTC_foundation_scaler.pkl)                   │
    │       │  Normalize features to zero-mean, unit-variance         │
    │       ▼                                                         │
    │  Sequence Window [64 bars x 15 features]                       │
    │       │  + Interval ID (5m = id 1)                             │
    │       ▼                                                         │
    │  ┌─────────────────────────────────────────────────────┐       │
    │  │       INTCFoundationModel (1.23M params)            │       │
    │  │       Transformer Encoder (4 heads, 2 layers)       │       │
    │  │                                                     │       │
    │  │  Input --> Embedding --> Transformer --> 3 Heads     │       │
    │  │                                                     │       │
    │  │  Head 1: Classification  --> [SELL, HOLD, BUY]      │       │
    │  │  Head 2: Regression      --> [5-step return forecast]│      │
    │  │  Head 3: Sizing          --> [0.0 - 1.0]           │       │
    │  └─────────────────────────────────────────────────────┘       │
    │       │                                                         │
    │       ▼                                                         │
    │  SIGNAL DECISION ENGINE                                        │
    │       │                                                         │
    │       ├── BUY if: class=BUY AND avg_return > 0                 │
    │       ├── BUY if: avg_return > 0.05% (forecast threshold)      │
    │       ├── SELL if: class=SELL AND avg_return < 0               │
    │       ├── SELL if: avg_return < -0.05%                         │
    │       └── HOLD otherwise                                       │
    │       │                                                         │
    │       ▼                                                         │
    │  LEVERAGE CALCULATOR                                           │
    │       │  Model:     sizing x 2.5 (max leverage)                │
    │       │  Heuristic: confidence + return tiers                  │
    │       │  Final:     75% model + 25% heuristic                  │
    │       │  Clamp:     [1.0x, 2.5x]                               │
    │       ▼                                                         │
    │  OUTPUT: Signal + Leverage Recommendation                      │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.divider()
    st.subheader("Model Configuration")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model Architecture**")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Model | INTCFoundationModel |
        | Parameters | 1,226,249 |
        | Encoder | Transformer (4 heads, 2 layers) |
        | Sequence Length | 64 bars |
        | Features | 15 technical indicators |
        | Horizon | 5-step forward |
        | Output Heads | 3 (cls + reg + sizing) |
        """)

        st.markdown("**Training Config**")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Loss Weights | CLS=0.35, REG=0.25, SZ=0.40 |
        | Label Smoothing | 0.05 |
        | Risk Penalty | 0.03 |
        | Sizing Labels | fwd_sharpe / 0.8, clipped [0,1] |
        | Negative Gate | Soft sigmoid (exp(-mean*500)) |
        """)

    with c2:
        st.markdown("**15 Input Features**")
        st.markdown("""
        | # | Feature | Description |
        |---|---------|-------------|
        | 1-5 | OHLCV | Open, High, Low, Close, Volume |
        | 6-7 | MA fast/slow | 10/30 period moving averages |
        | 8-9 | MACD + Signal | EMA12-EMA26, 9-period signal |
        | 10 | RSI | 14-period Relative Strength Index |
        | 11 | VWAP | Volume-Weighted Average Price |
        | 12-13 | BB upper/lower | 20-period Bollinger Bands |
        | 14-15 | Recent H/L | 20-period high/low breakout |
        """)

    st.divider()
    st.subheader("Margin & Risk Rules")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Margin Configuration**")
        st.markdown("""
        | Rule | Value |
        |------|-------|
        | Initial Capital | $10,000 |
        | Margin Ratio | 150% (borrow up to 1.5x equity) |
        | Max Leverage | 2.5x ($25,000 buying power) |
        | Maintenance Margin | 25% (Reg T) |
        | Margin Interest | 8% annual |
        """)

        st.markdown("**Stop-Loss Rules**")
        st.markdown("""
        | Condition | Threshold |
        |-----------|-----------|
        | Base (1x leverage) | -5% from entry |
        | Leveraged (>1x) | -5% from entry |
        | Margin Call | Equity / Position < 25% |
        """)

    with r2:
        st.markdown("**Signal Decision Rules**")
        st.markdown("""
        | Condition | Signal |
        |-----------|--------|
        | class=BUY AND avg_return > 0 | **BUY** |
        | avg_return > +0.05% | **BUY** |
        | class=SELL AND avg_return < 0 | **SELL** |
        | avg_return < -0.05% | **SELL** |
        | Otherwise | **HOLD** |
        """)

        st.markdown("**Leverage Tiers (Heuristic)**")
        st.markdown("""
        | Confidence | Return | Leverage |
        |-----------|--------|----------|
        | prob >= 0.70 | abs(ret) > 0.08% | 2.5x |
        | prob >= 0.55 | abs(ret) > 0.04% | 2.0x |
        | prob >= 0.45 | abs(ret) > 0.01% | 1.5x |
        | below thresholds | -- | 1.0x |
        """)

        st.markdown("**Final Leverage:** `0.75 * model + 0.25 * heuristic`, clamped [1.0, 2.5]")

# ============================================================
# Auto-refresh
# ============================================================
if auto_refresh:
    time.sleep(60)
    st.rerun()
