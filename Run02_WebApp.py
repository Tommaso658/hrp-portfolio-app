"""
Streamlit app – HRP Portfolio Replica (v4)
Updated: 27‑May‑2025

Changelog v4
------------
* **Usage warning**: script must be launched with `streamlit run`, not with plain `python`, to avoid runtime/session‑state errors.
* **NaN / Inf safeguard**: rolling windows now drop assets containing NaN or Inf before computing covariance and clustering.
* **Graceful skip**: if fewer than two driver assets remain after cleaning, that week is skipped (weights = previous).
* Minor docstrings & comments.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import plotly.graph_objects as go
from io import BytesIO
from pathlib import Path

# -------------------------  Data loader  ------------------------- #
@st.cache_data(show_spinner=False)
def load_prices(source) -> pd.DataFrame:
    """Load weekly close prices from Excel, coercing to a clean numeric DataFrame."""
    df = pd.read_excel(source, sheet_name="Copia_statica", engine="openpyxl")
    # Coerce layout to (dates index, tickers columns)
    if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]) or "date" in str(df.columns[0]).lower():
        df.set_index(df.columns[0], inplace=True)
    elif pd.api.types.is_datetime64_any_dtype(df.columns):  # dates on header row – transpose
        df = df.set_index(df.columns).T
    else:
        raise ValueError("Unexpected sheet layout – please check the Excel file.")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)
    return df

# -------------------------  HRP utilities  ----------------------- #

def _correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert correlation to distance; replace NaN (from const series) with 0."""
    dist = ((1.0 - corr) / 2.0) ** 0.5
    return dist.fillna(0)


def _seriation(Z: np.ndarray, N: int, cur_index: int) -> list:
    if cur_index < N:
        return [cur_index]
    left = int(Z[cur_index - N, 0])
    right = int(Z[cur_index - N, 1])
    return _seriation(Z, N, left) + _seriation(Z, N, right)


def _get_quasi_diag(corr: pd.DataFrame) -> list:
    dist = squareform(_correl_dist(corr).values, checks=False)
    link = linkage(dist, method="single")
    return _seriation(link, len(corr), 2 * len(corr) - 2)


def _cluster_var(cov: pd.DataFrame, w: pd.Series, assets: list[str]) -> float:
    sub_w = w[assets].values
    sub_cov = cov.loc[assets, assets].values
    return float(sub_w.T @ sub_cov @ sub_w)


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    std = np.sqrt(np.diag(cov))
    corr = cov / std / std[:, None]
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def _get_hrp_weights(cov: pd.DataFrame) -> pd.Series:
    # If only one asset, weight = 1
    if cov.shape[0] == 1:
        return pd.Series({cov.index[0]: 1.0})

    corr = cov_to_corr(cov)
    order = _get_quasi_diag(corr)
    assets = corr.index[order].tolist()
    cov = cov.loc[assets, assets]

    w = pd.Series(1.0 / np.diag(cov), index=assets)
    w /= w.sum()

    clusters = [assets]
    while clusters:
        nxt: list[list[str]] = []
        for cl in clusters:
            if len(cl) <= 2:
                continue
            mid = len(cl) // 2
            left, right = cl[:mid], cl[mid:]
            if len(left) > 1:
                nxt.append(left)
            if len(right) > 1:
                nxt.append(right)
            v_left = _cluster_var(cov, w, left)
            v_right = _cluster_var(cov, w, right)
            if v_left + v_right == 0:
                continue
            alpha = v_right / (v_left + v_right)
            w[left] *= 1 - alpha
            w[right] *= alpha
        clusters = nxt
    return w

# -------------------------  Core engine  ------------------------- #

def get_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()


def rolling_hrp(returns: pd.DataFrame, window: int = 150):
    weights_hist = {}
    for end in range(window, len(returns)):
        win = returns.iloc[end - window : end]
        win = win.dropna(axis=1, how="any")  # drop assets with NaN in window
        if win.shape[1] < 2:
            # Not enough assets to form a portfolio – copy previous weights if any
            continue
        cov = win.cov()
        if not np.isfinite(cov.values).all():
            continue  # skip if still NaN/Inf
        w = _get_hrp_weights(cov)
        weights_hist[returns.index[end]] = w

    W = pd.DataFrame(weights_hist).T.reindex(returns.index).ffill().fillna(0)
    replica_ret = (W.shift() * returns).sum(axis=1)
    return replica_ret.loc[W.index], W


def compute_metrics(series: pd.Series, rf: float = 0.0, freq: int = 52, var_lvl: float = 0.95) -> dict:
    ann_factor = np.sqrt(freq)
    cagr = series.add(1).prod() ** (freq / len(series)) - 1.0
    vol = series.std() * ann_factor
    sharpe = np.nan if vol == 0 else (cagr - rf) / vol
    var = np.percentile(series, (1.0 - var_lvl) * 100)
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, f"VaR ({int(var_lvl*100)}%)": var}

# -------------------------  Streamlit UI  ------------------------ #

def main():
    st.set_page_config(page_title="HRP Portfolio Replica", layout="wide")
    st.title("Portfolio Replica Strategy – HRP")

    with st.sidebar:
        st.header("Dati di input")
        uploaded = st.file_uploader("Carica il dataset Excel", type=["xlsx"])
        note = st.caption("⚠️ Esegui questo script con `streamlit run`, non con `python`, per usare l'interfaccia web.")

    if uploaded:
        prices = load_prices(uploaded)
        st.sidebar.success("File caricato ✅")
    else:
        default = Path(__file__).with_name("Dataset3_PortfolioReplicaStrategyErrataCorrige.xlsx")
        if not default.exists():
            st.sidebar.error("Dataset mancante. Carica un file .xlsx per procedere.")
            st.stop()
        prices = load_prices(default)
        st.sidebar.info(f"Usato file di default: {default.name}")

    targets = ["MXWO", "MXWD", "LEGATRUU", "HFRXGL"]
    drivers = [c for c in prices.columns if c not in targets]

    with st.sidebar:
        st.header("Pesi Target (%)")
        raw = {t: st.slider(t, 0, 100, 25) for t in targets}
        if sum(raw.values()) == 0:
            st.error("La somma dei pesi è zero.")
            st.stop()
        w_t = pd.Series(raw) / sum(raw.values())
        st.caption("Normalizzati a 100 %")
        window = st.number_input("Rolling window (sett.)", 52, 300, 150)
        rf = st.number_input("Risk‑free annuo (%)", 0.0, step=0.05, format="%.2f")
        var_lvl = st.slider("VaR conf. (%)", 90, 99, 95) / 100
        run = st.button("Calcola replica", type="primary")

    if not run:
        st.info("Imposta parametri e premi *Calcola replica*…")
        st.stop()

    rets = get_returns(prices)
    target_ret = rets[targets] @ w_t.values

    st.toast("⏳ HRP rolling optimisation…")
    replica_ret, _ = rolling_hrp(rets[drivers], int(window))

    df = pd.concat([target_ret.rename("Target"), replica_ret.rename("Replica")], axis=1).dropna()
    perf = df.add(1).cumprod()

    fig = go.Figure()
    for col in perf:
        fig.add_trace(go.Scatter(x=perf.index, y=perf[col], name=col))
    fig.update_layout(title="Cumulative Performance", xaxis_title="Date", yaxis_title="Growth of 1", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Metriche di Performance")
    metrics = pd.DataFrame([compute_metrics(df[c], rf=rf / 100, var_lvl=var_lvl) for c in df.columns], index=df.columns)
    st.dataframe(metrics.style.format("{:.2%}"))

    out = BytesIO()
    perf.to_csv(out)
    st.download_button("Download CSV serie storiche", out.getvalue(), "target_replica_series.csv", "text/csv")

if __name__ == "__main__":
    main()
