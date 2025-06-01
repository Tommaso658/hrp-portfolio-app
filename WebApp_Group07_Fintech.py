import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import plotly.graph_objects as go
from io import BytesIO
from pathlib import Path
import os, logging


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

def rolling_hrp_improved(
        returns: pd.DataFrame,
        target_ret: pd.Series,
        window: int = 150,
        add_intercept: bool = True,
        regularization: float = 0.01
    ):
    """
    Rolling OLS migliorato con intercetta corretta
    """
    weights_hist = {}
    intercept_hist = {}  # NUOVO: storia delle intercette

    for end in range(window, len(returns)+1):
        winR = returns.iloc[end-window:end]
        winT = target_ret.iloc[end-window:end]

        if winR.isnull().values.any() or winT.isnull().any():
            continue

        X = winR.values
        y = winT.values
        
        if add_intercept:
            # Aggiungi colonna di 1 per l'intercetta
            X = np.column_stack([np.ones(len(X)), X])
        
        # Ridge regression per stabilità
        if regularization > 0:
            XtX = X.T @ X
            # Aggiungi regolarizzazione solo ai coefficienti, non all'intercetta
            if add_intercept:
                reg_matrix = np.eye(X.shape[1])
                reg_matrix[0, 0] = 0  # Non regolarizzare l'intercetta
                XtX += regularization * reg_matrix
            else:
                XtX += regularization * np.eye(X.shape[1])
            
            coefs = np.linalg.solve(XtX, X.T @ y)
        else:
            coefs = np.linalg.pinv(X) @ y
        
        if add_intercept:
            # SALVA INTERCETTA E PESI SEPARATAMENTE
            intercept_hist[returns.index[end-1]] = coefs[0]
            w = pd.Series(coefs[1:], index=returns.columns)
        else:
            # Senza intercetta, metti 0
            intercept_hist[returns.index[end-1]] = 0.0
            w = pd.Series(coefs, index=returns.columns)

        weights_hist[returns.index[end-1]] = w

    # Crea DataFrame pesi e Serie intercette
    W = (
        pd.DataFrame(weights_hist).T
        .reindex(returns.index)
        .ffill()
        .fillna(0)
    )
    
    I = (
        pd.Series(intercept_hist)
        .reindex(returns.index)
        .ffill()
        .fillna(0)
    )

    # CALCOLO CORRETTO: portfolio + intercetta
    replica_ret = (W * returns).sum(axis=1) + I
    
    return replica_ret, W

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
        
        st.header("Parametri Ottimizzazione")
        window = st.number_input("Rolling window (sett.)", 52, 300, 150)
        add_intercept = st.checkbox("Aggiungi intercetta", value=True)
        regularization = st.slider("Regolarizzazione", 0.0, 0.1, 0.01, step=0.005)
        
        run = st.button("Calcola replica", type="primary")

    if not run:
        st.info("Imposta parametri e premi *Calcola replica*…")
        st.stop()

    def build_target_returns(prices: pd.DataFrame, weights: pd.Series, targets: list):
        """Costruisce target returns come nel main"""
        target_prices = prices[targets]
        target_ret = target_prices.pct_change().mul(weights, axis=1).sum(axis=1)
        return target_ret.dropna()

    # Usa questa funzione invece:
    rets = get_returns(prices)
    target_ret = build_target_returns(prices, w_t, targets)

    # Allinea gli indici
    common_idx = target_ret.index.intersection(rets.index)
    target_ret = target_ret.loc[common_idx]
    rets = rets.loc[common_idx]

    st.toast("⏳ Rolling optimisation…")
    
    # CHIAMATA FUNZIONE (invariata)
    replica_ret, weights_df = rolling_hrp_improved(
        rets[drivers], 
        target_ret, 
        int(window),
        add_intercept=add_intercept,
        regularization=regularization
    )
    
    # NUOVO (AGGIUNGI):
    # Trova la prima data con replica valida (non zero)
    first_valid_replica = replica_ret[replica_ret != 0].index[0]
    st.info(f"🔍 Prima data replica valida: {first_valid_replica}")
    
    # Taglia entrambe le serie dalla prima data valida
    target_aligned = target_ret.loc[first_valid_replica:]
    replica_aligned = replica_ret.loc[first_valid_replica:]
    
    # Concatena le serie allineate
    df = pd.concat([
        target_aligned.rename("Target"),
        replica_aligned.rename("Replica")
    ], axis=1).dropna()
    
    # Normalizzazione corretta (ora partono dallo stesso punto)
    perf = df.add(1).cumprod()

    # AGGIUNGI METRICHE DI DEBUG
    correlation = df['Target'].corr(df['Replica'])
    tracking_error = (df['Target'] - df['Replica']).std() * np.sqrt(52)
    
    st.info(f"📊 Correlazione: {correlation:.3f} | Tracking Error: {tracking_error:.3f}")

    fig = go.Figure()
    for col in perf:
        fig.add_trace(go.Scatter(x=perf.index, y=perf[col], name=col))
    fig.update_layout(title="Cumulative Performance", xaxis_title="Date", yaxis_title="Growth of 1", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # GRAFICI AGGIUNTIVI
    col1, col2 = st.columns(2)
    
    with col1:
        # Grafico pesi nel tempo (top 5 asset)
        top_assets = weights_df.abs().mean().nlargest(5).index
        fig_weights = go.Figure()
        for asset in top_assets:
            fig_weights.add_trace(go.Scatter(x=weights_df.index, y=weights_df[asset], name=asset, mode='lines'))
        fig_weights.update_layout(title="Evoluzione Pesi Top 5 Assets", xaxis_title="Date", yaxis_title="Peso")
        st.plotly_chart(fig_weights, use_container_width=True)
    
    with col2:
        # Distribuzione errori
        errors = df['Target'] - df['Replica']
        fig_errors = go.Figure(data=[go.Histogram(x=errors, nbinsx=50)])
        fig_errors.update_layout(title="Distribuzione Tracking Errors", xaxis_title="Error", yaxis_title="Frequenza")
        st.plotly_chart(fig_errors, use_container_width=True)

    st.subheader("Metriche di Performance")
    metrics = pd.DataFrame(
        [compute_metrics(df[c]) for c in df.columns],
        index=df.columns,
    )
    st.dataframe(metrics.style.format("{:.2%}"))

    # STATISTICHE SUI PESI
    with st.expander("📋 Statistiche Pesi"):
        st.write(f"**Range pesi:** Min: {weights_df.min().min():.3f}, Max: {weights_df.max().max():.3f}")
        st.write(f"**Media somma pesi assoluti:** {weights_df.abs().sum(axis=1).mean():.3f}")
        st.write("**Top 10 asset per peso medio:**")
        top_weights = weights_df.abs().mean().nlargest(10)
        st.dataframe(top_weights.to_frame("Peso Medio Assoluto"))

    out = BytesIO()
    perf.to_csv(out)
    st.download_button("Download CSV serie storiche", out.getvalue(), "target_replica_series.csv", "text/csv")
    # --------------- DEBUG QUICK CHECK ---------------
    with st.expander("🔍 Debug dati (Target vs Replica)"):
        st.write("**Target – primi 5 valori**")
        st.write(target_ret.head())
        st.write("**Replica – primi 5 valori**")
        st.write(replica_ret.head())

        st.write("**Statistiche Target**")
        st.write(target_ret.describe())
        st.write("**Statistiche Replica**")
        st.write(replica_ret.describe())

        st.write(
            f"**Prima data Target:** {target_ret.index[0]}  \n"
            f"**Prima data Replica:** {replica_ret.index[0]}  \n"
            f"**Lunghezza Target:** {len(target_ret)}  \n"
            f"**Lunghezza Replica:** {len(replica_ret)}"
        )
    # -----------------------------------------------

if __name__ == "__main__":
    main()


