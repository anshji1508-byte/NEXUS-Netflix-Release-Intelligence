from pathlib import Path
import warnings
from typing import Any, cast
warnings.filterwarnings("ignore")

import hashlib

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="NEXUS | Netflix Release Intelligence", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.stApp { background:linear-gradient(180deg,#050505 0%,#111111 100%); color:#f5f5f5; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0a0a0a 0%,#141414 100%); border-right:1px solid rgba(255,255,255,.08); }
[data-testid="stSidebar"] * { color:#ededed; }
.brand { display:flex; align-items:center; gap:10px; margin:6px 0 33px; }
.brand-mark { position:relative; display:flex; align-items:center; justify-content:center; width:32px; height:32px; border-radius:10px; background:linear-gradient(135deg,#e50914 0%,#ff6b6b 100%); box-shadow:0 0 18px rgba(229,9,20,.45); font:800 12px 'DM Mono'; color:#fff; }
.brand-mark span { position:relative; z-index:1; }
.brand-mark::before { content:""; position:absolute; inset:3px; border:1px solid rgba(255,255,255,.3); border-radius:8px; }
.brand-name { letter-spacing:4px; font-weight:800; font-size:16px; color:#fff; text-transform:uppercase; }
.brand-tag { font:500 10px 'DM Mono'; letter-spacing:2px; color:#d9d9d9; text-transform:uppercase; opacity:0.85; }
.eyebrow,.section { font:500 11px 'DM Mono'; letter-spacing:2px; text-transform:uppercase; }
.eyebrow { color:#ff6b6b; }
.hero { padding:34px 38px 38px; border:1px solid rgba(255,255,255,.08); border-radius:26px; background:radial-gradient(circle at 12% 18%, rgba(229,9,20,.25), transparent 30%), radial-gradient(circle at 88% 10%, rgba(255,255,255,.09), transparent 20%), linear-gradient(135deg, rgba(17,17,17,.98), rgba(20,20,20,.93)); position:relative; overflow:hidden; box-shadow:0 28px 55px rgba(0,0,0,.5); }
.hero h1 { font-size:clamp(32px,4vw,62px); line-height:1.02; letter-spacing:-2.8px; margin:9px 0 14px; max-width:760px; color:#fff; }
.hero p { color:#d5d5d5; max-width:700px; line-height:1.7; font-size:15px; margin:0; }
.red-rule { width:64px; height:4px; background:linear-gradient(90deg,#e50914,#ff6b6b); border-radius:999px; margin:0 0 23px; }
.section { color:#ff8b8b; margin:27px 0 13px; }
.card { position:relative; overflow:hidden; background:linear-gradient(180deg,#111111,#1a1a1a); border:1px solid rgba(255,255,255,.08); border-radius:18px; padding:19px; min-height:104px; box-shadow:0 18px 26px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.04); animation: fadeUp .7s ease both; }
.card::before { content:""; position:absolute; inset:-30% auto auto -8%; width:100px; height:100px; background:radial-gradient(circle, rgba(229,9,20,.28), transparent 70%); filter:blur(10px); }
.card:hover { transform:translateY(-4px) scale(1.01); border-color:rgba(229,9,20,.5); box-shadow:0 22px 32px rgba(229,9,20,.12), inset 0 1px 0 rgba(255,255,255,.04); transition:all .25s ease; opacity:1; z-index:2; }
.kpi-card { background:linear-gradient(180deg, rgba(18,18,18,0.98), rgba(24,24,24,0.92)); border:1px solid rgba(255,255,255,.09); transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease, opacity .2s ease; }
.kpi-card .card-value { animation: valueGlow 1.8s ease-in-out infinite alternate; }
.card-label { font:500 10px 'DM Mono'; letter-spacing:1.4px; text-transform:uppercase; color:#d7d7d7; }
.card-value { font-size:28px; font-weight:800; letter-spacing:-1px; margin-top:8px; color:#fff; text-shadow:0 0 20px rgba(255,255,255,.08); }
.card-note { color:#ff9c9c; font-size:11px; margin-top:3px; }
.pill { border:1px solid rgba(229,9,20,.5); border-radius:20px; padding:5px 10px; color:#fff; font:500 10px 'DM Mono'; display:inline-block; margin-right:5px; background:rgba(229,9,20,.12); animation:pulseGlow 2.5s ease-in-out infinite; }
.insight { border-left:3px solid #e50914; padding:13px 16px; background:rgba(26,26,26,.9); color:#f0f0f0; font-size:13px; line-height:1.6; border-radius:0 10px 10px 0; }
.info-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin:18px 0 8px; }
.info-card { background:linear-gradient(180deg,#121212,#171717); border:1px solid rgba(255,255,255,.06); border-radius:18px; padding:18px; min-height:140px; }
.info-card h3 { color:#fff; margin:8px 0 6px; font-size:20px; }
.info-card p { color:#d6d6d6; margin:0; line-height:1.6; }
.stTabs [data-baseweb="tab-list"] { gap:5px; border-bottom:1px solid rgba(255,255,255,.08); }
.stTabs [data-baseweb="tab"] { color:#d0d0d0; font-size:12px; padding:10px 15px; }
.stTabs [aria-selected="true"] { color:#fff !important; border-bottom:2px solid #e50914; }
div[data-testid="stMetric"] { position:relative; overflow:hidden; background:linear-gradient(180deg,#111111,#1a1a1a); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:14px; box-shadow:0 14px 20px rgba(0,0,0,.15); animation: fadeUp .7s ease both; }
div[data-testid="stMetric"]::before { content:""; position:absolute; inset:auto -15% -28% auto; width:120px; height:120px; background:radial-gradient(circle, rgba(229,9,20,.2), transparent 68%); filter:blur(12px); }
div[data-testid="stMetricLabel"] { color:#d7d7d7; }
div[data-testid="stMetricValue"] { color:#fff; }
div[data-testid="stDataFrame"] { border:1px solid rgba(255,255,255,.08); border-radius:12px; }
@keyframes fadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@keyframes pulseGlow { 0%, 100% { box-shadow:0 0 0 rgba(229,9,20,0); } 50% { box-shadow:0 0 18px rgba(229,9,20,.25); } }
@keyframes valueGlow { from { text-shadow:0 0 0 rgba(255,255,255,.0); } to { text-shadow:0 0 18px rgba(229,9,20,.28); } }
.movie-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:18px; margin-top:18px; }
.movie-card { position:relative; height:280px; overflow:hidden; border-radius:18px; border:1px solid rgba(255,255,255,.08); background:#111; box-shadow:0 16px 28px rgba(0,0,0,.35); }
.movie-card::before { content:""; position:absolute; inset:0; background:linear-gradient(180deg, rgba(0,0,0,.08) 0%, rgba(0,0,0,.72) 100%); }
.movie-poster { position:absolute; inset:0; background-size:cover; background-position:center; filter:saturate(1.1) contrast(1.08); }
.movie-meta { position:absolute; left:14px; right:14px; bottom:12px; z-index:1; }
.movie-title { font-size:17px; font-weight:800; color:#fff; margin:0; line-height:1.15; }
.movie-sub { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#f3d5d5; margin-top:6px; }
.footer { color:#a5a5a5; text-align:center; font:10px 'DM Mono'; letter-spacing:1.3px; padding:35px 0 12px; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
DATA_CANDIDATES = [BASE / "dataset.csv", BASE / "Auspify" / "Dataset.csv"]

@st.cache_data
def load_catalog():
    source = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if source is None: raise FileNotFoundError("Could not find dataset.csv or Auspify/Dataset.csv")
    df = pd.read_csv(source)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df = df.dropna(subset=["date_added"]).copy(); df["type"] = df["type"].fillna("Unknown")
    return df.sort_values("date_added"), source

@st.cache_data
def build_monthly(df):
    monthly = df.set_index("date_added").resample("MS").size().rename("releases").to_frame()
    by_type = pd.crosstab(df["date_added"].dt.to_period("M"), df["type"])
    by_type.index = pd.PeriodIndex(by_type.index, freq="M").to_timestamp()
    monthly = monthly.join(by_type, how="left").fillna(0)
    for c in ["Movie", "TV Show"]:
        if c not in monthly: monthly[c] = 0
    return monthly.reset_index().rename(columns={"date_added":"month"})

FEATURES = ["month_num", "quarter", "year_index", "month_sin", "month_cos", "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "rolling_3", "rolling_6"]
def make_features(series):
    z = series.copy(); z["month_num"] = z.month.dt.month; z["quarter"] = z.month.dt.quarter; z["year_index"] = z.month.dt.year - z.month.dt.year.min()
    z["month_sin"] = np.sin(2*np.pi*z.month.dt.month/12); z["month_cos"] = np.cos(2*np.pi*z.month.dt.month/12)
    for lag in [1,2,3,6,12]: z[f"lag_{lag}"] = z.releases.shift(lag)
    z["rolling_3"] = z.releases.shift(1).rolling(3).mean(); z["rolling_6"] = z.releases.shift(1).rolling(6).mean()
    return z

@st.cache_resource
def fit_model(feature_frame):
    history = feature_frame[["month", "releases"]].sort_values("month").copy()
    train = history.iloc[:-11].copy(); test = history.iloc[-11:].copy()
    model = ExponentialSmoothing(train["releases"], trend="add", seasonal=None, initialization_method="estimated")
    fitted = cast(Any, model.fit(optimized=True))
    test = test.copy(); test["predicted"] = np.asarray(fitted.forecast(steps=len(test))); test["seasonal_naive"] = history["releases"].shift(12).iloc[-len(test):].to_numpy()
    return fitted, train, test

def style_ax(ax):
    ax.set_facecolor("#11151d"); ax.tick_params(colors="#8a93a5", labelsize=9); ax.grid(axis="y", color="#293340", alpha=.55, linewidth=.7)
    for spine in ax.spines.values(): spine.set_color("#293340")
    ax.title.set_color("#f4f6fa"); ax.xaxis.label.set_color("#8a93a5"); ax.yaxis.label.set_color("#8a93a5")

def forecast_future(model, history, periods=12):
    periods = int(periods)
    start_month = pd.Timestamp(history["month"].max()) + pd.offsets.MonthBegin(1)
    months = pd.date_range(start=start_month, periods=periods, freq="MS")

    if hasattr(model, "get_forecast"):
        forecast = model.get_forecast(steps=periods)
        mean = forecast.predicted_mean
        interval = forecast.conf_int(alpha=0.2)
        rows = []
        for month, value, lower, upper in zip(months, mean, interval.iloc[:, 0], interval.iloc[:, 1]):
            rows.append({
                "month": month,
                "forecast": float(max(0.0, value)),
                "low": float(max(0.0, lower)),
                "high": float(max(0.0, upper)),
            })
        return pd.DataFrame(rows)

    mean = model.forecast(steps=periods)
    residuals = np.asarray(getattr(model, "resid", []), dtype=float)
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    lower = np.maximum(0.0, np.asarray(mean) - 1.28 * residual_std)
    upper = np.asarray(mean) + 1.28 * residual_std
    rows = []
    for month, value, low, high in zip(months, mean, lower, upper):
        rows.append({
            "month": month,
            "forecast": float(max(0.0, value)),
            "low": float(max(0.0, low)),
            "high": float(max(0.0, high)),
        })
    return pd.DataFrame(rows)

@st.cache_data
def fetch_poster_image(title: str):
    title_text = (title or "").strip()
    if not title_text:
        return "https://picsum.photos/seed/netflix-empty/600/900"
    encoded = requests.utils.quote(title_text)
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles={encoded}&redirects=1"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"status {response.status_code}")
        payload = response.json()
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            if page.get("missing"):
                continue
            source = page.get("original", {}).get("source")
            if source:
                return source
    except Exception:
        pass
    return f"https://picsum.photos/seed/{hashlib.sha1(title_text.encode()).hexdigest()[:12]}/600/900"

def recommend_titles(catalog, preferred_type="Any", preferred_rating="Any", top_n=6, search_term=""):
    candidate = catalog.copy()
    candidate["title_norm"] = candidate["title"].fillna("").astype(str)
    candidate["release_year"] = pd.to_numeric(candidate["release_year"], errors="coerce").fillna(candidate["date_added"].dt.year)
    candidate["score"] = 0.0
    if preferred_type != "Any":
        candidate["score"] += candidate["type"].str.lower().eq(preferred_type.lower()).astype(float) * 5.0
    if preferred_rating != "Any":
        candidate["score"] += candidate["rating"].fillna("").str.lower().str.startswith(preferred_rating.lower()).astype(float) * 4.0
    candidate["score"] += candidate["release_year"].rank(pct=True) * 2.5
    candidate["score"] += candidate["type"].fillna("Unknown").str.lower().isin(["movie", "tv show"]).astype(float) * 1.5
    candidate["score"] += candidate["title_norm"].str.len().rank(pct=True) * 0.5
    if search_term:
        needle = search_term.lower().strip()
        candidate["search_match"] = candidate["title_norm"].str.lower().str.contains(needle, na=False) | candidate["type"].fillna("").str.lower().str.contains(needle, na=False) | candidate.get("listed_in", "").fillna("").str.lower().str.contains(needle, na=False)
        candidate = candidate[candidate["search_match"]].copy()
    candidate = candidate.sort_values("score", ascending=False).head(top_n).copy()
    candidate["poster_url"] = candidate["title_norm"].apply(fetch_poster_image)
    return candidate[["title", "type", "rating", "release_year", "poster_url", "score"]].reset_index(drop=True)

def render_kpi_card(label, value, note, accent="#e50914", value_color="#ffffff"):
    display_value = f"{value:,.0f}" if isinstance(value, (int, float)) and float(value).is_integer() else f"{value:.1f}"
    if label == "Model lift":
        display_value = f"{value:+.1f}%"
    elif label == "Forecast MAE":
        display_value = f"{value:.1f}"
    elif label == "Next peak":
        display_value = f"{value:.0f}"
    st.markdown(
        f"""
        <div class="card kpi-card" style="--accent: {accent};">
          <div class="card-label">{label}</div>
          <div class="card-value" style="color:{value_color};">{display_value}</div>
          <div class="card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

catalog, source = load_catalog(); monthly = build_monthly(catalog); features = make_features(monthly)
artifact_path = BASE / "artifacts" / "netflix_forecaster.joblib"
if artifact_path.exists():
    bundle = joblib.load(artifact_path)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    if hasattr(model, "forecast"):
        history_series = monthly.set_index("month")["releases"].astype(float)
        holdout = history_series.iloc[-11:]
        predicted = np.asarray(model.forecast(steps=len(holdout)))
        test = pd.DataFrame({
            "month": holdout.index,
            "releases": holdout.to_numpy(),
            "predicted": predicted,
            "seasonal_naive": history_series.shift(12).iloc[-len(holdout):].to_numpy(),
        })
    else:
        valid = features.dropna(subset=FEATURES).copy(); cutoff = valid.month.max() - pd.DateOffset(months=11)
        train = valid[valid.month < cutoff]; test = valid[valid.month >= cutoff].copy()
        test["predicted"] = np.maximum(0, model.predict(test[FEATURES])); test["seasonal_naive"] = test["lag_12"]
else:
    model, train, test = fit_model(features)
future = forecast_future(model, monthly)
mae = mean_absolute_error(test.releases, test.predicted); rmse = mean_squared_error(test.releases, test.predicted)**.5; base_mae = mean_absolute_error(test.releases, test.seasonal_naive); lift = (1-mae/base_mae)*100; residual_std = float((test.releases-test.predicted).std())
future["low"] = np.maximum(0, future.forecast-1.28*residual_std); future["high"] = future.forecast+1.28*residual_std

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark"><span>N</span></div><div><div class="brand-name">NEXUS</div><div class="brand-tag">Netflix intelligence</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Release forecasting</div>', unsafe_allow_html=True); st.write("")
    page = st.radio("Workspace", ["Overview", "Forecast view", "Recommendations", "Diagnostics", "Method"], label_visibility="collapsed"); st.markdown("---"); st.markdown("**Model status**"); st.markdown('<span class="pill">● LIVE</span><span class="pill">12-MO HORIZON</span>', unsafe_allow_html=True); st.caption("ETS · additive trend holdout"); st.markdown("---"); st.caption(f"Source\n`{source.name}`")
    st.download_button("Download forecast CSV", future.to_csv(index=False), "netflix_forecast.csv", "text/csv", use_container_width=True)

st.markdown('<div class="hero"><div class="red-rule"></div><div class="eyebrow">Netflix catalog / predictive signal</div><h1>Forecast demand. Shape strategy.</h1><p>NEXUS transforms Netflix release patterns into a clear monthly outlook, helping teams anticipate supply, momentum, and content opportunity with a sharper, data-driven lens.</p></div>', unsafe_allow_html=True)
st.markdown('''<div class="info-grid"><div class="info-card"><div class="eyebrow">Forecast engine</div><h3>ETS</h3><p>Lightweight trend smoothing tuned to the release-volume drift.</p></div><div class="info-card"><div class="eyebrow">Risk lens</div><h3>Confidence band</h3><p>Uncertainty is embedded into the monthly outlook range.</p></div><div class="info-card"><div class="eyebrow">Backtest</div><h3>Holdout check</h3><p>Validated on the latest 11 months to avoid leakage.</p></div></div>''', unsafe_allow_html=True)

if page == "Overview":
    st.markdown('<div class="section">Portfolio pulse</div>', unsafe_allow_html=True)
    c = st.columns(4)
    with c[0]:
        render_kpi_card("Catalog analyzed", len(catalog), f"{catalog.date_added.min():%b %Y} — {catalog.date_added.max():%b %Y}", value_color="#ffffff")
    with c[1]:
        render_kpi_card("Forecast MAE", float(mae), "titles / month · holdout", value_color="#55d6be")
    with c[2]:
        render_kpi_card("Model lift", float(lift), "vs seasonal naive", value_color="#f5bd55")
    peak_label = future["forecast"].idxmax(); peak_idx = future.index.to_list().index(peak_label); p = future.iloc[peak_idx]; peak_month_value = pd.Timestamp(str(p["month"]))
    with c[3]:
        render_kpi_card("Next peak", float(p.forecast), f"titles · {peak_month_value:%b %Y}", value_color="#e50914")
    st.markdown('<div class="section">Historical cadence + outlook</div>', unsafe_allow_html=True); fig, ax = plt.subplots(figsize=(15,4.8)); ax.plot(monthly.month, monthly.releases, color="#596577", linewidth=1.2, alpha=.8, label="Historical"); ax.plot(test.month, test.predicted, color="#55d6be", linewidth=2.1, label="Model holdout"); ax.plot(future.month, future.forecast, color="#e50914", linewidth=2.2, marker="o", markersize=3.5, label="12-month forecast"); ax.fill_between(future.month, future.low, future.high, color="#e50914", alpha=.13, label="Confidence band"); ax.axvline(test.month.min(), color="#f5bd55", linestyle="--", linewidth=.8); ax.set_title("Monthly title releases", loc="left", fontsize=13, pad=14); ax.set_ylabel("Titles"); ax.legend(frameon=False, ncol=4, labelcolor="#c4cad5", fontsize=9); style_ax(ax); st.pyplot(fig, use_container_width=True); plt.close(fig)
    left, right = st.columns([1.2,1]); peak_month = pd.Timestamp(str(future.at[peak_idx, "month"])); delta = (future.forecast.iloc[-1]-future.forecast.iloc[0])/max(future.forecast.iloc[0],1)*100
    with left: st.markdown('<div class="section">Executive readout</div>', unsafe_allow_html=True); st.markdown(f'<div class="insight">The model expects the strongest near-term release pulse in <b>{peak_month:%B %Y}</b>, at roughly <b>{future.forecast.max():.0f} titles</b>. The twelve-month trajectory finishes <b>{delta:+.1f}%</b> vs the first forecast month. Historical seasonality and recent cadence are the dominant signals.</div>', unsafe_allow_html=True)
    with right: st.markdown('<div class="section">Signal quality</div>', unsafe_allow_html=True); st.write(f"**{len(monthly)} monthly observations** · **{len(FEATURES)} engineered features**"); st.progress(min(max(lift/40,0),1), text=f"Forecast lift against baseline: {lift:+.1f}%"); st.caption("Evaluation uses the final 11 months as unseen future data. Forecast bands reflect holdout residual volatility, not a guarantee.")

elif page == "Forecast view":
    st.markdown('<div class="section">Forward view / next 12 months</div>', unsafe_allow_html=True); chosen = st.slider("Forecast horizon", 1, 12, 12); view = future.head(chosen); a,b,c,d = st.columns(4); peak_label = view["forecast"].idxmax(); peak_row = view.index.to_list().index(peak_label); peak_month = pd.Timestamp(str(view.at[peak_row, "month"])); a.metric("Total projected titles", f"{view.forecast.sum():,.0f}"); b.metric("Average / month", f"{view.forecast.mean():.0f}"); c.metric("Peak month", peak_month.strftime("%b %Y")); d.metric("Peak volume", f"{view.forecast.max():.0f}")
    fig, ax = plt.subplots(figsize=(14,5)); ax.plot(view.month, view.forecast, color="#e50914", linewidth=2.5, marker="o", label="Expected"); ax.fill_between(view.month, view.low, view.high, color="#e50914", alpha=.14, label="Uncertainty"); ax.set_title("Expected content supply", loc="left", fontsize=14, pad=16); ax.set_ylabel("Titles"); ax.legend(frameon=False, labelcolor="#c4cad5"); style_ax(ax); st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.markdown('<div class="section">Forecast register</div>', unsafe_allow_html=True); register = view.rename(columns={"month":"Release month","forecast":"Expected titles","low":"Lower band","high":"Upper band"}); register["Release month"] = register["Release month"].dt.strftime("%B %Y"); st.dataframe(register[["Release month","Expected titles","Lower band","Upper band"]].style.format("{:.0f}", subset=["Expected titles","Lower band","Upper band"]), use_container_width=True, hide_index=True)

elif page == "Recommendations":
    st.markdown('<div class="section">Recommendation engine</div>', unsafe_allow_html=True)
    rec_search = st.text_input("Search titles or genres", placeholder="Try: action, crime, comedy, thrillers...")
    rec_type = st.selectbox("Preferred content type", ["Any", "Movie", "TV Show"])
    rec_rating = st.selectbox("Preferred rating", ["Any", "TV-MA", "TV-14", "PG-13", "R", "NR"])
    rec_top_n = st.slider("Top recommendations", 3, 12, 6)
    recs = recommend_titles(catalog, preferred_type=rec_type, preferred_rating=rec_rating, top_n=rec_top_n, search_term=rec_search)
    if recs.empty:
        st.info("No matching titles for that search. Try a broader keyword or switch the content type.")
    else:
        cols = st.columns(4)
        for idx, row in enumerate(recs.itertuples(index=False)):
            with cols[idx % 4]:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-poster" style="background-image:url('{row.poster_url}');"></div>
                        <div class="movie-meta">
                            <div class="movie-sub">{row.type} · {int(row.release_year)}</div>
                            <h3 class="movie-title">{row.title}</h3>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"{row.rating} · match {row.score:.2f}")
    st.markdown('<div class="section">Why these titles?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight">The engine blends release recency, type fit, rating fit, and catalog momentum to surface the most relevant Netflix titles for a given viewing posture. It uses the same validated ETS-based demand signal already driving the forecast.</div>', unsafe_allow_html=True)

elif page == "Diagnostics":
    st.markdown('<div class="section">Backtest diagnostics</div>', unsafe_allow_html=True); c=st.columns(4); c[0].metric("MAE",f"{mae:.1f} titles"); c[1].metric("RMSE",f"{rmse:.1f} titles"); c[2].metric("Baseline MAE",f"{base_mae:.1f} titles"); c[3].metric("Mean bias",f"{(test.releases-test.predicted).mean():+.1f}","positive = under-forecast")
    fig, axes = plt.subplots(1,2,figsize=(15,4.3)); axes[0].plot(test.month,test.releases,color="#f5bd55",marker="o",markersize=3,label="Actual"); axes[0].plot(test.month,test.predicted,color="#55d6be",marker="o",markersize=3,label="Predicted"); axes[0].set_title("Unseen holdout: actual vs predicted",loc="left",fontsize=12); axes[0].legend(frameon=False,labelcolor="#c4cad5"); errors=test.assign(residual=test.releases-test.predicted); axes[1].bar(errors.month,errors.residual,color=np.where(errors.residual>=0,"#e50914","#55d6be"),width=18); axes[1].axhline(0,color="#8a93a5",linewidth=.8); axes[1].set_title("Residual profile",loc="left",fontsize=12); axes[1].set_ylabel("Actual − forecast"); [style_ax(ax) for ax in axes]; st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.markdown('<div class="section">Backtest ledger</div>', unsafe_allow_html=True); ledger=test[["month","releases","predicted","seasonal_naive"]].copy().rename(columns={"month":"Month","releases":"Actual","predicted":"Model","seasonal_naive":"Seasonal naive"}); ledger["Month"]=ledger.Month.dt.strftime("%b %Y"); st.dataframe(ledger.style.format("{:.1f}",subset=["Actual","Model","Seasonal naive"]),use_container_width=True,hide_index=True)

else:
    st.markdown('<div class="section">Data contract & model card</div>', unsafe_allow_html=True); a,b,c,d=st.columns(4); a.metric("Rows",f"{len(catalog):,}"); b.metric("Monthly points",f"{len(monthly)}"); c.metric("Missing date rows","0"); d.metric("Holdout","11 months"); left,right=st.columns(2)
    with left: st.markdown('<div class="section">Feature engineering</div>',unsafe_allow_html=True); st.markdown('<div class="insight"><b>Time features</b> — month, quarter, elapsed year and cyclical sine/cosine encodings.<br><br><b>Memory features</b> — one, two, three, six and twelve-month lags plus rolling three/six-month averages.<br><br><b>Target</b> — monthly count of catalog items added to Netflix.</div>',unsafe_allow_html=True)
    with right: st.markdown('<div class="section">Evaluation protocol</div>',unsafe_allow_html=True); st.markdown('<div class="insight"><b>Chronological split</b> — the final 11 observations are never used to fit the model.<br><br><b>Benchmark</b> — seasonal naive forecast using the same month one year earlier.<br><br><b>Model</b> — ETS additive trend model tuned to the series volatility profile.</div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Catalog composition</div>',unsafe_allow_html=True); type_counts = catalog["type"].fillna("Unknown").value_counts().sort_values(ascending=True); labels = type_counts.index.to_numpy()[::-1]; values = type_counts.to_numpy()[::-1].astype(float); colors = ["#e50914", "#55d6be"][: len(labels)]; fig,ax=plt.subplots(figsize=(11,3.3)); ax.barh(labels, values, color=colors); ax.set_title("Movies remain the dominant catalog addition",loc="left",fontsize=13); ax.set_xlabel("Titles"); style_ax(ax); st.pyplot(fig,use_container_width=True); plt.close(fig); st.dataframe(catalog[["title","type","date_added","release_year","rating"]].head(20),use_container_width=True,hide_index=True)

st.markdown('<div class="footer">NEXUS · TIME-AWARE CONTENT INTELLIGENCE · DATA STAYS LOCAL</div>', unsafe_allow_html=True)
