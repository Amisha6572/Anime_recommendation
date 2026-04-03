# ============================================================
# ANIME RECOMMENDATION SYSTEM — RESUME READY
# ============================================================
import pandas as pd
import numpy as np
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import skew
from fuzzywuzzy import process
import shap
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG — must be first Streamlit call
# ============================================================
st.set_page_config(
    page_title="AniRec — Anime Recommender",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { background-color:#141414 !important; color:#e5e5e5 !important; font-family:'Inter',sans-serif !important; }
.stApp { background-color:#141414 !important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#1a1a2e,#16213e,#0f3460) !important; border-right:1px solid #e50914 !important; }
section[data-testid="stSidebar"] * { color:#e5e5e5 !important; }
.netflix-title { font-family:'Bebas Neue',cursive; font-size:3.8rem; background:linear-gradient(90deg,#e50914,#ff6b6b,#ffd700); -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-align:center; letter-spacing:4px; margin-bottom:0; }
.netflix-subtitle { text-align:center; color:#b3b3b3; font-size:1rem; margin-top:-8px; margin-bottom:28px; letter-spacing:2px; }
.anime-card { background:linear-gradient(145deg,#1f1f1f,#2a2a2a); border-radius:12px; overflow:hidden; border:1px solid #333; transition:all 0.3s ease; margin-bottom:10px; }
.anime-card:hover { border-color:#e50914; box-shadow:0 8px 32px rgba(229,9,20,0.4); transform:scale(1.03); }
.anime-card img { width:100%; height:260px; object-fit:cover; }
.anime-card-body { padding:10px; }
.anime-card-title { font-weight:700; font-size:0.82rem; color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:3px; }
.anime-card-rating { color:#ffd700; font-weight:700; font-size:0.82rem; }
.anime-card-meta { font-size:0.7rem; color:#b3b3b3; }
.match-badge { background:#46d369; color:#000; font-size:0.68rem; font-weight:700; padding:2px 7px; border-radius:4px; display:inline-block; margin-bottom:3px; }
.section-header { font-size:1.5rem; font-weight:700; color:#fff; border-left:4px solid #e50914; padding-left:12px; margin:28px 0 18px 0; }
.metric-card { background:linear-gradient(135deg,#1a1a2e,#16213e); border:1px solid #e50914; border-radius:12px; padding:18px; text-align:center; }
.metric-value { font-size:2rem; font-weight:700; color:#e50914; }
.metric-label { font-size:0.82rem; color:#b3b3b3; margin-top:4px; }
.stButton>button { background:linear-gradient(90deg,#e50914,#b20710) !important; color:white !important; border:none !important; border-radius:6px !important; font-weight:700 !important; padding:10px 28px !important; width:100% !important; }
.stButton>button:hover { box-shadow:0 4px 20px rgba(229,9,20,0.5) !important; }
.stTabs [data-baseweb="tab-list"] { background-color:#1f1f1f !important; border-radius:8px; padding:4px; }
.stTabs [data-baseweb="tab"] { color:#b3b3b3 !important; font-weight:600 !important; border-radius:6px !important; }
.stTabs [aria-selected="true"] { background-color:#e50914 !important; color:white !important; }
::-webkit-scrollbar { width:6px; } ::-webkit-scrollbar-track { background:#1f1f1f; } ::-webkit-scrollbar-thumb { background:#e50914; border-radius:3px; }
.hero-banner { background:linear-gradient(135deg,#0d0d0d,#1a0000,#0d0d0d); border:1px solid #e50914; border-radius:16px; padding:36px; text-align:center; margin-bottom:28px; }
hr { border-color:#333 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    anime = pd.read_csv("anime.csv")
    try:
        # Cap at 500k rows to avoid OOM on Streamlit Cloud
        rating = pd.read_csv(
            "rating.csv",
            nrows=500_000,
            dtype={"user_id": "int32", "anime_id": "int32", "rating": "int8"}
        )
    except Exception:
        rating = None
    return anime, rating

# ============================================================
# CLEANING & FEATURE ENGINEERING
# ============================================================
@st.cache_data(show_spinner=False)
def clean_and_engineer(anime_df):
    df = anime_df.copy()
    df["genre"]    = df["genre"].fillna("Unknown")
    df["type"]     = df["type"].fillna("Unknown")
    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    df["episodes"] = df["episodes"].fillna(df["episodes"].median())
    df["rating"]   = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"]   = df["rating"].fillna(df["rating"].median())
    df["members"]  = df["members"].replace(0, 1)

    members_skew  = float(skew(df["members"]))
    episodes_skew = float(skew(df["episodes"]))
    df["members_log"]  = np.log1p(df["members"])
    df["episodes_log"] = np.log1p(df["episodes"])

    sc = MinMaxScaler()
    df["rating_norm"]   = sc.fit_transform(df[["rating"]])
    df["members_norm"]  = sc.fit_transform(df[["members_log"]])
    df["episodes_norm"] = sc.fit_transform(df[["episodes_log"]])

    # TF-IDF feature: double-weight genre
    df["features"] = (
        df["genre"].str.replace(",", " ") + " " +
        df["type"] + " " +
        df["genre"].str.replace(",", " ")
    )
    return df, members_skew, episodes_skew

# ============================================================
# CONTENT MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def build_content_model(features_series):
    tfidf        = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(features_series)
    sim_matrix   = cosine_similarity(tfidf_matrix)
    return tfidf, tfidf_matrix, sim_matrix

# ============================================================
# RICH CLUSTER FEATURES
# ============================================================
@st.cache_data(show_spinner=False)
def build_cluster_features(_df, _tfidf_matrix):
    genre_series = _df["genre"].str.split(",").apply(lambda x: [g.strip() for g in x])
    top_genres   = (
        _df["genre"].str.split(",").explode().str.strip()
        .value_counts().head(30).index.tolist()
    )
    genre_matrix = np.zeros((len(_df), len(top_genres)), dtype=np.float32)
    for i, genres in enumerate(genre_series):
        for g in genres:
            if g in top_genres:
                genre_matrix[i, top_genres.index(g)] = 1.0

    type_dummies = pd.get_dummies(_df["type"], prefix="type").values.astype(np.float32)
    numeric      = _df[["rating_norm", "members_norm", "episodes_norm"]].values.astype(np.float32)
    X_rich       = np.hstack([genre_matrix, type_dummies, numeric])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_rich)

    pca_full = PCA(n_components=min(50, X_scaled.shape[1]), random_state=42)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = max(int(np.argmax(cumvar >= 0.90)) + 1, 5)

    X_pca = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)
    X_2d  = PCA(n_components=2,      random_state=42).fit_transform(X_scaled)
    return X_pca, X_2d, top_genres, n_comp, cumvar

@st.cache_data(show_spinner=False)
def find_optimal_k(_X_pca):
    sil_scores, inertias, ks = [], [], list(range(2, 13))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=15, max_iter=500)
        labels = km.fit_predict(_X_pca)
        sil_scores.append(silhouette_score(_X_pca, labels))
        inertias.append(km.inertia_)
    best_k = ks[int(np.argmax(sil_scores))]
    return ks, sil_scores, inertias, best_k

# ============================================================
# POSTER (Jikan API)
# ============================================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster(name):
    try:
        url = f"https://api.jikan.moe/v4/anime?q={requests.utils.quote(name)}&limit=1"
        res = requests.get(url, timeout=5).json()
        return res["data"][0]["images"]["jpg"]["large_image_url"]
    except Exception:
        return "https://placehold.co/300x420/141414/e50914?text=No+Poster"

# ============================================================
# FUZZY SEARCH
# ============================================================
def fuzzy_search(query, names, limit=10):
    return [r[0] for r in process.extract(query, names, limit=limit)]

# ============================================================
# RECOMMENDATIONS
# ============================================================
def get_recommendations(anime_name, df, sim_matrix, n=10):
    idx_series = pd.Series(df.index, index=df["name"]).drop_duplicates()
    if anime_name not in idx_series:
        return pd.DataFrame()
    idx        = idx_series[anime_name]
    scores     = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)
    scores     = [s for s in scores if s[0] != idx][:n]
    result     = df.iloc[[s[0] for s in scores]].copy()
    result["similarity"] = [s[1] for s in scores]
    result["match_pct"]  = (result["similarity"] * 100).round(1)
    return result

# ============================================================
# ANIME CARD HTML
# ============================================================
def anime_card_html(name, poster_url, rating_val, anime_type, genre, match_pct=None):
    badge       = f'<div class="match-badge">▶ {match_pct}% Match</div>' if match_pct else ""
    genre_short = ", ".join(genre.split(",")[:2]) if genre else ""
    return f"""
    <div class="anime-card">
        <img src="{poster_url}" alt="{name}"
             onerror="this.src='https://placehold.co/300x420/141414/e50914?text=No+Poster'"/>
        <div class="anime-card-body">
            {badge}
            <div class="anime-card-title" title="{name}">{name}</div>
            <div class="anime-card-rating">⭐ {rating_val}</div>
            <div class="anime-card-meta">{anime_type} · {genre_short}</div>
        </div>
    </div>"""

# ============================================================
# MAIN
# ============================================================
def main():
    with st.spinner("🎌 Loading anime universe..."):
        anime_raw, rating_df = load_data()
        use_ratings = rating_df is not None
        df, members_skew, episodes_skew = clean_and_engineer(anime_raw)
        tfidf, tfidf_matrix, sim_matrix  = build_content_model(df["features"])
        feature_names = tfidf.get_feature_names_out()

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:20px 0'>
            <div style='font-size:2.8rem'>🎌</div>
            <div style='font-size:1.3rem;font-weight:700;color:#e50914;letter-spacing:2px'>AniRec</div>
            <div style='font-size:0.72rem;color:#b3b3b3'>AI Anime Recommender</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio("Navigation", [
            "🏠 Home & Recommend",
            "📊 EDA Dashboard",
            "🔬 PCA Explorer",
            "🤖 Clustering",
            "🧠 SHAP Explainability",
            "📈 Top Charts",
            "👤 User Ratings"
        ], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("<div style='font-size:0.72rem;color:#555;text-align:center'>Streamlit · scikit-learn · Plotly · SHAP<br>Data: MyAnimeList · Posters: Jikan API</div>",
                    unsafe_allow_html=True)

    # ── HOME & RECOMMEND ─────────────────────────────────────
    if page == "🏠 Home & Recommend":
        st.markdown('<div class="hero-banner"><div class="netflix-title">ANIREC</div><div class="netflix-subtitle">YOUR PERSONAL ANIME UNIVERSE · AI-POWERED RECOMMENDATIONS</div></div>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in zip(
            [c1, c2, c3, c4],
            [f"{len(df):,}",
             str(df["genre"].str.split(",").explode().str.strip().nunique()),
             str(df["type"].nunique()),
             f"{df['rating'].max():.2f}"],
            ["Total Anime", "Unique Genres", "Media Types", "Top Rating"]
        ):
            col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>',
                         unsafe_allow_html=True)

        st.markdown('<div class="section-header">🔍 Find Your Next Anime</div>', unsafe_allow_html=True)

        col_s, col_t, col_n = st.columns([3, 1, 1])
        with col_s:
            query = st.text_input("Search anime", placeholder="e.g. Naruto, Attack on Titan",
                                  label_visibility="collapsed")
        with col_t:
            type_filter = st.selectbox("Filter by type", ["All"] + sorted(df["type"].unique().tolist()),
                                       label_visibility="collapsed")
        with col_n:
            n_recs = st.selectbox("Number of results", [5, 10, 15, 20],
                                  label_visibility="collapsed")

        if query:
            suggestions   = fuzzy_search(query, df["name"].tolist(), limit=10)
            selected_anime = st.selectbox("Did you mean:", suggestions)
        else:
            selected_anime = st.selectbox("Or pick a popular anime:",
                                          df.nlargest(50, "rating")["name"].tolist())

        all_genres   = sorted({g.strip() for genres in df["genre"].dropna() for g in genres.split(",")})
        genre_filter = st.multiselect("Filter by genre (optional)", all_genres)

        if st.button("🎬 Get Recommendations"):
            with st.spinner("Finding similar anime..."):
                recs = get_recommendations(selected_anime, df, sim_matrix, n=n_recs * 3)
                if type_filter != "All":
                    recs = recs[recs["type"] == type_filter]
                if genre_filter:
                    recs = recs[recs["genre"].apply(lambda g: any(gf in g for gf in genre_filter))]
                recs = recs.head(n_recs)

            if recs.empty:
                st.warning("No results. Try removing filters.")
            else:
                sel = df[df["name"] == selected_anime].iloc[0]
                st.markdown(f'<div class="section-header">📌 Selected: {selected_anime}</div>',
                            unsafe_allow_html=True)
                ic, pc = st.columns([3, 1])
                with ic:
                    st.markdown(f"""
                    <div style='background:#1f1f1f;border-radius:12px;padding:18px;border-left:4px solid #e50914'>
                        <div style='font-size:1.3rem;font-weight:700;color:#fff;margin-bottom:8px'>{selected_anime}</div>
                        <div style='color:#b3b3b3;margin-bottom:5px'>⭐ Rating: <span style='color:#ffd700;font-weight:700'>{sel['rating']:.2f}</span></div>
                        <div style='color:#b3b3b3;margin-bottom:5px'>📺 Type: <span style='color:#fff'>{sel['type']}</span></div>
                        <div style='color:#b3b3b3;margin-bottom:5px'>🎭 Genre: <span style='color:#fff'>{sel['genre']}</span></div>
                        <div style='color:#b3b3b3'>👥 Members: <span style='color:#fff'>{int(sel['members']):,}</span></div>
                    </div>""", unsafe_allow_html=True)
                with pc:
                    st.image(fetch_poster(selected_anime), width=300)

                st.markdown('<div class="section-header">🎯 Recommended For You</div>', unsafe_allow_html=True)
                rec_list = recs.reset_index(drop=True)
                for row_start in range(0, len(rec_list), 5):
                    chunk = rec_list.iloc[row_start:row_start + 5]
                    cols  = st.columns(5)
                    for col, (_, rec) in zip(cols, chunk.iterrows()):
                        with col:
                            st.markdown(anime_card_html(
                                rec["name"], fetch_poster(rec["name"]),
                                f"{rec['rating']:.2f}", rec["type"],
                                rec["genre"], rec["match_pct"]
                            ), unsafe_allow_html=True)

                fig_sim = px.bar(rec_list, x="similarity", y="name", orientation="h",
                                 color="similarity", color_continuous_scale="Reds",
                                 template="plotly_dark", title="Content Similarity Scores",
                                 labels={"similarity": "Cosine Similarity", "name": ""})
                fig_sim.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                      font_color="#e5e5e5", height=400,
                                      yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_sim, use_container_width=True)

        st.markdown('<div class="section-header">🔥 Top Rated Anime</div>', unsafe_allow_html=True)
        top5 = df.nlargest(5, "rating").reset_index(drop=True)
        cols = st.columns(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            with cols[i]:
                st.markdown(anime_card_html(row["name"], fetch_poster(row["name"]),
                                            f"{row['rating']:.2f}", row["type"], row["genre"]),
                            unsafe_allow_html=True)

    # ── EDA DASHBOARD ────────────────────────────────────────
    elif page == "📊 EDA Dashboard":
        st.markdown('<div class="netflix-title" style="font-size:2.4rem">EDA DASHBOARD</div>', unsafe_allow_html=True)
        st.markdown('<div class="netflix-subtitle">Exploratory Data Analysis · Distributions · Transformations</div>', unsafe_allow_html=True)

        o1, o2, o3, o4 = st.columns(4)
        for col, val, lbl in zip([o1, o2, o3, o4],
                                  [df.shape[0], df.shape[1],
                                   int(df.isnull().sum().sum()), int(df.duplicated().sum())],
                                  ["Records", "Features", "Missing (post-clean)", "Duplicates"]):
            col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>',
                         unsafe_allow_html=True)

        st.markdown('<div class="section-header">📊 Skewness & Transformations</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#1f1f1f;border-radius:10px;padding:14px;border:1px solid #333;margin-bottom:18px'>
            <span style='color:#b3b3b3;font-size:0.88rem'>
            📌 <b style='color:#e50914'>members</b> skewness = <b style='color:#ffd700'>{members_skew:.2f}</b>
            {"→ right-skewed → <b style='color:#46d369'>log1p applied</b>" if abs(members_skew)>1 else "→ acceptable"}<br>
            📌 <b style='color:#e50914'>episodes</b> skewness = <b style='color:#ffd700'>{episodes_skew:.2f}</b>
            {"→ right-skewed → <b style='color:#46d369'>log1p applied</b>" if abs(episodes_skew)>1 else "→ acceptable"}
            </span>
        </div>""", unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs(["Rating", "Members", "Episodes", "Type & Genre"])

        with t1:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Rating Distribution", "Avg Rating by Type"])
            fig.add_trace(go.Histogram(x=df["rating"], nbinsx=30, marker_color="#e50914"), row=1, col=1)
            tr = df.groupby("type")["rating"].mean().reset_index()
            fig.add_trace(go.Bar(x=tr["type"], y=tr["rating"], marker_color="#e50914"), row=1, col=2)
            fig.update_layout(template="plotly_dark", paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                              font_color="#e5e5e5", height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Members (Raw — skewed)", "Members (log1p — normalised)"])
            fig.add_trace(go.Histogram(x=df["members"],     nbinsx=40, marker_color="#ff6b6b"), row=1, col=1)
            fig.add_trace(go.Histogram(x=df["members_log"], nbinsx=40, marker_color="#46d369"), row=1, col=2)
            fig.update_layout(template="plotly_dark", paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                              font_color="#e5e5e5", height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Raw skewness: {members_skew:.2f}  →  After log1p: {skew(df['members_log']):.2f}")

        with t3:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Episodes (Raw — skewed)", "Episodes (log1p — normalised)"])
            fig.add_trace(go.Histogram(x=df["episodes"],     nbinsx=40, marker_color="#ff6b6b"), row=1, col=1)
            fig.add_trace(go.Histogram(x=df["episodes_log"], nbinsx=40, marker_color="#46d369"), row=1, col=2)
            fig.update_layout(template="plotly_dark", paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                              font_color="#e5e5e5", height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with t4:
            ca, cb = st.columns(2)
            with ca:
                tc = df["type"].value_counts().reset_index()
                tc.columns = ["type", "count"]
                fig_t = px.pie(tc, values="count", names="type", hole=0.4,
                               template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Bold,
                               title="Anime by Type")
                fig_t.update_layout(paper_bgcolor="#141414", font_color="#e5e5e5")
                st.plotly_chart(fig_t, use_container_width=True)
            with cb:
                gs = df["genre"].str.split(",").explode().str.strip().value_counts().head(15).reset_index()
                gs.columns = ["genre", "count"]
                fig_g = px.bar(gs, x="count", y="genre", orientation="h",
                               color="count", color_continuous_scale="Reds",
                               template="plotly_dark", title="Top 15 Genres")
                fig_g.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                    font_color="#e5e5e5", height=440,
                                    yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_g, use_container_width=True)

        st.markdown('<div class="section-header">🔗 Correlation Matrix</div>', unsafe_allow_html=True)
        corr = df[["rating", "members_log", "episodes_log", "rating_norm", "members_norm"]].corr()
        fig_c = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                          template="plotly_dark", title="Feature Correlation Heatmap")
        fig_c.update_layout(paper_bgcolor="#141414", font_color="#e5e5e5", height=380)
        st.plotly_chart(fig_c, use_container_width=True)

        st.markdown('<div class="section-header">📦 Rating Outliers by Type</div>', unsafe_allow_html=True)
        fig_b = px.box(df, x="type", y="rating", color="type", template="plotly_dark",
                       color_discrete_sequence=px.colors.qualitative.Bold)
        fig_b.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                            font_color="#e5e5e5", height=380)
        st.plotly_chart(fig_b, use_container_width=True)

        st.markdown('<div class="section-header">🔵 Rating vs Popularity</div>', unsafe_allow_html=True)
        fig_s = px.scatter(df, x="members_log", y="rating", color="type",
                           hover_data=["name"], size="rating_norm",
                           template="plotly_dark", title="log(Members) vs Rating",
                           color_discrete_sequence=px.colors.qualitative.Bold)
        fig_s.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                            font_color="#e5e5e5", height=430)
        st.plotly_chart(fig_s, use_container_width=True)

    # ── PCA EXPLORER ─────────────────────────────────────────
    elif page == "🔬 PCA Explorer":
        st.markdown('<div class="netflix-title" style="font-size:2.4rem">PCA EXPLORER</div>', unsafe_allow_html=True)
        st.markdown('<div class="netflix-subtitle">Dimensionality Reduction · Explained Variance</div>', unsafe_allow_html=True)

        p1, p2 = st.columns(2)
        with p1:
            pca_dims = st.radio("Dimensions", ["2D", "3D"], horizontal=True)
        with p2:
            color_by = st.selectbox("Colour by", ["type", "rating"])

        if st.button("🔬 Run PCA"):
            n_comp = 2 if pca_dims == "2D" else 3
            with st.spinner("Computing PCA..."):
                X      = tfidf_matrix.toarray()
                Xs     = StandardScaler(with_mean=False).fit_transform(X)
                pca    = PCA(n_components=n_comp, random_state=42)
                Xp     = pca.fit_transform(Xs)
                expl   = pca.explained_variance_ratio_ * 100
                pdf    = df[["name", "type", "rating", "genre"]].copy().reset_index(drop=True)
                pdf["PC1"] = Xp[:, 0]; pdf["PC2"] = Xp[:, 1]
                if n_comp == 3:
                    pdf["PC3"] = Xp[:, 2]

            if n_comp == 2:
                fig_p = px.scatter(pdf, x="PC1", y="PC2", color=color_by,
                                   hover_data=["name", "rating", "genre"],
                                   title=f"PCA 2D — PC1={expl[0]:.1f}%  PC2={expl[1]:.1f}%",
                                   template="plotly_dark",
                                   color_discrete_sequence=px.colors.qualitative.Bold)
            else:
                fig_p = px.scatter_3d(pdf, x="PC1", y="PC2", z="PC3", color=color_by,
                                      hover_data=["name", "rating"],
                                      title=f"PCA 3D — {sum(expl[:3]):.1f}% variance",
                                      template="plotly_dark",
                                      color_discrete_sequence=px.colors.qualitative.Bold)
            fig_p.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                font_color="#e5e5e5", height=540)
            st.plotly_chart(fig_p, use_container_width=True)

            # Explained variance chart
            st.markdown('<div class="section-header">📊 Explained Variance</div>', unsafe_allow_html=True)
            pca20  = PCA(n_components=min(20, Xs.shape[1]), random_state=42).fit(Xs)
            ev     = pca20.explained_variance_ratio_ * 100
            cumev  = np.cumsum(ev)
            fig_ev = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ev.add_trace(go.Bar(x=list(range(1, len(ev)+1)), y=ev,
                                    marker_color="#e50914", name="Individual"))
            fig_ev.add_trace(go.Scatter(x=list(range(1, len(cumev)+1)), y=cumev,
                                        line=dict(color="#ffd700", width=2), name="Cumulative"),
                             secondary_y=True)
            fig_ev.update_layout(template="plotly_dark", paper_bgcolor="#141414",
                                 plot_bgcolor="#1f1f1f", font_color="#e5e5e5",
                                 height=380, showlegend=False,
                                 title="Explained Variance per Component")
            st.plotly_chart(fig_ev, use_container_width=True)
            n95 = int(np.argmax(cumev >= 95)) + 1
            st.markdown(f"""
            <div style='background:#1f1f1f;border-radius:10px;padding:14px;border-left:4px solid #46d369'>
                <span style='color:#e5e5e5;font-size:0.88rem'>
                📌 <b style='color:#46d369'>{n95} components</b> explain <b style='color:#ffd700'>95%</b> of variance.<br>
                📌 First 2 PCs explain <b style='color:#ffd700'>{expl[0]+expl[1]:.1f}%</b>.
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("👆 Click 'Run PCA' to generate the visualisation.")

    # ── CLUSTERING ───────────────────────────────────────────
    elif page == "🤖 Clustering":
        st.markdown('<div class="netflix-title" style="font-size:2.4rem">CLUSTERING</div>', unsafe_allow_html=True)
        st.markdown('<div class="netflix-subtitle">KMeans · DBSCAN · Rich Feature Matrix · Silhouette Optimisation</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#1f1f1f;border-radius:10px;padding:14px;border-left:4px solid #46d369;margin-bottom:18px'>
            <span style='color:#e5e5e5;font-size:0.88rem'>
            <b style='color:#46d369'>Why silhouette is high:</b> We cluster on a
            <b style='color:#ffd700'>rich dense matrix</b> (genre multi-hot + type one-hot + normalised
            rating/members/episodes) reduced via <b style='color:#ffd700'>PCA to 90% variance</b>,
            not raw sparse TF-IDF.
            </span>
        </div>""", unsafe_allow_html=True)

        ck1, ck2 = st.columns(2)
        with ck1:
            algo = st.radio("Algorithm", ["KMeans Auto-k", "KMeans Manual k", "DBSCAN"], horizontal=True)
        with ck2:
            manual_k = st.slider("Manual k", 2, 15, 6)

        if st.button("🤖 Run Clustering"):
            with st.spinner("Building features & clustering..."):
                X_pca, X_2d, top_genres, n_comp, cumvar = build_cluster_features(df, tfidf_matrix)
                pdf = df[["name", "type", "rating", "genre"]].copy().reset_index(drop=True)
                pdf["PC1"] = X_2d[:, 0]; pdf["PC2"] = X_2d[:, 1]

            st.markdown(f"""
            <div style='background:#1f1f1f;border-radius:8px;padding:10px 14px;margin-bottom:14px;border:1px solid #333'>
                <span style='color:#b3b3b3;font-size:0.83rem'>
                📐 <b style='color:#ffd700'>{n_comp} PCA components</b> (90% variance) from
                {len(top_genres)} genre + type + 3 numeric features
                </span>
            </div>""", unsafe_allow_html=True)

            if algo in ["KMeans Auto-k", "KMeans Manual k"]:
                with st.spinner("Sweeping k=2…12 for best silhouette..."):
                    ks, sil_scores, inertias, best_k = find_optimal_k(X_pca)
                chosen_k = best_k if algo == "KMeans Auto-k" else manual_k

                # Silhouette + elbow chart
                st.markdown('<div class="section-header">📈 Silhouette vs k</div>', unsafe_allow_html=True)
                fig_sw = make_subplots(rows=1, cols=2,
                                       subplot_titles=["Silhouette (↑ better)", "Elbow — Inertia (↓ better)"])
                fig_sw.add_trace(go.Scatter(
                    x=ks, y=sil_scores, mode="lines+markers",
                    line=dict(color="#46d369", width=2),
                    marker=dict(color=["#e50914" if k == best_k else "#46d369" for k in ks], size=10)
                ), row=1, col=1)
                fig_sw.add_trace(go.Scatter(
                    x=ks, y=inertias, mode="lines+markers",
                    line=dict(color="#e50914", width=2),
                    marker=dict(color="#ffd700", size=8)
                ), row=1, col=2)
                fig_sw.add_vline(x=best_k, line_dash="dash", line_color="#ffd700",
                                 annotation_text=f"Best k={best_k}", row=1, col=1)
                fig_sw.update_layout(template="plotly_dark", paper_bgcolor="#141414",
                                     plot_bgcolor="#1f1f1f", font_color="#e5e5e5",
                                     height=360, showlegend=False)
                st.plotly_chart(fig_sw, use_container_width=True)

                km     = KMeans(n_clusters=chosen_k, random_state=42, n_init=15, max_iter=500)
                labels = km.fit_predict(X_pca)
                sil    = silhouette_score(X_pca, labels)
                dbs    = davies_bouldin_score(X_pca, labels)
                pdf["Cluster"] = labels.astype(str)

                s1, s2, s3 = st.columns(3)
                s1.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#46d369">{sil:.3f}</div><div class="metric-label">Silhouette (↑ better)</div></div>', unsafe_allow_html=True)
                s2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ffd700">{dbs:.3f}</div><div class="metric-label">Davies-Bouldin (↓ better)</div></div>', unsafe_allow_html=True)
                s3.markdown(f'<div class="metric-card"><div class="metric-value">{chosen_k}</div><div class="metric-label">Clusters</div></div>', unsafe_allow_html=True)
                title = f"KMeans k={chosen_k} · Silhouette={sil:.3f} · DB={dbs:.3f}"

            else:  # DBSCAN
                db     = DBSCAN(eps=0.8, min_samples=3).fit(X_pca)
                labels = db.labels_
                n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
                noise  = list(labels).count(-1)
                pdf["Cluster"] = labels.astype(str)
                sil    = silhouette_score(X_pca, labels) if n_cl > 1 else 0.0
                title  = f"DBSCAN · {n_cl} clusters · {noise} noise · Silhouette={sil:.3f}"

            fig_cl = px.scatter(pdf, x="PC1", y="PC2", color="Cluster",
                                hover_data=["name", "rating", "type", "genre"],
                                title=title, template="plotly_dark",
                                color_discrete_sequence=px.colors.qualitative.Bold)
            fig_cl.update_traces(marker=dict(size=7, opacity=0.8))
            fig_cl.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                 font_color="#e5e5e5", height=560)
            st.plotly_chart(fig_cl, use_container_width=True)

            if algo != "DBSCAN":
                st.markdown('<div class="section-header">🎭 Cluster Composition</div>', unsafe_allow_html=True)
                pdf["Cluster_int"] = labels
                pdf["genre_col"]   = df["genre"].values
                cdata = []
                for c in sorted(pdf["Cluster_int"].unique()):
                    sub = pdf[pdf["Cluster_int"] == c]
                    top_g = (sub["genre_col"].str.split(",").explode().str.strip()
                             .value_counts().head(3).index.tolist())
                    top3  = sub.nlargest(3, "rating")[["name", "rating"]].values
                    names_str = " | ".join([f"{r[0]} ({r[1]:.1f})" for r in top3])
                    genre_str = ", ".join(top_g)
                    cdata.append({"cluster": c, "size": len(sub)})
                    st.markdown(f"""
                    <div style='background:#1f1f1f;border-radius:8px;padding:10px 14px;margin-bottom:7px;border-left:3px solid #e50914'>
                        <span style='color:#e50914;font-weight:700'>Cluster {c}</span>
                        <span style='color:#b3b3b3;font-size:0.8rem;margin-left:8px'>{len(sub)} anime</span>
                        <div style='color:#ffd700;font-size:0.78rem;margin-top:3px'>🎭 {genre_str}</div>
                        <div style='color:#e5e5e5;font-size:0.76rem;margin-top:2px'>⭐ {names_str}</div>
                    </div>""", unsafe_allow_html=True)

                cdf    = pd.DataFrame(cdata)
                fig_cs = px.bar(cdf, x="cluster", y="size", color="size",
                                color_continuous_scale="Reds", template="plotly_dark",
                                title="Anime Count per Cluster")
                fig_cs.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                     font_color="#e5e5e5", height=320)
                st.plotly_chart(fig_cs, use_container_width=True)
        else:
            st.info("👆 Configure and click 'Run Clustering' to start.")

    # ── SHAP ─────────────────────────────────────────────────
    elif page == "🧠 SHAP Explainability":
        st.markdown('<div class="netflix-title" style="font-size:2.4rem">SHAP EXPLAINABILITY</div>', unsafe_allow_html=True)
        st.markdown('<div class="netflix-subtitle">What Drives Recommendations</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#1f1f1f;border-radius:10px;padding:14px;border-left:4px solid #e50914;margin-bottom:18px'>
            <span style='color:#e5e5e5;font-size:0.88rem'>
            SHAP explains which TF-IDF genre/type features contribute most to the recommendation score.
            Higher |SHAP| = stronger influence on similarity.
            </span>
        </div>""", unsafe_allow_html=True)

        n_shap = st.slider("Samples to explain", 20, 80, 40, step=10)

        if st.button("🧠 Compute SHAP Values"):
            with st.spinner("Computing SHAP (may take ~30s)..."):
                X_arr    = tfidf_matrix.toarray()[:n_shap]
                bg       = X_arr[:min(20, n_shap)]
                model_fn = lambda x: np.sum(x, axis=1)
                explainer = shap.KernelExplainer(model_fn, bg)
                shap_vals = explainer.shap_values(X_arr[:min(30, n_shap)], nsamples=50)

                mean_shap  = np.abs(shap_vals).mean(axis=0)
                top_idx    = np.argsort(mean_shap)[::-1][:20]
                top_feats  = [feature_names[i] for i in top_idx]
                top_vals   = mean_shap[top_idx]

            fig_sh = px.bar(x=top_vals, y=top_feats, orientation="h",
                            color=top_vals, color_continuous_scale="Reds",
                            template="plotly_dark", title="Top 20 Features by Mean |SHAP|",
                            labels={"x": "Mean |SHAP|", "y": "Feature"})
            fig_sh.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                 font_color="#e5e5e5", height=540,
                                 yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_sh, use_container_width=True)

            shap_df  = pd.DataFrame(shap_vals[:20, top_idx[:15]], columns=top_feats[:15])
            fig_heat = px.imshow(shap_df.T, color_continuous_scale="RdBu_r",
                                 template="plotly_dark", title="SHAP Heatmap (Top 15 × 20 samples)",
                                 labels={"x": "Sample", "y": "Feature", "color": "SHAP"})
            fig_heat.update_layout(paper_bgcolor="#141414", font_color="#e5e5e5", height=430)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("👆 Click 'Compute SHAP Values' to run.")

    # ── TOP CHARTS ───────────────────────────────────────────
    elif page == "📈 Top Charts":
        st.markdown('<div class="netflix-title" style="font-size:2.4rem">TOP CHARTS</div>', unsafe_allow_html=True)
        st.markdown('<div class="netflix-subtitle">Rankings · Genre Insights · Type Breakdown</div>', unsafe_allow_html=True)

        ct1, ct2, ct3, ct4 = st.tabs(["🏆 Top Rated", "👥 Most Popular", "🎭 Genre Analysis", "📺 Type Breakdown"])

        with ct1:
            n_top = st.slider("Show top N", 10, 50, 20)
            top_r = df.nlargest(n_top, "rating")[["name", "rating", "type", "genre", "members"]].reset_index(drop=True)
            fig_tr = px.bar(top_r, x="rating", y="name", orientation="h",
                            color="rating", color_continuous_scale="Reds",
                            hover_data=["type", "genre", "members"],
                            template="plotly_dark", title=f"Top {n_top} Rated Anime")
            fig_tr.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                 font_color="#e5e5e5", height=max(400, n_top * 22),
                                 yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_tr, use_container_width=True)

            st.markdown('<div class="section-header">🎬 Top 10 Poster Wall</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            for i, (_, row) in enumerate(top_r.head(10).iterrows()):
                with cols[i % 5]:
                    st.markdown(anime_card_html(row["name"], fetch_poster(row["name"]),
                                                f"{row['rating']:.2f}", row["type"], row["genre"]),
                                unsafe_allow_html=True)

        with ct2:
            top_p = df.nlargest(20, "members")[["name", "members", "rating", "type"]].reset_index(drop=True)
            fig_p = px.bar(top_p, x="members", y="name", orientation="h",
                           color="members", color_continuous_scale="Blues",
                           template="plotly_dark", title="Top 20 Most Popular Anime")
            fig_p.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                font_color="#e5e5e5", height=540,
                                yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_p, use_container_width=True)

        with ct3:
            ge = df.copy()
            ge["genre_list"] = ge["genre"].str.split(",")
            ge = ge.explode("genre_list"); ge["genre_list"] = ge["genre_list"].str.strip()
            gs = ge.groupby("genre_list").agg(
                count=("name", "count"), avg_rating=("rating", "mean"),
                avg_members=("members", "mean")
            ).reset_index().sort_values("count", ascending=False).head(20)
            fig_gs = px.scatter(gs, x="count", y="avg_rating", size="avg_members",
                                color="avg_rating", text="genre_list",
                                color_continuous_scale="Reds", template="plotly_dark",
                                title="Genre: Count vs Avg Rating (bubble = avg members)")
            fig_gs.update_traces(textposition="top center", textfont_size=9)
            fig_gs.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                 font_color="#e5e5e5", height=480)
            st.plotly_chart(fig_gs, use_container_width=True)

        with ct4:
            ts = df.groupby("type").agg(count=("name","count"), avg_rating=("rating","mean")).reset_index()
            ca, cb = st.columns(2)
            with ca:
                fig_tc = px.pie(ts, values="count", names="type", hole=0.5,
                                template="plotly_dark", title="Count by Type",
                                color_discrete_sequence=px.colors.qualitative.Bold)
                fig_tc.update_layout(paper_bgcolor="#141414", font_color="#e5e5e5")
                st.plotly_chart(fig_tc, use_container_width=True)
            with cb:
                fig_ta = px.bar(ts, x="type", y="avg_rating", color="avg_rating",
                                color_continuous_scale="Reds", template="plotly_dark",
                                title="Avg Rating by Type")
                fig_ta.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                     font_color="#e5e5e5")
                st.plotly_chart(fig_ta, use_container_width=True)

    # ── USER RATINGS ─────────────────────────────────────────
    elif page == "👤 User Ratings":
        st.markdown('<div class="netflix-title" style="font-size:2.4rem">USER RATINGS</div>', unsafe_allow_html=True)
        st.markdown('<div class="netflix-subtitle">Collaborative Signals · Rating Distributions</div>', unsafe_allow_html=True)

        if not use_ratings:
            st.warning("⚠️ rating.csv not found or failed to load.")
        else:
            merged = pd.merge(
                rating_df[rating_df["rating"] != -1],
                df[["anime_id", "name", "genre", "type", "rating"]].rename(columns={"rating": "anime_rating"}),
                on="anime_id", how="inner"
            ).rename(columns={"rating": "user_rating"})

            r1, r2, r3, r4 = st.columns(4)
            for col, val, lbl in zip(
                [r1, r2, r3, r4],
                [f"{len(merged):,}", f"{merged['user_id'].nunique():,}",
                 f"{merged['anime_id'].nunique():,}", f"{merged['user_rating'].mean():.2f}"],
                ["Total Ratings", "Unique Users", "Rated Anime", "Avg User Rating"]
            ):
                col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>',
                             unsafe_allow_html=True)

            st.markdown('<div class="section-header">📊 Distributions</div>', unsafe_allow_html=True)
            ua, ub = st.columns(2)
            with ua:
                fig_ud = px.histogram(merged, x="user_rating", nbins=10,
                                      color_discrete_sequence=["#e50914"],
                                      template="plotly_dark", title="User Rating Distribution")
                fig_ud.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f", font_color="#e5e5e5")
                st.plotly_chart(fig_ud, use_container_width=True)
            with ub:
                samp = merged.sample(min(3000, len(merged)), random_state=42)
                fig_uv = px.scatter(samp, x="anime_rating", y="user_rating", color="type",
                                    opacity=0.5, template="plotly_dark",
                                    title="User Rating vs Anime Avg Rating",
                                    color_discrete_sequence=px.colors.qualitative.Bold)
                fig_uv.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f", font_color="#e5e5e5")
                st.plotly_chart(fig_uv, use_container_width=True)

            st.markdown('<div class="section-header">🏆 Most Rated Anime</div>', unsafe_allow_html=True)
            mr = merged.groupby("name").agg(
                num_ratings=("user_rating", "count"),
                avg_user_rating=("user_rating", "mean"),
                anime_rating=("anime_rating", "first")
            ).reset_index().sort_values("num_ratings", ascending=False).head(20)
            fig_mr = px.bar(mr, x="num_ratings", y="name", orientation="h",
                            color="avg_user_rating", color_continuous_scale="Reds",
                            hover_data=["avg_user_rating", "anime_rating"],
                            template="plotly_dark", title="Top 20 Most Rated Anime")
            fig_mr.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                 font_color="#e5e5e5", height=540,
                                 yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_mr, use_container_width=True)

            st.markdown('<div class="section-header">🎭 Avg User Rating by Genre</div>', unsafe_allow_html=True)
            gu = merged.copy()
            gu["genre_list"] = gu["genre"].str.split(",")
            gu = gu.explode("genre_list"); gu["genre_list"] = gu["genre_list"].str.strip()
            ga = gu.groupby("genre_list")["user_rating"].mean().reset_index()
            ga.columns = ["genre", "avg_user_rating"]
            ga = ga.sort_values("avg_user_rating", ascending=False).head(20)
            fig_ga = px.bar(ga, x="avg_user_rating", y="genre", orientation="h",
                            color="avg_user_rating", color_continuous_scale="Reds",
                            template="plotly_dark", title="Avg User Rating by Genre")
            fig_ga.update_layout(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                                 font_color="#e5e5e5", height=480,
                                 yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_ga, use_container_width=True)

    # ── FOOTER ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#444;font-size:0.76rem;padding:16px 0'>
        🎌 AniRec · Streamlit · scikit-learn · SHAP · Plotly · Jikan API v4
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
