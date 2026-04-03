# 🎌 AniRec — AI-Powered Anime Recommendation System

A professional, resume-ready anime recommendation system with Netflix-style UI, built with Streamlit, scikit-learn, and SHAP explainability.

## ✨ Features

- **🎬 Content-Based Recommender**: TF-IDF + Cosine Similarity on genres and types
- **🎨 Netflix-Style Dark UI**: Custom CSS with smooth animations and hover effects
- **📊 Comprehensive EDA**: Distribution analysis, skewness detection, log transformations
- **🔬 PCA Visualization**: 2D and 3D dimensionality reduction with explained variance
- **🤖 Clustering**: KMeans and DBSCAN with silhouette scores and elbow curves
- **🧠 SHAP Explainability**: Understand which features drive recommendations
- **🖼️ Real Anime Posters**: Fetched dynamically from Jikan API v4
- **🔍 Fuzzy Search**: Find anime even with typos
- **👤 User Ratings Analysis**: Collaborative filtering insights
- **📈 Interactive Charts**: Plotly visualizations with dark theme

## 📁 Project Structure

```
anime_recommendation/
├── app.py                    # Main Streamlit application
├── download_posters.py       # Script to pre-download posters
├── requirements.txt          # Python dependencies
├── anime.csv                 # Anime dataset (required)
├── rating.csv                # User ratings dataset (required)
├── anime_recommendation/     # Folder for downloaded posters
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure `anime.csv` and `rating.csv` are in the project root directory.

### 3. (Optional) Download Posters

Pre-download anime posters for faster loading:

```bash
python download_posters.py
```

This will download the top 100 anime posters to `anime_recommendation/` folder.

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Anime Recommender"
git remote add origin <your-repo-url>
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository, branch (`main`), and main file (`app.py`)
5. Click "Deploy"

### Step 3: Upload Data Files

Since `rating.csv` is large (>100MB), you have two options:

**Option A: Use Git LFS**
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
git add rating.csv
git commit -m "Add large CSV with LFS"
git push
```

**Option B: Host externally**
Upload `rating.csv` to Google Drive, Dropbox, or AWS S3, then modify `load_data()` in `app.py` to download from URL.

## 📊 Dataset

This project uses the **MyAnimeList Dataset**:
- `anime.csv`: ~12,000 anime with metadata (name, genre, type, rating, members)
- `rating.csv`: ~7M user ratings (user_id, anime_id, rating)

## 🎯 Key Technical Highlights

### EDA & Feature Engineering
- **Skewness detection**: Identifies right-skewed distributions in `members` and `episodes`
- **Log transformation**: Applies `log1p` to normalize skewed features
- **MinMax scaling**: Normalizes features for model input
- **Missing value handling**: Median imputation for ratings, genre filling

### Content-Based Recommender
- **TF-IDF Vectorization**: Extracts features from genre and type text
- **Cosine Similarity**: Computes pairwise similarity matrix
- **Top-N Recommendations**: Returns most similar anime with match percentages

### SHAP Explainability
- **KernelExplainer**: Explains which TF-IDF features drive similarity scores
- **Feature importance**: Visualizes top contributing genres/types
- **Heatmaps**: Shows SHAP values across samples

### PCA & Clustering
- **PCA**: Reduces high-dimensional TF-IDF space to 2D/3D for visualization
- **KMeans**: Groups anime into clusters with silhouette scoring
- **DBSCAN**: Density-based clustering for outlier detection
- **Elbow curve**: Helps determine optimal number of clusters

## 🎨 UI/UX Features

- **Netflix-inspired design**: Dark theme with red accents
- **Anime cards**: Hover effects, poster images, match percentages
- **Responsive layout**: Works on desktop and tablet
- **Interactive filters**: Genre, type, and result count selection
- **Fuzzy search**: Autocomplete with typo tolerance
- **Smooth animations**: CSS transitions and hover effects

## 📦 Dependencies

- `streamlit` — Web framework
- `pandas` — Data manipulation
- `numpy` — Numerical computing
- `scikit-learn` — ML algorithms (TF-IDF, PCA, KMeans, DBSCAN)
- `plotly` — Interactive visualizations
- `shap` — Model explainability
- `requests` — API calls for posters
- `fuzzywuzzy` — Fuzzy string matching
- `scipy` — Statistical functions

## 🔧 Configuration

### Adjust Rating Sample Size

In `app.py`, modify `load_data()`:

```python
rating = pd.read_csv("rating.csv", nrows=500_000)  # Change to 1M, 2M, etc.
```

### Change Poster Source

To use local posters instead of API:

```python
def fetch_poster(name):
    local_path = f"anime_recommendation/{anime_id}.jpg"
    if os.path.exists(local_path):
        return local_path
    # Fallback to API...
```

## 📈 Performance Tips

1. **Cache data loading**: Already implemented with `@st.cache_data`
2. **Sample large datasets**: `rating.csv` is sampled to 500k rows by default
3. **Limit API calls**: Use `download_posters.py` to pre-fetch images
4. **Reduce PCA samples**: Lower `n_samples` in SHAP computation for faster results

## 🎓 Resume Highlights

This project demonstrates:
- ✅ End-to-end ML pipeline (EDA → Feature Engineering → Model → Deployment)
- ✅ Content-based recommendation system
- ✅ Model explainability with SHAP
- ✅ Dimensionality reduction (PCA)
- ✅ Unsupervised learning (KMeans, DBSCAN)
- ✅ Data visualization (Plotly)
- ✅ Web app development (Streamlit)
- ✅ API integration (Jikan API)
- ✅ Production deployment (Streamlit Cloud)
- ✅ Professional UI/UX design

## 📝 License

This project is for educational and portfolio purposes. Anime data is sourced from MyAnimeList. Posters are fetched from Jikan API (unofficial MAL API).

## 🤝 Contributing

Feel free to fork, improve, and submit PRs!

## 📧 Contact

Built by [Your Name] — [your.email@example.com]

---

⭐ **Star this repo if you found it helpful!**
