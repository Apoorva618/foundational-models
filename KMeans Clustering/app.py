import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Penguin Species Clustering", layout="wide")
st.title("🐧 Clustering Penguin Species using KMeans")
st.markdown(
    "Explore how KMeans groups penguins by bill dimensions, flipper length, "
    "body mass, and sex. Use the sidebar slider to change **k** and watch the "
    "PCA plot update live."
)

# ── 1. Load & preprocess ─────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    # Uses seaborn's built-in penguins dataset — no local file needed
    df = sns.load_dataset("penguins")

    # Rename to match original notebook column names
    df = df.rename(columns={
        "bill_length_mm": "culmen_length_mm",
        "bill_depth_mm":  "culmen_depth_mm",
    })

    # Drop rows with any missing values
    df = df.dropna().reset_index(drop=True)

    numeric_cols = ["culmen_length_mm", "culmen_depth_mm",
                    "flipper_length_mm", "body_mass_g"]

    # Scale continuous features
    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(df[numeric_cols])
    scaled_df  = pd.DataFrame(scaled_arr, columns=numeric_cols)

    # One-hot-encode sex (drop_first keeps one binary column: 'Male')
    sex_encoded = pd.get_dummies(df["sex"], drop_first=True, dtype=int)

    X_train = pd.concat(
        [scaled_df, sex_encoded.reset_index(drop=True)], axis=1
    )

    # Fit PCA once on the full feature matrix
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train)

    return df, X_train, X_pca, numeric_cols

df, X_train, X_pca, numeric_cols = load_and_preprocess()

# ── 2. Pre-compute elbow data (cached) ───────────────────────────────────────
@st.cache_data
def compute_elbow(data_hash, data):   # data_hash lets us cache by content
    k_range, inertias = [], []
    for k in range(1, 8):
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km.fit(data)
        k_range.append(k)
        inertias.append(km.inertia_)
    return k_range, inertias

# Stable hash so cache isn't busted on every run
data_hash   = hash(X_train.values.tobytes())
k_range, inertias = compute_elbow(data_hash, X_train)

# ── 3. Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Parameters")
k_selected = st.sidebar.slider(
    "Number of Clusters (k):", min_value=2, max_value=6, value=3
)
st.sidebar.markdown(f"**Dataset:** {len(df)} penguins after dropping NaNs")
st.sidebar.markdown(
    "**Features used:** culmen length, culmen depth, flipper length, "
    "body mass, sex (binary)"
)

# ── 4. Fit KMeans for selected k ─────────────────────────────────────────────
kmeans         = KMeans(n_clusters=k_selected, init="k-means++",
                        random_state=42, n_init=10)
dynamic_labels = kmeans.fit_predict(X_train)

# ── 5. Plot row: Elbow | PCA ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

PALETTE = sns.color_palette("Set1", n_colors=6)

with col1:
    st.subheader("① Elbow Method")
    fig_elbow, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(k_range, inertias, marker="o", color="#2b5c8f", linewidth=2,
            label="Inertia (WCSS)")
    ax.axvline(x=k_selected, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Selected  k = {k_selected}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia / WCSS")
    ax.set_title("Optimal k Selection")
    ax.set_xticks(k_range)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig_elbow)
    plt.close(fig_elbow)

with col2:
    st.subheader(f"② PCA Cluster Separation  (k = {k_selected})")
    fig_pca, ax = plt.subplots(figsize=(6, 4.5))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=dynamic_labels,
        palette=PALETTE[:k_selected],
        alpha=0.85, s=60, ax=ax,
        legend="full",
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"KMeans — {k_selected} Clusters in PCA Space")
    ax.legend(title="Cluster", loc="best")
    ax.grid(alpha=0.2)
    st.pyplot(fig_pca)
    plt.close(fig_pca)

# ── 6. Cluster characteristics table ─────────────────────────────────────────
st.divider()
st.subheader(f"📋 Average Feature Values per Cluster  (k = {k_selected})")

summary_df = (
    df.assign(Cluster=dynamic_labels)
      .groupby("Cluster")[numeric_cols]
      .mean()
      .round(2)
)
summary_df.index = [f"Cluster {i}" for i in summary_df.index]

st.dataframe(
    summary_df.style.background_gradient(cmap="Blues", axis=0),
    use_container_width=True,
)

# ── 7. Cluster size breakdown ─────────────────────────────────────────────────
st.subheader("🔢 Cluster Size Breakdown")
size_df = (
    pd.Series(dynamic_labels, name="Cluster")
      .value_counts()
      .sort_index()
      .rename(index=lambda i: f"Cluster {i}")
      .reset_index()
)
size_df.columns = ["Cluster", "Count"]
size_df["Share (%)"] = (size_df["Count"] / size_df["Count"].sum() * 100).round(1)
st.dataframe(size_df, use_container_width=True, hide_index=True)
