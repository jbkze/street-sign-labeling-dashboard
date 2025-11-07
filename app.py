import streamlit as st
from supabase import create_client
import pandas as pd
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Labeling Dashboard")

SUPABASE_URL = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
TABLE_NAME = "labels"
PAGE_SIZE = 1000  # Supabase max rows per query

# ---------------- CONNECT ----------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------- LOAD ALL DATA WITH PAGINATION + STATUS ----------------
def load_all_data():
    # Gesamtzahl holen
    res = supabase.table(TABLE_NAME).select("id", count="exact").limit(1).execute()
    total_count = res.count or 0

    all_rows = []
    progress_text = "📦 Lade Daten aus Supabase..."
    progress_bar = st.progress(0, text=progress_text)

    for offset in range(0, total_count, PAGE_SIZE):
        start = offset
        end = offset + PAGE_SIZE - 1
        response = supabase.table(TABLE_NAME).select("*").range(start, end).execute()
        rows = response.data or []
        all_rows.extend(rows)
        time.sleep(0.05)  # minimale Pause, um API nicht zu überlasten

        progress = min(len(all_rows) / total_count, 1.0)
        progress_bar.progress(progress, text=f"{progress_text} ({len(all_rows)}/{total_count})")

    progress_bar.empty()  # entfernt die Progressbar nach Abschluss
    return pd.DataFrame(all_rows)


# ---------------- FETCH ----------------
df = load_all_data()

if df.empty:
    st.warning("Keine Daten gefunden.")
    st.stop()

# ---------------- METRICS ----------------
total_labels = len(df)
unique_images = df["image"].nunique()
multi_labeled_count = (
    df.groupby("image")["user"].nunique().ge(2).sum()
)

at_least_single_labeled_count = (
    df.groupby("image")["user"].nunique().ge(1).sum()
)

col1, col2, col3 = st.columns(3)
col1.metric("Progress", f'{int(at_least_single_labeled_count / 7656 * 100)} %')
col2.metric("Images labeled by ≥1 users", at_least_single_labeled_count)
col3.metric("Images labeled by ≥2 users", multi_labeled_count)

st.divider()

# ---------------- TOP 10 CONTRIBUTORS ----------------
st.subheader("🏅 Top Contributors")
top_users = df["user"].value_counts().reset_index()
top_users.columns = ["user", "label_count"]
st.bar_chart(top_users.set_index("user")["label_count"], horizontal=True, sort=False)

# ---------------- MOST FREQUENT LABELS ----------------
st.subheader("🏷️ Most Selected Labels")
df_exploded = df.explode("label")
top_labels = df_exploded["label"].value_counts().head(20).reset_index()
top_labels.columns = ["label", "count"]
st.bar_chart(top_labels.set_index("label")["count"], horizontal=True, sort=False)

# ---------------- LABELING OVER TIME ----------------
st.subheader("📈 Labeling Progress Over Time")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

if df["timestamp"].dt.tz is None:
    df["timestamp"] = df["timestamp"].dt.tz_localize('UTC')

df["timestamp"] = df["timestamp"].dt.tz_convert('Europe/Berlin')
df_sorted = df.sort_values("timestamp")
df_sorted["cumulative_count"] = range(1, len(df_sorted) + 1)
st.line_chart(df_sorted.set_index("timestamp")["cumulative_count"])

# ---------------- LABEL CO-OCCURRENCE ----------------
st.subheader("🔥 Label Co-Occurrence")

import ast
from itertools import combinations
import plotly.express as px

# Explodierte Labels liegen bereits als Listen vor, ggf. sicherstellen
df['label'] = df['label'].apply(lambda x: x if isinstance(x, list) else ast.literal_eval(x))

# Alle möglichen Labels
all_labels = df_exploded['label'].unique()
co_matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)

# Co-Occurrence zählen
for labels in df['label']:
    for a, b in combinations(labels, 2):
        co_matrix.loc[a, b] += 1
        co_matrix.loc[b, a] += 1  # symmetrisch
# Diagonale = Gesamtanzahl eines Labels
for label in all_labels:
    co_matrix.loc[label, label] = df_exploded['label'].value_counts().get(label, 0)

# Plotly Heatmap
fig = px.imshow(
    co_matrix.values,
    x=co_matrix.columns,
    y=co_matrix.index,
    text_auto=True,
    color_continuous_scale='deep'  # <-- rot-gelb
)
fig.update_layout(
    xaxis_title="Label",
    yaxis_title="Label",
    xaxis_side="top",
    width=700,
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- AGREEMENT BETWEEN TWO LABELERS ----------------
st.subheader("👥 Label Agreement")

import ast
import numpy as np
import plotly.express as px

# --- Helper to clean label field ---
def ensure_list(x):
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return []
    if not isinstance(x, list):
        return []
    flat = []
    for item in x:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat

df["label"] = df["label"].apply(ensure_list)

# --- Select images with exactly two annotators ---
multi_two = df.groupby("image").filter(lambda g: g["user"].nunique() == 2)

# --- Compute agreement (Jaccard index) ---
agreement = []
for image, group in multi_two.groupby("image"):
    if len(group) == 2:
        labels_a = set(group.iloc[0]["label"])
        labels_b = set(group.iloc[1]["label"])
        intersection = len(labels_a & labels_b)
        union = len(labels_a | labels_b)
        jaccard = intersection / union if union > 0 else np.nan
        agreement.append({
            "image": image,
            "user_a": group.iloc[0]["user"],
            "user_b": group.iloc[1]["user"],
            "labels_a": labels_a,
            "labels_b": labels_b,
            "common_labels": list(labels_a & labels_b),
            "jaccard": jaccard,
            "n_labels_a": len(labels_a),
            "n_labels_b": len(labels_b)
        })

agreement_df = pd.DataFrame(agreement).dropna(subset=["jaccard"])

if agreement_df.empty:
    st.info("Keine Bilder mit genau zwei Annotatoren gefunden.")
else:
    # --- KPI ---
    mean_jaccard = agreement_df["jaccard"].mean()
    median_jaccard = agreement_df["jaccard"].median()
    st.metric("Ø Agreement (Jaccard)", f"{mean_jaccard:.2f}")
    st.metric("Median Agreement", f"{median_jaccard:.2f}")

    # --- Distribution Plot ---
    fig = px.histogram(
        agreement_df,
        x="jaccard",
        nbins=20,
        color_discrete_sequence=["#F1C232"],
        title="Distribution of Annotator Agreement (Jaccard Index)",
    )
    fig.update_layout(
        xaxis_title="Jaccard Index (0 = no overlap, 1 = identical)",
        yaxis_title="Number of Images",
        bargap=0.1,
    )
    st.plotly_chart(fig, use_container_width=True)

# --- AWS URL Prefix ---
AWS_URL = "https://d3b45akprxecp4.cloudfront.net/GTSD-220-test/"

# --- Optional table for low-agreement cases ---
low_agreement = agreement_df.sort_values("jaccard")

with st.expander("📉 Lowest Agreement Examples"):
    # Bild-URL-Spalte hinzufügen
    low_agreement["image_url"] = AWS_URL + low_agreement["image"]

    def fmt_labels(labels):
        if not labels:
            return "–"
        return ", ".join(sorted(labels))

    for _, row in low_agreement.iterrows():
        st.markdown(f"**Image:** `{row['image']}` — **Jaccard:** {row['jaccard']:.2f}")
        st.image(AWS_URL + row["image"], width=250)
        st.markdown(
            f"👤 **{row['user_a']}** labels: `{fmt_labels(row['labels_a'])}`  \n"
            f"👤 **{row['user_b']}** labels: `{fmt_labels(row['labels_b'])}`  \n"
        )
        st.divider()
