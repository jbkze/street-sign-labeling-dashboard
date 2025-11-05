import streamlit as st
from supabase import create_client
import pandas as pd
import pytz

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Labeling Dashboard")

SUPABASE_URL = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
TABLE_NAME = "labels"  # Deine Tabelle

# ---------------- CONNECT ----------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data(ttl=300)
def load_data():
    data = supabase.table(TABLE_NAME).select("*").execute()
    return pd.DataFrame(data.data)

df = load_data()

if df.empty:
    st.warning("Keine Daten gefunden.")
    st.stop()

# ---------------- METRICS ----------------
total_labels = len(df)
unique_images = df["image"].nunique()

multi_labeled_count = (
    df.groupby("image")["user"]
    .nunique()
    .ge(2)
    .sum()
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Labels", total_labels)
col2.metric("Unique Images", unique_images)
col3.metric("Images labeled by ≥2 users", multi_labeled_count)

st.divider()

# ---------------- TOP 10 CONTRIBUTORS ----------------
st.subheader("🏅 Top 10 Contributors")

# Top 10 Users mit Anzahl Labels
top_users = df["user"].value_counts().head(10).reset_index()
top_users.columns = ["user", "label_count"]

# Sortieren nach label_count absteigend
top_users = top_users.sort_values("label_count", ascending=False)

# Streamlit Bar Chart
st.bar_chart(top_users.set_index("user")["label_count"], horizontal=True, sort=False)

# ---------------- MOST FREQUENT LABELS ----------------
st.subheader("🏷️ Most Selected Labels")

# Explodiere die Arrays in einzelne Labels
df_exploded = df.explode("label")

# Zähle die Häufigkeit jedes einzelnen Labels
top_labels = df_exploded["label"].value_counts().head(10).reset_index()
top_labels.columns = ["label", "count"]

# Sortieren absteigend (optional, aber st.bar_chart zeigt Index)
top_labels = top_labels.sort_values("count", ascending=False)

# Bar Chart
st.bar_chart(top_labels.set_index("label")["count"], horizontal=True, sort=False)

# ---------------- LABELING OVER TIME ----------------
st.subheader("📈 Labeling Progress Over Time")

# Timestamp in datetime konvertieren
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Nach Minute gruppieren
df_time = df.groupby(df["timestamp"].dt.floor("min")).size().reset_index(name="count")

# Nur gültige Timestamps
df_time = df_time.dropna(subset=["timestamp"])

# Index setzen
df_time.set_index("timestamp", inplace=True)

# Gleitender Mittelwert (z.B. 5 Minuten Fenster)
df_time["count_smoothed"] = df_time["count"].rolling(window=5, min_periods=1).mean()

# Line chart mit geglätteten Werten
st.line_chart(df_time["count_smoothed"])
