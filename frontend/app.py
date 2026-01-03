import os, sys, json, time
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------- KAFKA (OPTIONAL) ----------------
try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except Exception:
    KAFKA_AVAILABLE = False

# ---------------- PATH ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from backend.predictor import SurvivalPredictor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Titanic Survival Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD CSS ----------------
with open("frontend/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- LOAD MODEL & DATA ----------------
@st.cache_resource
def load_model():
    return SurvivalPredictor("models/survival_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/titanic_cleaned.csv")

predictor = load_model()
df = load_data()

# ==================================================
# 🧭 SIDEBAR – GLOBAL CONTROLS
# ==================================================
st.sidebar.markdown("## ⚙️ Global Controls")

threshold = st.sidebar.slider(
    "Classification Threshold",
    0.1, 0.9, 0.5, 0.05
)

show_advanced = st.sidebar.checkbox("Show Advanced Model Insights")

st.sidebar.divider()
st.sidebar.caption("Titanic Survival Intelligence v1.1")

# ==================================================
# 🧠 HEADER
# ==================================================
st.markdown("""
<div class="header-card">
  <h1>🚢 Titanic Survival Intelligence</h1>
  <p>A machine learning platform for survival prediction & analytics</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# 🎛 PASSENGER FILTERS
# ==================================================
st.markdown("## Passenger Filters")

f1, f2, f3, f4 = st.columns([1.2, 1.8, 1.2, 1.8])

with f1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])

with f2:
    age_range = st.slider("Age Range", 0, 80, (20, 60))

with f3:
    gender = st.radio("Gender", ["male", "female"], horizontal=True)

with f4:
    max_fare = st.slider("Max Fare", 0, 500, 200)

filtered = df[
    (df["Pclass"] == pclass) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Sex"] == gender) &
    (df["Fare"] <= max_fare)
].copy()

# 🔐 SAFE PREDICTION (CRITICAL FIX)
if filtered.shape[0] > 0:
    probs, preds, risks = predictor.predict(filtered)
else:
    probs, preds, risks = [], [], []

# ==================================================
# 📊 KPI CARDS
# ==================================================
st.markdown("## Intelligence KPIs")

k1, k2, k3, k4 = st.columns(4)

def kpi(title, value):
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """

k1.markdown(kpi("Passengers", len(filtered)), unsafe_allow_html=True)
k2.markdown(
    kpi("Avg Survival", f"{sum(probs)/len(probs):.2%}" if len(probs) else "—"),
    unsafe_allow_html=True
)
k3.markdown(kpi("High Risk", int(sum(r > 60 for r in risks))), unsafe_allow_html=True)
k4.markdown(kpi("Model ROC-AUC", "0.91"), unsafe_allow_html=True)

# ==================================================
# 📋 BATCH PREDICTIONS
# ==================================================
st.markdown("## Batch Survival Predictions")

s1, s2 = st.columns([2, 1])

with s1:
    sort_col = st.selectbox(
        "Sort by",
        ["Survival Probability", "Risk Score", "Age", "Fare"]
    )

with s2:
    order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

if filtered.shape[0] > 0:
    filtered["Survival Probability"] = probs
    filtered["Risk Score"] = risks
    filtered["Prediction"] = preds

    filtered = filtered.sort_values(
        sort_col,
        ascending=(order == "Ascending")
    )

    st.dataframe(
        filtered[
            ["Name", "Age", "Sex", "Pclass", "Fare",
             "Survival Probability", "Risk Score", "Prediction"]
        ],
        height=320,
        width="stretch"
    )
else:
    st.warning("No passengers match the selected filters.")

# ==================================================
# 🔮 INDIVIDUAL PREDICTION (ALWAYS SAFE)
# ==================================================
st.markdown("## Individual Passenger Prediction")

i1, i2, i3 = st.columns(3)

with i1:
    age = st.slider("Age", 0, 80, 30)
    sex_i = st.radio("Sex", ["male", "female"], horizontal=True)

with i2:
    pclass_i = st.selectbox("Class", [1, 2, 3])
    fare = st.slider("Fare", 0.0, 500.0, 50.0)

with i3:
    sibsp = st.number_input("Siblings", 0, 8, 0)
    parch = st.number_input("Parents", 0, 6, 0)

sample = pd.DataFrame([{
    "Age": age,
    "Sex": sex_i,
    "Pclass": pclass_i,
    "Fare": fare,
    "SibSp": sibsp,
    "Parch": parch
}])

p, pr, r = predictor.predict(sample)

o1, o2, o3 = st.columns(3)

o1.markdown(kpi("Survival Probability", f"{p[0]:.2%}"), unsafe_allow_html=True)
o2.markdown(kpi("Risk Index", f"{r[0]}/100"), unsafe_allow_html=True)
o3.markdown(
    kpi("Prediction", "Survived" if pr[0] == 1 else "Did Not Survive"),
    unsafe_allow_html=True
)

st.progress(int(p[0] * 100))

# ==================================================
# 📈 ANALYTICS (SAFE)
# ==================================================
st.markdown("## Analytics Overview")

if len(probs) > 0:
    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.histogram(
            probs, nbins=20,
            title="Survival Probability Distribution"
        )
        fig1.update_layout(height=300)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.histogram(
            risks, nbins=20,
            title="Risk Score Distribution"
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Analytics unavailable — no passengers after filtering.")

# ==================================================
# 🔴 REAL-TIME KAFKA PREDICTIONS
# ==================================================
st.markdown("## 🔴 Real-Time Kafka Predictions")

if not KAFKA_AVAILABLE:
    st.info("Kafka client not available. Streaming disabled.")
else:
    enable_stream = st.toggle("Enable Kafka Streaming")

    if enable_stream:
        try:
            consumer = KafkaConsumer(
                "titanic_stream",
                bootstrap_servers="localhost:9092",
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                consumer_timeout_ms=3000
            )

            for msg in consumer:
                live_row = pd.DataFrame([msg.value])

                probs_live, preds_live, risks_live = predictor.predict(live_row)

                live_row["Survival Probability"] = round(probs_live[0], 3)
                live_row["Risk Score"] = risks_live[0]
                live_row["Prediction"] = (
                    "Survived" if preds_live[0] == 1 else "Did Not Survive"
                )

                st.dataframe(live_row, height=120, width="stretch")
                break

        except Exception:
            st.error("Kafka broker not running. Start Kafka + Zookeeper.")
