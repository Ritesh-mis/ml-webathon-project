import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Performance Predictor",
    page_icon="🤖",
    layout="wide"
)


import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
<style>

/* ===== Main Gradient Background ===== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #89CFF0, #EFB6C8);
}

/* ===== Glass Card Main Block ===== */
.main .block-container {
    background: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.15);
}

/* ===== Sidebar Soft Glass ===== */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.6);
}

/* ===== Buttons Rounded ===== */
.stButton>button {
    border-radius: 12px;
    height: 3em;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)




# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>🎯 AI Performance Predictor</h1>
<p style='text-align:center;color:gray;'>Interactive ML Dashboard with Charts & Model Insights</p>
""", unsafe_allow_html=True)

# ---------------- DATA ----------------
data = pd.DataFrame({
    "hours":[1,2,5,6,8,3,7,4,2,9],
    "attendance":[40,50,80,90,95,60,85,70,55,97],
    "assignments":[1,2,7,8,9,3,8,5,2,10],
    "pass":[0,0,1,1,1,0,1,1,0,1]
})

X = data[["hours","attendance","assignments"]]
y = data["pass"]

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Input Controls")

hours = st.sidebar.slider("Study Hours", 0, 12, 5)
attendance = st.sidebar.slider("Attendance %", 0, 100, 75)
assignments = st.sidebar.slider("Assignments Done", 0, 10, 5)

# ---------------- METRICS ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", "Logistic Reg")
c2.metric("Accuracy", f"{acc*100:.1f}%")
c3.metric("Rows", len(data))
c4.metric("Features", X.shape[1])

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "🔮 Prediction",
    "📊 Data Charts",
    "🧠 Model Info"
])

# ================= TAB 1 — PREDICTION =================
with tab1:
    st.subheader("Run Live Prediction")

    if st.button("🚀 Predict", use_container_width=True):
        pred = model.predict([[hours, attendance, assignments]])

        if pred[0] == 1:
            st.success("✅ Student Will PASS")
            st.balloons()
        else:
            st.error("❌ Student Will FAIL")

# ================= TAB 2 — CHARTS =================
with tab2:
    st.subheader("Dataset Preview")
    st.dataframe(data, use_container_width=True)

    colA, colB = st.columns(2)

    with colA:
        st.write("Hours Studied Distribution")
        st.bar_chart(data["hours"])

    with colB:
        st.write("Attendance Trend")
        st.line_chart(data["attendance"])

    st.write("Feature Comparison")
    st.bar_chart(data[["hours","attendance","assignments"]])

# ================= TAB 3 — MODEL INFO =================
with tab3:
    st.subheader("Model Details")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Problem Type:** Binary Classification")
    st.write("**Target:** Pass / Fail")

    st.write("### Features Used")
    st.write(list(X.columns))

    # Feature importance (coefficients)
    coef = pd.DataFrame({
        "Feature": X.columns,
        "Impact": model.coef_[0]
    })

    st.write("### Feature Impact")
    st.dataframe(coef, use_container_width=True)
    st.bar_chart(coef.set_index("Feature"))

    st.write("### Training Process")
    st.write("""
    - Data split into train/test  
    - Model trained on training data  
    - Accuracy evaluated on unseen data  
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built for Hackathon Demo — ML + Streamlit Dashboard")
