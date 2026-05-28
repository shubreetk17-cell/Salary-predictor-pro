import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import export_text

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    .main { background-color: #f8f9ff; }

    .stApp { background: linear-gradient(135deg, #f0f4ff 0%, #faf8ff 50%, #f0fff4 100%); }

    .hero-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 20px;
        padding: 40px;
        color: white;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    .hero-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99,179,237,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #63b3ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.65);
        margin-top: 8px;
        font-weight: 300;
    }

    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border-left: 4px solid #4776e6;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; font-family: 'JetBrains Mono', monospace; }
    .metric-label { font-size: 0.82rem; color: #888; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

    .result-high {
        background: linear-gradient(135deg, #0f9b58, #00c88e);
        border-radius: 20px;
        padding: 35px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(15,155,88,0.35);
    }
    .result-low {
        background: linear-gradient(135deg, #e05f2c, #f7971e);
        border-radius: 20px;
        padding: 35px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(224,95,44,0.35);
    }
    .result-label { font-size: 1rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.12em; }
    .result-value { font-size: 3rem; font-weight: 700; margin: 8px 0; }
    .result-desc { font-size: 0.92rem; opacity: 0.85; }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
        border-bottom: 3px solid #4776e6;
        padding-bottom: 8px;
        margin-bottom: 20px;
        display: inline-block;
    }

    .input-card {
        background: white;
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }

    .skill-tag {
        display: inline-block;
        background: #eef2ff;
        color: #4776e6;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #4776e6, #8e54e9);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 40px;
        font-size: 1.05rem;
        font-weight: 700;
        font-family: 'Sora', sans-serif;
        width: 100%;
        transition: all 0.3s;
        letter-spacing: 0.03em;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(71,118,230,0.45);
    }

    .stSelectbox > div > div { border-radius: 10px; }
    .stSlider > div { padding: 0; }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("salary_model.pkl")
    dept_enc = joblib.load("dept_encoder.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, dept_enc, meta

@st.cache_data
def load_data():
    return pd.read_csv("employee_salary_data.csv")

model, dept_enc, meta = load_model()
df = load_data()


# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <p class="hero-title">💼 Employee Salary Predictor</p>
    <p class="hero-subtitle">
        Decision Tree ML Model &nbsp;·&nbsp; 1,500 Employee Records &nbsp;·&nbsp;
        Predicts High / Low Salary Category
    </p>
</div>
""", unsafe_allow_html=True)


# ── Top Metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    (f"{meta['accuracy']}%",       "Test Accuracy",        "#4776e6"),
    (f"{meta['cv_accuracy']}%",    "5-Fold CV Accuracy",   "#0f9b58"),
    (f"{meta['tree_depth']}",      "Tree Depth",           "#8e54e9"),
    (f"{meta['n_leaves']}",        "Decision Leaves",      "#e05f2c"),
]
for col, (val, lbl, color) in zip([c1,c2,c3,c4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:{color}">
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Salary", "📊 Data Insights", "🌳 Model Details"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Enter Employee Details</p>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)

        age = st.slider("🎂 Age", min_value=18, max_value=60, value=28,
                        help="Employee's current age")

        education = st.selectbox("🎓 Education Level",
            ["High School", "Bachelor's", "Master's", "PhD"])

        department = st.selectbox("🏢 Department",
            sorted(meta["departments"]))

        experience = st.slider("💼 Experience (Years)",
            min_value=0, max_value=25, value=4,
            help="Total years of professional experience")

        st.markdown("---")
        st.markdown("**🛠️ Skills (select all that apply)**")

        all_skills = [
            "Python", "SQL", "Machine Learning", "Data Analysis",
            "Java", "JavaScript", "Cloud (AWS/GCP)", "Power BI",
            "Excel", "Communication", "Leadership", "Project Management"
        ]
        skill_cols = st.columns(2)
        selected_skills = []
        for i, skill in enumerate(all_skills):
            with skill_cols[i % 2]:
                if st.checkbox(skill, key=f"skill_{i}"):
                    selected_skills.append(skill)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        predict_btn = st.button("🔮 Predict Salary Category", use_container_width=True)

    with col_result:
        if predict_btn:
            edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
            edu_enc = edu_map[education]
            dept_encoded = dept_enc.transform([department])[0]
            num_skills = max(len(selected_skills), 1)

            input_df = pd.DataFrame([[age, edu_enc, experience, num_skills, dept_encoded]],
                                    columns=meta["features"])

            prediction  = model.predict(input_df)[0]
            proba       = model.predict_proba(input_df)[0]
            classes     = model.classes_
            prob_dict   = dict(zip(classes, proba))

            high_prob = prob_dict.get("High", 0)
            low_prob  = prob_dict.get("Low", 0)

            # Result card
            if prediction == "High":
                st.markdown(f"""
                <div class="result-high">
                    <div class="result-label">Prediction Result</div>
                    <div class="result-value">HIGH SALARY</div>
                    <div class="result-desc">This employee likely earns above <strong>$70,000/year</strong><br>
                    Confidence: <strong>{high_prob*100:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <div class="result-label">Prediction Result</div>
                    <div class="result-value">LOW SALARY</div>
                    <div class="result-desc">This employee likely earns below <strong>$70,000/year</strong><br>
                    Confidence: <strong>{low_prob*100:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=high_prob * 100,
                title={"text": "High Salary Probability", "font": {"size": 14, "family": "Sora"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0f9b58" if high_prob >= 0.5 else "#e05f2c"},
                    "steps": [
                        {"range": [0, 40],  "color": "#ffeaea"},
                        {"range": [40, 60], "color": "#fff8ea"},
                        {"range": [60, 100],"color": "#eafff4"},
                    ],
                    "threshold": {"line": {"color": "#4776e6", "width": 3}, "value": 50}
                }
            ))
            fig_gauge.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20),
                                     font={"family": "Sora"},
                                     paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Input summary
            st.markdown("**📋 Input Summary**")
            summary_data = {
                "Field": ["Age", "Education", "Department", "Experience", "Skills Count"],
                "Value": [age, education, department, f"{experience} years", num_skills]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

            if selected_skills:
                st.markdown("**Selected Skills:**")
                tags = " ".join([f'<span class="skill-tag">{s}</span>' for s in selected_skills])
                st.markdown(tags, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:white; border-radius:16px; padding:40px; text-align:center;
                        box-shadow:0 4px 20px rgba(0,0,0,0.06); min-height:400px;
                        display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <div style="font-size:3.5rem; margin-bottom:16px;">🎯</div>
                <div style="font-size:1.15rem; font-weight:600; color:#1a1a2e;">Ready to Predict</div>
                <div style="font-size:0.9rem; color:#888; margin-top:8px;">
                    Fill in employee details on the left<br>and click the Predict button
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Dataset Insights</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Salary distribution
        counts = df["Salary_Label"].value_counts()
        fig1 = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            hole=0.5,
            marker_colors=["#0f9b58", "#e05f2c"],
            textinfo="label+percent"
        ))
        fig1.update_layout(title="Salary Category Distribution",
                           font={"family": "Sora"}, paper_bgcolor="rgba(0,0,0,0)",
                           showlegend=True, height=320,
                           margin=dict(t=50, b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        # Education vs Salary
        edu_salary = df.groupby(["Education", "Salary_Label"]).size().unstack(fill_value=0)
        fig2 = go.Figure()
        colors = {"High": "#0f9b58", "Low": "#e05f2c"}
        edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
        edu_salary = edu_salary.reindex(edu_order)
        for label in ["High", "Low"]:
            if label in edu_salary.columns:
                fig2.add_trace(go.Bar(
                    name=label, x=edu_salary.index, y=edu_salary[label],
                    marker_color=colors[label]
                ))
        fig2.update_layout(barmode="group", title="Education Level vs Salary Category",
                           font={"family": "Sora"}, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=320,
                           margin=dict(t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        # Experience distribution
        fig3 = px.histogram(df, x="Experience_Years", color="Salary_Label",
                            color_discrete_map={"High": "#0f9b58", "Low": "#e05f2c"},
                            nbins=20, barmode="overlay", opacity=0.7,
                            title="Experience Distribution by Salary")
        fig3.update_layout(font={"family": "Sora"}, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        # Department salary
        dept_avg = df.groupby("Department")["Salary_USD"].mean().sort_values(ascending=True)
        fig4 = go.Figure(go.Bar(
            x=dept_avg.values, y=dept_avg.index, orientation="h",
            marker=dict(
                color=dept_avg.values,
                colorscale=[[0, "#e05f2c"], [0.5, "#f7971e"], [1, "#0f9b58"]],
                showscale=False
            )
        ))
        fig4.update_layout(title="Avg Salary by Department (USD)",
                           font={"family": "Sora"}, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=320,
                           margin=dict(t=50, b=10, l=120))
        st.plotly_chart(fig4, use_container_width=True)

    # Dataset preview
    st.markdown("---")
    st.markdown('<p class="section-header">Dataset Sample</p>', unsafe_allow_html=True)
    st.dataframe(
        df.drop(columns=["Education_Enc", "Department_Enc"], errors="ignore").head(20),
        use_container_width=True, height=350
    )
    st.caption(f"📦 Total Records: {len(df):,} | High Salary: {(df.Salary_Label=='High').sum()} | Low Salary: {(df.Salary_Label=='Low').sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL DETAILS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Decision Tree — Model Details</p>', unsafe_allow_html=True)

    col_x, col_y = st.columns(2)

    with col_x:
        # Feature Importance
        fi = meta["feature_importances"]
        fi_df = pd.DataFrame(fi.items(), columns=["Feature", "Importance"])\
                  .sort_values("Importance", ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, "#c3d5ff"], [1, "#4776e6"]],
                showscale=False
            ),
            text=[f"{v*100:.1f}%" for v in fi_df["Importance"]],
            textposition="outside"
        ))
        fig_fi.update_layout(
            title="Feature Importance", font={"family": "Sora"},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(t=50, b=10, l=120, r=60),
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0")
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_y:
        # Model Stats
        st.markdown("**🔢 Model Configuration**")
        stats = {
            "Algorithm":        "Decision Tree Classifier",
            "Criterion":        "Gini Impurity",
            "Max Depth":        meta["tree_depth"],
            "Number of Leaves": meta["n_leaves"],
            "Test Accuracy":    f"{meta['accuracy']}%",
            "5-Fold CV":        f"{meta['cv_accuracy']}% ± {meta['cv_std']}%",
            "Training Samples": "1,200",
            "Test Samples":     "300",
            "Classes":          ", ".join(meta["classes"]),
        }
        stats_df = pd.DataFrame(stats.items(), columns=["Parameter", "Value"])
        st.dataframe(stats_df, hide_index=True, use_container_width=True, height=340)

    # How Decision Tree Works
    st.markdown("---")
    st.markdown('<p class="section-header">How This Decision Tree Works</p>', unsafe_allow_html=True)

    steps = [
        ("🌱 Root Node", "Age", "The model first splits on Age. Older employees have a higher probability of falling in the High salary category."),
        ("📚 Level 2", "Education", "Next split on Education Level. Master's and PhD holders are routed towards High salary branches."),
        ("🏢 Level 3", "Department", "Department influences the prediction. Data Science & IT departments trend towards High salary."),
        ("🛠️ Level 4", "Skills", "Number of skills further refines the prediction. More skills → higher salary probability."),
        ("✅ Leaf Node", "Final Decision", "The leaf node assigns the majority class (High/Low) along with a confidence probability."),
    ]
    for icon_lbl, feature, desc in steps:
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:16px; padding:14px 0;
                    border-bottom:1px solid #f0f0f0;">
            <div style="background:#eef2ff; border-radius:10px; padding:10px 16px;
                        font-weight:700; color:#4776e6; white-space:nowrap; min-width:110px;
                        text-align:center; font-size:0.85rem;">{icon_lbl}</div>
            <div>
                <div style="font-weight:600; color:#1a1a2e;">{feature}</div>
                <div style="font-size:0.88rem; color:#666; margin-top:3px;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:20px; color:#aaa; font-size:0.82rem;">
    Built with 🌳 Scikit-learn Decision Tree &nbsp;|&nbsp;
    Streamlit &nbsp;|&nbsp; 1,500 Employee Records
</div>
""", unsafe_allow_html=True)
