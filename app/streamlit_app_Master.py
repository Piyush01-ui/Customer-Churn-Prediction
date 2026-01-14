import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

# ====================================================
# PAGE CONFIG
# ====================================================
st.set_page_config(
    page_title="AI Customer Churn Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# LOAD ARTIFACTS (CACHED)
# ====================================================
@st.cache_resource
def load_model():
    return joblib.load(
        "C:/Users/joshi/Desktop/AI_Made_Projects/Customer_Churn_New/model/churn_model.pkl"
    )

@st.cache_resource
def load_scaler():
    return joblib.load(
        "C:/Users/joshi/Desktop/AI_Made_Projects/Customer_Churn_New/model/scaler.pkl"
    )

@st.cache_resource
def load_features():
    return joblib.load(
        "C:/Users/joshi/Desktop/AI_Made_Projects/Customer_Churn_New/model/feature_columns.pkl"
    )

@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

model = load_model()
scaler = load_scaler()
feature_columns = load_features()
explainer = load_shap_explainer(model)

# ====================================================
# HEADER
# ====================================================
st.title("🚀 AI Customer Churn Intelligence Platform")
st.markdown(
    "Advanced churn analytics for **SaaS, subscription & growth teams** — "
    "predict churn, understand *why*, and take action."
)

# ====================================================
# SIDEBAR
# ====================================================
st.sidebar.header("⚙️ Controls")

risk_threshold = st.sidebar.slider(
    "High-Risk Threshold",
    0.4, 0.9, 0.7, 0.05
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Dataset (CSV)",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 This system combines **predictive ML + explainable AI (SHAP)** "
    "to support revenue retention decisions."
)

# ====================================================
# MAIN APP
# ====================================================
if uploaded_file:
    raw = pd.read_csv(uploaded_file)

    st.subheader("🧾 Dataset Preview")
    st.dataframe(raw.head(), use_container_width=True)

    # ====================================================
    # PREPROCESSING (MATCH TRAINING)
    # ====================================================
    df = raw.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["contract_risk_flag"] = (df["Contract"] == "Month-to-month").astype(int)

    df_model = df.drop(columns=["Churn"], errors="ignore")
    df_model = pd.get_dummies(df_model)
    df_model = df_model.reindex(columns=feature_columns, fill_value=0)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]
    df_model[num_cols] = scaler.transform(df_model[num_cols])

    # ====================================================
    # PREDICTION
    # ====================================================
    churn_probs = model.predict_proba(df_model)[:, 1]
    df["Churn Probability"] = churn_probs

    df["Risk Segment"] = pd.cut(
        churn_probs,
        bins=[0, 0.3, risk_threshold, 1],
        labels=["Low", "Medium", "High"]
    )

    # ====================================================
    # EXECUTIVE KPI DASHBOARD
    # ====================================================
    st.markdown("## 📊 Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", len(df))
    c2.metric("Avg Churn Risk", f"{churn_probs.mean():.2%}")
    c3.metric("High-Risk Customers", (churn_probs > risk_threshold).sum())
    c4.metric("Projected Churn Rate", f"{(churn_probs > risk_threshold).mean():.2%}")

    # ====================================================
    # DISTRIBUTION (INTERACTIVE)
    # ====================================================
    st.markdown("## 📈 Churn Probability Distribution")

    fig_dist = px.histogram(
        df,
        x="Churn Probability",
        color="Risk Segment",
        nbins=40,
        opacity=0.75,
        title="Customer Churn Risk Distribution"
    )

    fig_dist.add_vline(
        x=risk_threshold,
        line_dash="dash",
        annotation_text="High Risk Threshold"
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    # ====================================================
    # SEGMENT PIE
    # ====================================================
    st.markdown("## 🧩 Customer Risk Segmentation")

    fig_pie = px.pie(
        df,
        names="Risk Segment",
        hole=0.4,
        title="Risk Segment Breakdown"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # ====================================================
    # FILTER PANEL
    # ====================================================
    st.markdown("## 🎯 Customer Explorer")

    colA, colB, colC = st.columns(3)

    risk_filter = colA.multiselect(
        "Risk Segment",
        df["Risk Segment"].dropna().unique(),
        default=["High"]
    )

    tenure_filter = colB.slider(
        "Tenure (months)",
        int(df["tenure"].min()),
        int(df["tenure"].max()),
        (0, 24)
    )

    charge_filter = colC.slider(
        "Monthly Charges",
        float(df["MonthlyCharges"].min()),
        float(df["MonthlyCharges"].max()),
        (
            float(df["MonthlyCharges"].min()),
            float(df["MonthlyCharges"].max())
        )
    )

    filtered = df[
        (df["Risk Segment"].isin(risk_filter)) &
        (df["tenure"].between(*tenure_filter)) &
        (df["MonthlyCharges"].between(*charge_filter))
    ]

    # ====================================================
    # SCATTER ANALYSIS
    # ====================================================
    st.markdown("## 🔍 Churn Drivers: Who & Why")

    fig_scatter = px.scatter(
        filtered,
        x="tenure",
        y="MonthlyCharges",
        size="Churn Probability",
        color="Risk Segment",
        hover_data=[
            "Contract",
            "InternetService",
            "PaymentMethod",
            "Churn Probability"
        ],
        title="Tenure vs Monthly Charges (Churn Risk)"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # ====================================================
    # CUSTOMER TABLE
    # ====================================================
    st.markdown("## 🚨 High-Risk Customer List")

    st.dataframe(
        filtered.sort_values("Churn Probability", ascending=False),
        use_container_width=True
    )

    # ====================================================
    # CUSTOMER-LEVEL EXPLAINABILITY (SHAP)
    # ====================================================
    st.markdown("## 🧠 Customer-Level Churn Explanation (Explainable AI)")

    if len(filtered) > 0:
        selected_idx = st.selectbox(
            "Select a customer for deep analysis",
            filtered.index.tolist()
        )

        customer_raw = df.loc[selected_idx]
        customer_row = df_model.loc[[selected_idx]]

        shap_values = explainer(customer_row)

        if len(shap_values.values.shape) == 3:
            values = shap_values.values[0, :, 1]
            base_value = shap_values.base_values[0, 1]
        else:
            values = shap_values.values[0]
            base_value = shap_values.base_values[0]

        fig, ax = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(
                values=values,
                base_values=base_value,
                data=customer_row.iloc[0],
                feature_names=customer_row.columns
            ),
            max_display=8,
            show=False
        )

        st.pyplot(fig)

        shap_df = pd.DataFrame({
            "Feature": customer_row.columns,
            "Impact": values
        }).sort_values("Impact", ascending=False)

        st.markdown("### 🔥 Top Churn Drivers")
        st.dataframe(shap_df.head(5), use_container_width=True)

        # ====================================================
        # RECOMMENDATIONS
        # ====================================================
        st.markdown("### 🎯 Retention Recommendations")

        actions = []

        if "contract_risk_flag" in shap_df.head(5)["Feature"].values:
            actions.append("Offer long-term contract incentives")

        if "MonthlyCharges" in shap_df.head(5)["Feature"].values:
            actions.append("Provide pricing optimization / downgrade option")

        if "tenure" in shap_df.head(5)["Feature"].values:
            actions.append("Launch loyalty or onboarding engagement")

        for a in actions:
            st.success(a)

        # ====================================================
        # LTV IMPACT
        # ====================================================
        st.markdown("### 💰 Revenue Impact Estimation")

        annual_ltv = customer_raw["MonthlyCharges"] * 12
        revenue_risk = annual_ltv * customer_raw["Churn Probability"]

        st.metric(
            "Estimated Annual Revenue at Risk",
            f"₹ {revenue_risk:,.0f}"
        )

    # ====================================================
    # AUTO INSIGHTS
    # ====================================================
    st.markdown("## 🧠 AI-Generated Insights")

    if len(filtered) > 0:
        st.info(f"""
• {len(filtered)} customers fall into your selected risk window  
• Avg tenure: {filtered['tenure'].mean():.1f} months  
• Avg monthly charge: ₹{filtered['MonthlyCharges'].mean():.0f}  
• Majority are **month-to-month contracts**

👉 **Strategic Recommendation:**  
Run targeted retention campaigns before next billing cycle.
""")
    else:
        st.info("No customers match current filters.")

else:
    st.info("📂 Upload a CSV file to start churn intelligence analysis.")