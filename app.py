
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "churn_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
REPORT_PATH = ROOT / "reports" / "model_comparison.json"

st.set_page_config(
    page_title="Churn Intelligence | AI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

/* GLOBAL */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F4F7FB;
}

/* MAIN CONTENT TEXT */

[data-testid="stMain"] {
    color: #172B4D !important;
}

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #172B4D !important;
    font-weight: 750 !important;
}

[data-testid="stMain"] p,
[data-testid="stMain"] li {
    color: #334155;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background-color: #101B32 !important;
}

[data-testid="stSidebar"] * {
    color: #E5EDFA !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] hr {
    border-color: #34445E !important;
}

/* PAGE HEADER */

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #14213D !important;
    letter-spacing: -1px;
}

.subtitle {
    color: #64748B !important;
    font-size: 14px;
    margin-bottom: 22px;
}

/* KPI METRIC CARDS */

div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label *,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #172B4D !important;
}

div[data-testid="stMetric"] label {
    font-weight: 650 !important;
}

/* FORM */

[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 22px !important;
}

[data-testid="stForm"] h1,
[data-testid="stForm"] h2,
[data-testid="stForm"] h3,
[data-testid="stForm"] h4 {
    color: #172B4D !important;
}

/* INPUT LABELS */

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stForm"] label,
[data-testid="stForm"] label *,
[data-testid="stForm"] legend {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* TEXT AND NUMBER INPUTS */

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

/* SELECTBOX */

[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-color: #CBD5E1 !important;
    color: #172B4D !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] *,
[data-testid="stMultiSelect"] div[data-baseweb="select"] * {
    color: #172B4D !important;
}

/* DROPDOWN OPTIONS */

[role="listbox"],
[role="option"] {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
}

[role="option"]:hover {
    background-color: #EAF1FF !important;
}

/* SLIDER */

[data-testid="stSlider"] label,
[data-testid="stSlider"] p {
    color: #334155 !important;
}

/* BUTTONS */

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    border-radius: 9px !important;
    font-weight: 700 !important;
    min-height: 43px;
}

/* ALERTS */

[data-testid="stAlert"] p {
    color: #172B4D !important;
}

/* DATAFRAME */

[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

/* DIVIDER */

hr {
    border-color: #E2E8F0 !important;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #64748B !important;
    font-size: 12px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df


@st.cache_data
def load_metrics():
    if REPORT_PATH.exists():
        return pd.read_json(
            REPORT_PATH,
            orient="index"
        )

    return pd.DataFrame()


if not MODEL_PATH.exists():
    st.error("Model not found. Run: python src/train.py")
    st.stop()

if not DATA_PATH.exists():
    st.error(
        "Cleaned dataset not found. "
        "Check data/processed/telco_churn_cleaned.csv"
    )
    st.stop()

try:
    model = load_model()
    df = load_data()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("# 📊 CHURN AI")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")

    st.divider()

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Select Dashboard",
        [
            "Overview",
            "Predict Churn",
            "Model Performance",
            "Customer Insights"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### PROJECT INFORMATION")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Dataset:** IBM Telco Churn")
    st.write("**Framework:** Streamlit")
    st.write("**Language:** Python")

    st.divider()

    st.caption("AI-Powered Customer Retention")
    st.caption("Machine Learning Project")


# ============================================================
# 5. COMMON METRICS AND FUNCTIONS
# ============================================================

total_customers = len(df)

churned = (df["Churn"] == "Yes").sum()

retained = (df["Churn"] == "No").sum()

churn_rate = churned / total_customers * 100

avg_monthly = df["MonthlyCharges"].mean()


def page_header(title, description):

    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="subtitle">{description}</div>',
        unsafe_allow_html=True
    )


def chart_theme(fig, height=350):

    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

        font=dict(
            family="Inter",
            color="#334155",
            size=13
        ),

        title=dict(
            font=dict(
                color="#172B4D",
                size=17
            )
        ),

        margin=dict(
            l=25,
            r=25,
            t=65,
            b=40
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#334155")
        ),

        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_color="#172B4D",
            font_size=13
        )
    )

    fig.update_xaxes(
        showgrid=False,
        color="#334155",
        tickfont=dict(color="#334155"),
        title_font=dict(color="#172B4D")
    )

    fig.update_yaxes(
        gridcolor="#EDF1F7",
        color="#334155",
        tickfont=dict(color="#334155"),
        title_font=dict(color="#172B4D")
    )

    return fig


# ============================================================
# 6. PAGE ONE: OVERVIEW
# ============================================================

if page == "Overview":

    page_header(
        "Customer Analytics Overview",
        "Monitor customer behavior, retention and churn patterns."
    )

    # KPI CARDS - NATIVE STREAMLIT METRICS

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}",
            help="Total number of customers in the dataset."
        )

    with c2:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.1f}%",
            help="Percentage of customers who left."
        )

    with c3:
        st.metric(
            "Retained Customers",
            f"{retained:,}",
            help="Customers who stayed."
        )

    with c4:
        st.metric(
            "Average Monthly Bill",
            f"${avg_monthly:.2f}",
            help="Average monthly customer charges."
        )

    st.divider()

    st.subheader("📊 Churn Analytics")

    left, right = st.columns(2)

    # DONUT CHART

    with left:

        fig = px.pie(
            df,
            names="Churn",
            hole=0.65,
            title="Customer Churn Distribution",
            color="Churn",

            color_discrete_map={
                "Yes": "#F05B70",
                "No": "#20B997"
            }
        )

        fig.update_traces(
            textinfo="percent+label",
            marker=dict(
                line=dict(color="white", width=3)
            )
        )

        fig.add_annotation(
            text=f"{churn_rate:.1f}%<br>Churn",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=20,
                color="#172B4D"
            )
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True
        )

    # CONTRACT CHART

    with right:

        contract_data = (
            df.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="Customers")
        )

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Customers",
            color="Churn",
            barmode="group",
            title="Customers by Contract Type",

            color_discrete_map={
                "Yes": "#F05B70",
                "No": "#20B997"
            }
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True
        )

    left, right = st.columns(2)

    # TENURE HISTOGRAM

    with left:

        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            nbins=25,
            barmode="overlay",
            opacity=0.75,
            title="Customer Tenure Distribution",

            labels={
                "tenure": "Tenure (Months)"
            },

            color_discrete_map={
                "Yes": "#F05B70",
                "No": "#3978E8"
            }
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True
        )

    # MONTHLY CHARGES BOX PLOT

    with right:

        fig = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            title="Monthly Charges by Churn",

            color_discrete_map={
                "Yes": "#F05B70",
                "No": "#3978E8"
            }
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True
        )

    st.info(
        "Business Insight: Compare contract types, tenure and "
        "monthly charges to identify customer groups that may "
        "need retention support."
    )


# ============================================================
# 7. PAGE TWO: PREDICT CHURN
# ============================================================

elif page == "Predict Churn":

    page_header(
        "AI Customer Churn Prediction",
        "Enter customer details to estimate their probability of leaving."
    )

    st.info(
        "Complete the customer details below. "
        "The trained Logistic Regression pipeline will make the prediction."
    )

    with st.form("prediction_form"):

        st.subheader("1. Personal Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

        with c2:
            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

        with c3:
            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

        c1, c2 = st.columns(2)

        with c1:
            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

        with c2:
            tenure = st.slider(
                "Tenure (Months)",
                0,
                72,
                12
            )

        st.divider()

        st.subheader("2. Service Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

        with c2:
            security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"]
            )

            backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )

            protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )

        with c3:
            support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"]
            )

            tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )

            movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )

        st.divider()

        st.subheader("3. Account Information")

        c1, c2 = st.columns(2)

        with c1:
            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

        with c2:
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=1000.0,
                value=70.35
            )

        st.divider()

        st.subheader("4. Billing Information")

        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=844.20
        )

        submitted = st.form_submit_button(
            "🚀 Generate Churn Prediction",
            use_container_width=True,
            type="primary"
        )

    # PREDICTION

    if submitted:

        if tenure == 0 and total != 0:

            st.warning(
                "For zero-month tenure, check that Total Charges is zero."
            )

        elif tenure > 0 and total == 0:

            st.warning(
                "Please check Total Charges for a customer "
                "with existing tenure."
            )

        else:

            customer = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": support,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total
            }

            customer_df = pd.DataFrame([customer])

            try:

                prediction = int(
                    model.predict(customer_df)[0]
                )

                probability = float(
                    model.predict_proba(customer_df)[0, 1]
                )

                st.divider()

                st.subheader("📈 Prediction Results")

                c1, c2 = st.columns(2)

                with c1:

                    if prediction == 1:
                        st.error("⚠️ Customer predicted to churn")
                    else:
                        st.success("✅ Customer predicted to stay")

                    st.metric(
                        "Churn Probability",
                        f"{probability * 100:.2f}%"
                    )

                # Probability-based illustrative risk category

                if probability < 0.30:
                    risk = "Low"

                elif probability < 0.60:
                    risk = "Medium"

                else:
                    risk = "High"

                with c2:

                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,

                            number={
                                "suffix": "%",
                                "font": {
                                    "color": "#172B4D"
                                }
                            },

                            title={
                                "text": "Churn Risk Indicator"
                            },

                            gauge={
                                "axis": {
                                    "range": [0, 100]
                                },

                                "bar": {
                                    "color": "#3978E8"
                                },

                                "steps": [
                                    {
                                        "range": [0, 30],
                                        "color": "#D1FAE5"
                                    },
                                    {
                                        "range": [30, 60],
                                        "color": "#FEF3C7"
                                    },
                                    {
                                        "range": [60, 100],
                                        "color": "#FFE4E6"
                                    }
                                ]
                            }
                        )
                    )

                    st.plotly_chart(
                        chart_theme(gauge, 300),
                        use_container_width=True
                    )

                st.metric("Risk Category", risk)

                st.progress(probability)

                st.caption(
                    "Risk categories are illustrative probability bands. "
                    "The model's classification threshold is 0.5."
                )

                st.divider()

                st.subheader("💡 Retention Recommendations")

                if risk == "High":

                    st.error(
                        "Consider proactive customer outreach, "
                        "service reviews and appropriate retention offers."
                    )

                elif risk == "Medium":

                    st.warning(
                        "Monitor customer satisfaction and review "
                        "potential service or billing concerns."
                    )

                else:

                    st.success(
                        "Continue positive customer engagement "
                        "and maintain service quality."
                    )

                # DOWNLOAD PREDICTION

                report = customer.copy()

                report["Prediction"] = (
                    "Churn" if prediction == 1 else "No Churn"
                )

                report["Churn_Probability"] = round(
                    probability,
                    4
                )

                report["Risk_Level"] = risk

                st.download_button(
                    "⬇️ Download Prediction Report",

                    data=pd.DataFrame(
                        [report]
                    ).to_csv(index=False),

                    file_name="churn_prediction.csv",

                    mime="text/csv"
                )

            except Exception as error:

                st.error(f"Prediction failed: {error}")


# ============================================================
# 8. PAGE THREE: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    page_header(
        "Machine Learning Model Performance",
        "Compare the classification algorithms trained on the dataset."
    )

    if REPORT_PATH.exists():

        metrics = load_metrics()

        st.subheader("Model Comparison")

        st.dataframe(
            metrics.style.format("{:.3f}"),
            use_container_width=True
        )

        chart_data = (
            metrics.reset_index()
            .rename(columns={"index": "Model"})
            .melt(
                id_vars="Model",
                var_name="Metric",
                value_name="Score"
            )
        )

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Classification Model Comparison",

            color_discrete_sequence=[
                "#3978E8",
                "#F59E0B"
            ],

            text_auto=".3f"
        )

        fig.update_yaxes(
            range=[0, 1]
        )

        st.plotly_chart(
            chart_theme(fig, 450),
            use_container_width=True
        )

        st.info(
            "Logistic Regression achieved higher ROC-AUC in the "
            "current experiment, while Random Forest achieved "
            "higher recall. Model selection depends on business priorities."
        )

    else:

        st.warning(
            "Model comparison report not found."
        )


# ============================================================
# 9. PAGE FOUR: CUSTOMER INSIGHTS
# ============================================================

elif page == "Customer Insights":

    page_header(
        "Customer Insights",
        "Explore customer behavior, billing and service patterns."
    )

    st.subheader("Filter Customers")

    selected_contract = st.multiselect(
        "Select Contract Types",

        options=sorted(
            df["Contract"].dropna().unique()
        ),

        default=sorted(
            df["Contract"].dropna().unique()
        )
    )

    filtered = df[
        df["Contract"].isin(selected_contract)
    ]

    st.caption(
        f"Showing {len(filtered):,} customers"
    )

    c1, c2 = st.columns(2)

    # MONTHLY CHARGES

    with c1:

        fig = px.histogram(
            filtered,
            x="MonthlyCharges",
            color="Churn",
            nbins=30,
            title="Monthly Charges Distribution",

            color_discrete_map={
                "Yes": "#F05B70",
                "No": "#3978E8"
            }
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True
        )

    # TENURE

    with c2:

        fig = px.box(
            filtered,
            x="Churn",
            y="tenure",
            color="Churn",
            title="Tenure by Churn Status",

            color_discrete_map={
                "Yes": "#F05B70",
                "No": "#3978E8"
            }
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True
        )

    c1, c2 = st.columns(2)

    # INTERNET SERVICE CHURN

    with c1:

        service_churn = (
            filtered.groupby("InternetService")["Churn"]
            .apply(
                lambda x: (x == "Yes").mean() * 100
            )
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            service_churn,
            x="InternetService",
            y="Churn Rate",
            title="Churn Rate by Internet Service",
            color="Churn Rate",
            color_continuous_scale="Blues",
            text_auto=".1f"
        )

        st.plotly_chart(
            chart_theme(fig),
            use_container_width=True
        )

    # PAYMENT METHOD CHURN

    with c2:

        payment_churn = (
            filtered.groupby("PaymentMethod")["Churn"]
            .apply(
                lambda x: (x == "Yes").mean() * 100
            )
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            payment_churn,
            x="PaymentMethod",
            y="Churn Rate",
            title="Churn Rate by Payment Method",
            color="Churn Rate",
            color_continuous_scale="Teal",
            text_auto=".1f"
        )

        fig.update_xaxes(
            tickangle=-20
        )

        st.plotly_chart(
            chart_theme(fig, 400),
            use_container_width=True
        )

    # CUSTOMER TABLE

    st.subheader("Customer Data Preview")

    st.dataframe(
        filtered.head(100),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Filtered Customer Data",

        data=filtered.to_csv(index=False),

        file_name="filtered_customer_data.csv",

        mime="text/csv"
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        CHURN INTELLIGENCE
        <br>
        Built with Python | Streamlit | Scikit-learn | Plotly
        <br><br>
        Customer Churn Prediction Project
    </div>
    """,
    unsafe_allow_html=True
)
