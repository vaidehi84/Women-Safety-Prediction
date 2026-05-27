import os
import pandas as pd
import streamlit as st
from utils.data_processing import load_crime_data, preprocess_data
from utils.model_utils import (
    load_model,
    save_model,
    build_training_pipeline,
    train_models,
    select_best_model,
    build_feature_options,
    create_prediction_record,
    calculate_safety_score,
    severity_category,
    get_hotspot_areas,
)
from utils.visualization import (
    render_trend_chart,
    render_area_heatmap,
    render_crime_distribution,
    render_severity_pie,
    render_gender_comparison,
)
from utils.helpers import get_emergency_contacts, get_women_safety_tips, get_women_safety_quotes, format_large_number

MODEL_PATH = os.path.join("models", "best_crime_model.joblib")
DATA_PATH = os.path.join("dataset", "sample_crime_data.csv")

st.set_page_config(
    page_title="Women Safety Crime Hotspot Prediction",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open(os.path.join("assets", "style.css"), "r", encoding="utf-8") as style_file:
    st.markdown(f"<style>{style_file.read()}</style>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    raw_data = load_crime_data(DATA_PATH)
    processed = preprocess_data(raw_data)
    return processed

@st.cache_resource(show_spinner=False)
def get_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)

    data = load_and_prepare_data()
    X, y, label_encoder, pipeline = build_training_pipeline(data)
    models = train_models(X, y)
    best_name, best_model, metrics = select_best_model(models, X, y)
    st.session_state["training_metrics"] = metrics
    trained = pipeline.set_params(classifier=best_model)
    trained.fit(X, y)
    artifact = {"pipeline": trained, "label_encoder": label_encoder}
    save_model(artifact, MODEL_PATH)
    return artifact

@st.cache_data
def get_dashboard_stats(data):
    total_crimes = int(data["cases_reported"].sum())
    unique_areas = data["area"].nunique()
    top_area = data.groupby("area")["cases_reported"].sum().idxmax()
    crime_types = data["crime_type"].nunique()
    return total_crimes, unique_areas, top_area, crime_types


def render_sidebar():
    st.sidebar.markdown("<div class='sidebar-header'>🚨 Women Safety AI</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-subtitle'>Crime Hotspot Predictor</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    menu = st.sidebar.radio(
        "Navigation",
        ["Home", "Prediction", "Dashboard", "Safety Tips", "Emergency Contacts", "About"],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div class='sidebar-info'>💡 Tip: Use the prediction page to generate safety scores for areas.</div>",
        unsafe_allow_html=True,
    )
    return menu


def render_home(data, stats):
    total_crimes, unique_areas, top_area, crime_types = stats

    st.markdown(
        """
        <div class='hero-section'>
            <div class='hero-glow'></div>
            <div class='hero-content'>
                <h1 class='hero-title'>Women Safety<br><span class='gradient-text'>Crime Hotspot Prediction</span></h1>
                <p class='hero-subtitle'>AI-powered analytics for safer communities. Predict risk, explore trends, and identify high-risk areas with confidence.</p>
                <div class='hero-cta'>
                    <span class='badge badge-primary'>🤖 AI Driven</span>
                    <span class='badge badge-success'>📊 Data Insights</span>
                    <span class='badge badge-danger'>🛡️ Women Safety</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat1, stat2, stat3, stat4 = st.columns(4, gap="medium")
    stat_cards = [
        ("Total Cases", format_large_number(total_crimes), "📈", "box-primary"),
        ("Areas Tracked", unique_areas, "📍", "box-success"),
        ("Top Risk Area", top_area, "⚠️", "box-danger"),
        ("Crime Categories", crime_types, "🔖", "box-warning"),
    ]

    for col, (title, value, icon, style) in zip((stat1, stat2, stat3, stat4), stat_cards):
        with col:
            st.markdown(
                f"""
                <div class='stat-box {style}'>
                    <div class='stat-icon'>{icon}</div>
                    <div class='stat-title'>{title}</div>
                    <div class='stat-number'>{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>📈 Crime Insights & Trends</h2>", unsafe_allow_html=True)

    col_trend, col_dist = st.columns((2, 1), gap="medium")
    with col_trend:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.plotly_chart(render_trend_chart(data), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_dist:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.plotly_chart(render_crime_distribution(data), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>🎯 Top Hotspot Areas</h2>", unsafe_allow_html=True)
    hotspot_data = get_hotspot_areas(data).head(10).reset_index().rename(columns={"cases_reported": "Total Cases"})
    st.dataframe(hotspot_data, use_container_width=True, hide_index=True)


def render_prediction(area_options, model, label_encoder, stats):
    total_crimes, unique_areas, top_area, crime_types = stats

    st.markdown("<h1 class='page-title'>🔍 Crime Risk Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Predict area risk and generate a women safety score with the trained model.</p>", unsafe_allow_html=True)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col_form, col_info = st.columns((2, 1), gap="large")
    with col_form:
        st.markdown("<div class='form-container'>", unsafe_allow_html=True)
        with st.form("prediction_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            state = col1.selectbox("State / UT", sorted(area_options["state"]))
            district = col1.selectbox("District / Area", sorted(area_options["area"]))
            crime_type = col2.selectbox("Crime Type", sorted(area_options["crime_type"]))
            victim_gender = col2.selectbox("Victim Gender", sorted(area_options["victim_gender"]))
            year = col1.selectbox("Year", sorted(area_options["year"]))
            month = col2.selectbox("Month", sorted(area_options["month"]))
            submit = st.form_submit_button("🔮 Predict Safety Risk")
        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        record = create_prediction_record(
            state=state,
            district=district,
            crime_type=crime_type,
            victim_gender=victim_gender,
            year=year,
            month=month,
        )
        prediction_idx = model.predict(record)[0]
        prediction = label_encoder.inverse_transform([prediction_idx])[0]
        probability = model.predict_proba(record).max()
        risk_score = calculate_safety_score(record, prediction, probability)
        category = severity_category(prediction)

        result_col1, result_col2, result_col3 = st.columns(3, gap="large")
        cards = [
            ("Predicted Severity", category, "result-success"),
            ("Safety Score", f"{risk_score}/100", "result-info"),
            ("Confidence", f"{probability:.1%}", "result-warning"),
        ]
        for col, (label, value, style) in zip((result_col1, result_col2, result_col3), cards):
            with col:
                st.markdown(
                    f"""
                    <div class='result-card {style}'>
                        <div class='result-label'>{label}</div>
                        <div class='result-value'>{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        report_data = {
            "State": [state],
            "Area": [district],
            "Crime Type": [crime_type],
            "Victim Gender": [victim_gender],
            "Year": [year],
            "Month": [month],
            "Predicted Severity": [category],
            "Confidence": [f"{probability:.1%}"],
            "Safety Score": [risk_score],
        }
        report_df = pd.DataFrame(report_data)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Report",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="women_safety_prediction_report.csv",
            mime="text/csv",
        )

    with col_info:
        st.markdown(
            f"""
            <div class='info-card info-blue'>
                <div class='info-title'>ℹ️ Model Status</div>
                <p class='info-text'>Training metrics are available internally.</p>
                <p class='info-text'>Top Risk Area: <b>{top_area}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='info-card info-danger'>
                <div class='info-title'>🎯 Focus</div>
                <p class='info-text'>Use the prediction engine for safer route planning and hotspot awareness.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dashboard(data, stats):
    total_crimes, unique_areas, top_area, crime_types = stats

    st.markdown("<h1 class='page-title'>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Explore crime patterns and hotspot analysis.</p>", unsafe_allow_html=True)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    kpi1, kpi2, kpi3 = st.columns(3, gap="medium")
    kpi1.metric("Total Incidents", format_large_number(total_crimes), delta="+12%")
    kpi2.metric("Coverage", f"{unique_areas} areas")
    kpi3.metric("Crime Types", crime_types)

    tab1, tab2 = st.tabs(["📈 Visualizations", "🗺️ Heatmap"])
    with tab1:
        col_pie, col_gender = st.columns(2, gap="medium")
        with col_pie:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(render_severity_pie(data), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_gender:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(render_gender_comparison(data), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    with tab2:
        fig = render_area_heatmap(data)
        st.pyplot(fig, use_container_width=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>🔝 Top Affected Areas</h2>", unsafe_allow_html=True)
    top_areas = get_hotspot_areas(data).head(12).reset_index().rename(columns={"cases_reported": "Cases"})
    st.dataframe(top_areas, use_container_width=True, hide_index=True)


def render_safety_tips():
    st.markdown("<h1 class='page-title'>💡 Women Safety Tips</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Smart, practical safety advice for everyday situations.</p>", unsafe_allow_html=True)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='quote-box'>
            <span class='quote-icon'>✨</span>
            <p class='quote-text'>"{get_women_safety_quotes()[0]}"</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    tips = get_women_safety_tips()
    emojis = ["👀", "📍", "🚗", "🎯", "🚨", "💪"]
    for idx, tip in enumerate(tips):
        col_icon, col_content = st.columns((0.15, 1), gap="small")
        with col_icon:
            st.markdown(f"<div class='tip-icon'>{emojis[idx]}</div>", unsafe_allow_html=True)
        with col_content:
            st.markdown(
                f"""
                <div class='tip-card'>
                    <h3 class='tip-title'>{tip['title']}</h3>
                    <p class='tip-text'>{tip['detail']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_emergency_contacts():
    st.markdown("<h1 class='page-title'>🚑 Emergency Contacts</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Quickly access helplines for urgent women safety support.</p>", unsafe_allow_html=True)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='alert-banner'>⚠️ In case of danger, use the helpline buttons to call immediately.</div>",
        unsafe_allow_html=True,
    )

    contacts = get_emergency_contacts()
    icons = {"Police Helpline": "🚔", "Women Helpline": "👩", "Ambulance Service": "🚑", "Cyber Crime Helpline": "💻", "Childline": "👶"}

    for label, details in contacts.items():
        icon = icons.get(label, "📞")
        st.markdown(
            f"""
            <div class='emergency-card'>
                <div class='emergency-header'>
                    <span class='emergency-icon'>{icon}</span>
                    <h3 class='emergency-title'>{label}</h3>
                </div>
                <p class='emergency-desc'>{details['description']}</p>
                <p class='emergency-number'>📞 {details['number']}</p>
                <a href='tel:{details['number']}' class='emergency-button'>Call Now</a>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about():
    st.markdown("<h1 class='page-title'>ℹ️ About</h1>", unsafe_allow_html=True)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='developer-card'>
            <div class='dev-header'>👨‍💻 Developer</div>
            <h3 class='dev-name'>Vaidehi Sharma</h3>
            <p class='dev-bio'>Undergraduate student | AI & ML enthusiast passionate about building intelligent solutions for women safety.</p>
            <p class='dev-bio'><b>Project:</b> Women Safety Crime Hotspot Prediction System - A machine learning application to predict crime patterns and empower safer communities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    menu = render_sidebar()
    data = load_and_prepare_data()
    artifact = get_model()
    model = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    stats = get_dashboard_stats(data)
    area_options = build_feature_options(data)

    if menu == "Home":
        render_home(data, stats)
    elif menu == "Prediction":
        render_prediction(area_options, model, label_encoder, stats)
    elif menu == "Dashboard":
        render_dashboard(data, stats)
    elif menu == "Safety Tips":
        render_safety_tips()
    elif menu == "Emergency Contacts":
        render_emergency_contacts()
    else:
        render_about()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error(f"An unexpected error occurred: {exc}")
