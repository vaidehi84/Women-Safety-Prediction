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
from utils.helpers import get_emergency_contacts, get_women_safety_tips, format_large_number

MODEL_PATH = os.path.join("models", "best_crime_model.joblib")
DATA_PATH = os.path.join("dataset", "crime_in_india.csv")

st.set_page_config(
    page_title="Women Safety Crime Hotspot Prediction",
    page_icon="🚨",
    layout="wide",
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


def main():
    st.sidebar.title("Women Safety Crime Hotspot Prediction")
    menu = st.sidebar.radio(
        "Navigation",
        ["Home", "Prediction", "Dashboard", "Safety Tips", "About"],
    )

    data = load_and_prepare_data()
    artifact = get_model()
    model = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    total_crimes, unique_areas, top_area, crime_types = get_dashboard_stats(data)
    area_options = build_feature_options(data)

    if menu == "Home":
        st.markdown("## Women Safety Crime Hotspot Prediction System")
        st.markdown(
            "Use intelligent analytics and machine learning to identify crime-prone areas, predict women safety risk, and visualize hotspots across Indian districts."
        )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Crime Cases", format_large_number(total_crimes))
        kpi2.metric("Tracked Areas", unique_areas)
        kpi3.metric("Most Affected Area", top_area)
        kpi4.metric("Crime Types", crime_types)

        st.markdown("### Key insights")
        st.write(
            "The dashboard presents area-wise safety insights, trend analytics, and a prediction engine to help women identify safer neighborhoods."
        )

        trend, dist = st.columns((2, 1))
        with trend:
            st.subheader("Yearly Crime Trend")
            st.plotly_chart(render_trend_chart(data), use_container_width=True)

        with dist:
            st.subheader("Crime Category Distribution")
            st.plotly_chart(render_crime_distribution(data), use_container_width=True)

        st.subheader("Top Hotspots")
        st.dataframe(
            get_hotspot_areas(data).head(10).reset_index().rename(
                columns={"cases_reported": "total_cases"}
            ),
            use_container_width=True,
        )

    elif menu == "Prediction":
        st.markdown("## Area-wise Crime Risk Prediction")
        st.write("Enter the details below to predict the crime severity and women safety score for a selected area.")

        with st.form("prediction_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            state = col1.selectbox("State / UT", sorted(area_options["state"]), index=0)
            district = col1.selectbox("District / Area", sorted(area_options["area"]), index=0)
            crime_type = col2.selectbox("Crime Type", sorted(area_options["crime_type"]), index=0)
            victim_gender = col2.selectbox("Victim Gender", sorted(area_options["victim_gender"]), index=0)
            year = col1.selectbox("Year", sorted(area_options["year"]), index=0)
            month = col2.selectbox("Month", sorted(area_options["month"]), index=0)
            submitted = st.form_submit_button("Predict Safety Risk")

        if submitted:
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

            st.success(f"Predicted crime severity: {category}")
            st.write(f"Model confidence: {probability:.1%}")
            st.write(f"Women safety score: {risk_score}/100")

            report = {
                "State": state,
                "Area": district,
                "Crime Type": crime_type,
                "Victim Gender": victim_gender,
                "Year": year,
                "Month": month,
                "Predicted Severity": category,
                "Confidence": f"{probability:.1%}",
                "Safety Score": risk_score,
            }
            download_df = pd.DataFrame([report])
            st.download_button(
                label="Download Prediction Report",
                data=download_df.to_csv(index=False).encode("utf-8"),
                file_name="women_safety_prediction_report.csv",
                mime="text/csv",
            )

    elif menu == "Dashboard":
        st.markdown("## Crime Analytics Dashboard")
        top_hotspots = get_hotspot_areas(data).head(12)

        st.write("Analyze the most impacted locations, crime proportions by severity, and monthly data patterns.")
        st.pyplot(render_area_heatmap(data), use_container_width=True)

        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.subheader("Severity Distribution")
            st.plotly_chart(render_severity_pie(data), use_container_width=True)
        with metrics_col2:
            st.subheader("Gender Comparison")
            st.plotly_chart(render_gender_comparison(data), use_container_width=True)

        st.subheader("Top Hotspot Areas")
        st.table(
            top_hotspots.reset_index().rename(
                columns={"cases_reported": "Total Cases"}
            )
        )

    elif menu == "Safety Tips":
        st.markdown("## Women Safety Tips & Emergency Support")
        st.write(
            "This section provides practical, easy-to-follow safety advice and emergency resources for women."
        )
        for tip in get_women_safety_tips():
            st.markdown(f"- {tip}")

        st.subheader("Emergency Helpline Numbers")
        contacts = get_emergency_contacts()
        for name, number in contacts.items():
            st.markdown(f"**{name}:** {number}")

    else:
        st.markdown("## About Project")
        st.write(
            "A full-stack AI/ML solution built for placement-ready portfolios with crime hotspot prediction, safety scoring, and visual analytics."
        )
        st.write(
            "The system is designed to support decision-making for women safety by combining historical crime data with predictive analytics and interactive reports."
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error(f"An unexpected error occurred: {exc}")
