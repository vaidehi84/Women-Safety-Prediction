import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


def render_trend_chart(data: pd.DataFrame):
    trend = data.groupby("year")["cases_reported"].sum().reset_index()
    fig = px.line(
        trend,
        x="year",
        y="cases_reported",
        markers=True,
        title="Yearly Reported Crime Trend",
        labels={"cases_reported": "Total Cases", "year": "Year"},
    )
    fig.update_layout(template="plotly_white", margin=dict(t=40, b=20, l=20, r=20))
    return fig


def render_crime_distribution(data: pd.DataFrame):
    distribution = data["crime_type"].value_counts().reset_index()
    distribution.columns = ["Crime Type", "Count"]
    fig = px.bar(
        distribution,
        x="Crime Type",
        y="Count",
        title="Crime Type Distribution",
        labels={"Count": "Number of Cases", "Crime Type": "Crime Category"},
    )
    fig.update_layout(xaxis_tickangle=-45, template="plotly_white", margin=dict(t=40, b=100))
    return fig


def render_area_heatmap(data: pd.DataFrame):
    pivot = (
        data.groupby(["area", "year"])["cases_reported"].sum().unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title("Heatmap of Crime Cases by Area and Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Area")
    fig.tight_layout()
    return fig


def render_severity_pie(data: pd.DataFrame):
    severity = data["severity_category"].value_counts().reset_index()
    severity.columns = ["Severity", "Count"]
    fig = px.pie(
        severity,
        names="Severity",
        values="Count",
        title="Crime Severity Breakdown",
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_white", margin=dict(t=40, b=20, l=20, r=20))
    return fig


def render_gender_comparison(data: pd.DataFrame):
    gender_comparison = data["victim_gender"].value_counts().reset_index()
    gender_comparison.columns = ["Gender", "Count"]
    fig = px.bar(
        gender_comparison,
        x="Gender",
        y="Count",
        title="Victim Gender Comparison",
        labels={"Count": "Number of Cases", "Gender": "Victim Gender"},
    )
    fig.update_layout(template="plotly_white", margin=dict(t=40, b=80, l=20, r=20))
    return fig
