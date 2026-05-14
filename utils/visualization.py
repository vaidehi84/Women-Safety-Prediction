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
        title="Yearly Crime Cases Trend",
        labels={"cases_reported": "Total Cases", "year": "Year"},
        template="plotly_white",
    )
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
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
        template="plotly_white",
    )
    fig.update_layout(xaxis_tickangle=-45, margin=dict(t=40, b=140))
    return fig


def render_area_heatmap(data: pd.DataFrame):
    pivot = (
        data.groupby(["area", "year"])["cases_reported"].sum().unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar_kws={"shrink": 0.75})
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
        template="plotly_white",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20))
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
        template="plotly_white",
    )
    fig.update_layout(margin=dict(t=40, b=80, l=20, r=20))
    return fig


def render_area_comparison(data: pd.DataFrame):
    area_cases = (
        data.groupby("area")["cases_reported"].sum().reset_index().sort_values("cases_reported", ascending=False).head(10)
    )
    fig = px.bar(
        area_cases,
        x="cases_reported",
        y="area",
        orientation="h",
        title="Top 10 Most Crime-Prone Areas",
        labels={"cases_reported": "Total Cases", "area": "Area"},
        template="plotly_white",
    )
    fig.update_layout(margin=dict(t=40, b=80, l=150, r=20))
    return fig


def render_severity_trend(data: pd.DataFrame):
    severity_trend = (
        data.groupby(["year", "severity_category"])["cases_reported"].sum().reset_index()
    )
    fig = px.line(
        severity_trend,
        x="year",
        y="cases_reported",
        color="severity_category",
        markers=True,
        title="Severity Trend by Year",
        labels={"cases_reported": "Total Cases", "year": "Year", "severity_category": "Severity"},
        template="plotly_white",
    )
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    return fig


def render_safest_vs_dangerous(data: pd.DataFrame):
    area_cases = data.groupby("area")["cases_reported"].sum().reset_index()
    top = area_cases.sort_values("cases_reported", ascending=False).head(5)
    bottom = area_cases.sort_values("cases_reported", ascending=True).head(5)
    top["group"] = "Most Dangerous"
    bottom["group"] = "Safest"
    combined = pd.concat([top, bottom], ignore_index=True)
    fig = px.bar(
        combined,
        x="cases_reported",
        y="area",
        color="group",
        orientation="h",
        title="Safest vs Dangerous Areas",
        labels={"cases_reported": "Total Cases", "area": "Area", "group": "Category"},
        template="plotly_white",
    )
    fig.update_layout(margin=dict(t=40, b=80, l=120, r=20))
    return fig
