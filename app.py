"""Streamlit web application for Local Medicine Availability Predictor."""

from __future__ import annotations

import folium
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from model import (
    MEDICINE_OPTIONS,
    canonicalize_medicine_name,
    load_dataset,
    load_or_train_model,
)
from pharmacy_locator import (
    discover_nearby_pharmacies,
    enrich_pharmacies_with_inventory,
    resolve_location,
)
from predictor import (
    get_analytics_summary,
    get_availability_distribution,
    get_inventory_trend_series,
    predict_for_pharmacies,
    predict_stock_out_risk,
)

st.set_page_config(
    page_title="Medicine Availability Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #475569;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #e2f0ff 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 16px 30px rgba(15, 23, 42, 0.06);
    }
    .metric-label {
        color: #334155;
    }
    .metric-value {
        color: #0f172a;
    }
    .dashboard-panel {
        background: #ffffff;
        padding: 1.25rem 1.5rem;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.75rem;
    }
    .dashboard-panel .stMetric {
        margin-bottom: 0.75rem;
    }
    .stAlert {
        border-left: 4px solid #2563eb;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def get_inventory_data() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource
def get_model():
    return load_or_train_model()


def risk_class(level: str) -> str:
    return {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(level, "")


def build_map(user_lat: float, user_lon: float, ranked: pd.DataFrame) -> folium.Map:
    """Create interactive Folium map with user and pharmacy markers."""
    fmap = folium.Map(location=[user_lat, user_lon], zoom_start=14, tiles="OpenStreetMap")

    folium.Marker(
        [user_lat, user_lon],
        popup="Your Location",
        tooltip="You are here",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(fmap)

    for _, row in ranked.iterrows():
        prob = row["availability_probability"]
        color = "green" if prob >= 0.6 else "orange" if prob >= 0.4 else "red"
        popup_html = (
            f"<b>{row['pharmacy_name']}</b><br>"
            f"Availability: {prob:.0%}<br>"
            f"Risk: {row['stock_out_risk']}<br>"
            f"Distance: {row['distance_km']:.2f} km"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=10,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"#{int(row['rank'])} {row['pharmacy_name']}",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
        ).add_to(fmap)

    return fmap


def generate_recommendation_explanation(pharmacy_row: dict, ranked_df: pd.DataFrame) -> list[str]:
    """Generate human-readable explanations for why a pharmacy is recommended."""
    explanations = []

    availability = float(pharmacy_row.get("availability_probability", 0))
    if availability >= 0.7:
        explanations.append("✅ High historical availability (70%+)")
    elif availability >= 0.5:
        explanations.append("⚠️ Moderate availability (50-70%)")

    recency = float(pharmacy_row.get("report_recency_days", 30))
    if recency <= 7:
        explanations.append("🕐 Recently reported in stock (within 7 days)")
    elif recency <= 14:
        explanations.append("🕐 Relatively recent stock report (within 2 weeks)")

    distance = float(pharmacy_row.get("distance_km", 5))
    median_dist = ranked_df["distance_km"].median() if len(ranked_df) > 0 else 5
    if distance <= median_dist:
        explanations.append(f"📍 Close to your location ({distance:.1f} km)")

    risk = str(pharmacy_row.get("stock_out_risk", "Medium"))
    if risk == "Low":
        explanations.append("✓ Low stock-out risk")
    elif risk == "Medium":
        explanations.append("⚠️ Moderate stock-out risk")

    quantity = float(pharmacy_row.get("inventory_quantity", 0))
    if quantity >= 50:
        explanations.append("📦 Good inventory level on hand")
    elif quantity > 0:
        explanations.append("📦 Some inventory available")

    reliability = pharmacy_row.get("pharmacy_reliability", 0.5)
    if isinstance(reliability, (int, float)) and reliability >= 0.6:
        explanations.append("⭐ Consistent reporting history")

    return explanations if explanations else ["Pharmacy meets selection criteria"]


def create_availability_chart(dist_df: pd.DataFrame) -> go.Figure:
    """Create Plotly bar chart for availability distribution."""
    if dist_df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    dist_df.columns = ["Status", "Count"]
    colors = {"Available": "#10b981", "Not Available": "#ef4444"}
    fig = go.Figure(
        data=[
            go.Bar(
                x=dist_df["Status"],
                y=dist_df["Count"],
                marker=dict(
                    color=[colors.get(status, "#6b7280") for status in dist_df["Status"]],
                ),
                text=dist_df["Count"],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Medicine Availability Status",
        xaxis_title="Status",
        yaxis_title="Count",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        height=350,
    )
    return fig


def create_stock_out_risk_chart(ranked: pd.DataFrame) -> go.Figure:
    """Create Plotly pie chart for stock-out risk distribution."""
    if ranked.empty:
        return go.Figure().add_annotation(text="No data available")
    
    risk_counts = ranked["stock_out_risk"].value_counts().reset_index()
    risk_counts.columns = ["Risk", "Count"]
    
    colors = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
    fig = go.Figure(
        data=[
            go.Pie(
                labels=risk_counts["Risk"],
                values=risk_counts["Count"],
                marker=dict(
                    colors=[colors.get(risk, "#6b7280") for risk in risk_counts["Risk"]],
                ),
                textposition="inside",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Pharmacies: %{value}<br>Percentage: %{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Stock-Out Risk Distribution",
        margin=dict(l=40, r=40, t=40, b=40),
        height=350,
    )
    return fig


def create_inventory_trend_chart(trend_df: pd.DataFrame, pharmacy_name: str) -> go.Figure:
    """Create Plotly line chart for inventory trend."""
    if trend_df.empty:
        return go.Figure().add_annotation(text="No trend data available")
    
    trend_df = trend_df.copy()
    trend_df["last_reported"] = pd.to_datetime(trend_df["last_reported"])
    trend_df = trend_df.sort_values("last_reported")
    
    fig = go.Figure(
        data=[
            go.Scatter(
                x=trend_df["last_reported"],
                y=trend_df["quantity"],
                mode="lines+markers",
                name="Quantity",
                line=dict(color="#3b82f6", width=3),
                marker=dict(size=8),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Quantity: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"Inventory Trend: {pharmacy_name}",
        xaxis_title="Date",
        yaxis_title="Quantity",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        height=350,
    )
    return fig


def create_pharmacy_ranking_chart(ranked: pd.DataFrame) -> go.Figure:
    """Create Plotly scatter chart for pharmacy rankings."""
    if ranked.empty:
        return go.Figure().add_annotation(text="No data available")
    
    ranked_top = ranked.head(10).copy()
    
    fig = go.Figure(
        data=[
            go.Scatter(
                x=ranked_top["distance_km"],
                y=ranked_top["availability_probability"],
                mode="markers+text",
                marker=dict(
                    size=ranked_top["inventory_quantity"].clip(5, 50),
                    color=ranked_top["availability_probability"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Availability"),
                ),
                text=ranked_top["rank"].astype(int),
                textposition="top center",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Distance: %{x:.1f} km<br>"
                    "Availability: %{y:.0%}<br>"
                    "Rank: %{customdata[1]}<extra></extra>"
                ),
                customdata=ranked_top[["pharmacy_name", "rank"]],
            )
        ]
    )
    fig.update_layout(
        title="Top 10 Pharmacies: Distance vs Availability",
        xaxis_title="Distance (km)",
        yaxis_title="Availability Probability",
        hovermode="closest",
        margin=dict(l=40, r=40, t=40, b=40),
        height=350,
    )
    return fig


def show_reporting_form() -> None:
    """Display crowdsourced medicine reporting form in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Report Medicine Availability")
    st.sidebar.markdown("Help others by reporting stock availability at your local pharmacy.")
    
    with st.sidebar.form("report_form"):
        report_pharmacy = st.text_input(
            "Pharmacy Name",
            placeholder="e.g., Apollo Pharmacy, MedPlus",
            key="report_pharmacy",
        )
        report_medicine = st.selectbox(
            "Medicine Name",
            MEDICINE_OPTIONS,
            key="report_medicine",
        )
        report_status = st.radio(
            "Availability",
            ["Available", "Not Available"],
            horizontal=True,
            key="report_status",
        )
        report_quantity = st.number_input(
            "Quantity in Stock",
            min_value=0,
            max_value=500,
            value=0,
            step=1,
            key="report_quantity",
        )
        report_date = st.date_input(
            "Report Date",
            key="report_date",
        )
        
        report_submitted = st.form_submit_button(
            "Submit Report",
            type="secondary",
            use_container_width=True,
            key="report_submit",
        )
    
    if report_submitted:
        if not report_pharmacy or not report_medicine:
            st.sidebar.error("Please fill in all fields.")
            return
        
        from model import save_report

        success = save_report(
            pharmacy_name=report_pharmacy,
            medicine_name=report_medicine,
            availability_status=report_status,
            quantity=report_quantity,
            report_date=report_date.strftime("%Y-%m-%d"),
        )
        
        if success:
            st.sidebar.success("✅ Report submitted! Thank you for contributing.")
            # Clear cache to reload data
            st.cache_data.clear()
            st.rerun()
        else:
            st.sidebar.error("❌ Failed to submit report. Please try again.")


def main() -> None:
    """Main application entry point."""
    st.markdown('<p class="main-header">💊 Local Medicine Availability Predictor</p>', unsafe_allow_html=True)

    st.markdown(
        '<p class="sub-header">AI-powered predictions for medicine stock at nearby pharmacies</p>',
        unsafe_allow_html=True,
    )

    inventory_df = get_inventory_data()
    model = get_model()

    with st.sidebar:
        st.header("Search")
        st.markdown(
            "Select or type a medicine name. Common aliases are supported, including Dolo 650, Calpol, Cetzine, Glycomet, and Mox."
        )
        medicine_name = st.selectbox(
            "Medicine Name",
            MEDICINE_OPTIONS,
            index=0,
            key="search_medicine",
        )
        city = st.text_input(
            "City",
            value="New Delhi",
            help="Required. Supported: Chennai, Bangalore, Hyderabad, Mumbai, Delhi, Pune.",
            key="search_city",
        )
        address = st.text_input(
            "Address (optional)",
            value="Connaught Place",
            help="Full address supported (flat, apartment, street, area, city). Leave blank for city-wide search.",
            key="search_address",
        )
        radius_km = st.slider(
            "Search Radius (km)",
            min_value=1.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            key="search_radius",
        )
        st.info("Tip: leave Address blank to search around the city center.")
        with st.form("search_form"):
            search_clicked = st.form_submit_button(
                "Predict Availability",
                type="primary",
                use_container_width=True,
                key="search_submit",
            )
        
        show_reporting_form()

    if search_clicked:
        st.session_state["prediction_data"] = None
        st.session_state["prediction_error"] = None

    prediction_data = st.session_state.get("prediction_data")
    prediction_error = st.session_state.get("prediction_error")

    if not search_clicked and prediction_data is None:
        st.info("Enter medicine and location details in the sidebar, then click **Predict Availability**.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dataset Records", f"{len(inventory_df):,}")
        col2.metric("Medicines Tracked", str(inventory_df["medicine_name"].nunique()))
        col3.metric("Pharmacies in Data", str(inventory_df["pharmacy_name"].nunique()))
        avail_rate = (
            inventory_df["availability_status"].eq("Available").mean() * 100
        )
        col4.metric("Avg Historical Availability", f"{avail_rate:.1f}%")
        return

    if search_clicked or prediction_data is None:
        selected_medicine = canonicalize_medicine_name(medicine_name)
        if selected_medicine != medicine_name:
            st.caption(f"Medicine search normalized to: **{selected_medicine}**")

        try:
            with st.spinner("Geocoding your location..."):
                location = resolve_location(city, address or None)

            if not location.success or location.latitude is None or location.longitude is None:
                raise ValueError(location.error or "Could not geocode the provided location.")

            user_lat, user_lon = location.latitude, location.longitude
            if location.used_fallback:
                warning_message = location.message
            else:
                warning_message = None

            with st.spinner("Discovering nearby pharmacies..."):
                pharmacies = discover_nearby_pharmacies(user_lat, user_lon, inventory_df, radius_km=radius_km)

            if not pharmacies:
                raise ValueError("No pharmacies found nearby. Try increasing the search radius.")

            enriched = enrich_pharmacies_with_inventory(pharmacies, inventory_df, selected_medicine)

            with st.spinner("Running ML predictions..."):
                ranked = predict_for_pharmacies(enriched, selected_medicine, model=model, inventory_df=inventory_df)

            analytics = get_analytics_summary(ranked)
            top = ranked.iloc[0]

            st.session_state["prediction_data"] = {
                "selected_medicine": selected_medicine,
                "location": location,
                "user_lat": user_lat,
                "user_lon": user_lon,
                "warning_message": warning_message,
                "ranked": ranked,
                "analytics": analytics,
                "top": top,
            }
            prediction_data = st.session_state["prediction_data"]
        except Exception as exc:
            st.session_state["prediction_error"] = str(exc)
            prediction_error = st.session_state["prediction_error"]

    if prediction_error:
        st.error(prediction_error)
        if isinstance(prediction_data, dict) and prediction_data.get("location"):
            location = prediction_data["location"]
            if location.attempted_queries:
                with st.expander("Geocoding attempts"):
                    for attempt in location.attempted_queries:
                        st.write(f"- `{attempt}`")
        st.info(
            "Tips: use a supported city name, leave Address blank for city-only search, "
            "or try a well-known landmark."
        )
        return

    if prediction_data is None:
        return

    selected_medicine = prediction_data["selected_medicine"]
    location = prediction_data["location"]
    user_lat = prediction_data["user_lat"]
    user_lon = prediction_data["user_lon"]
    warning_message = prediction_data["warning_message"]
    ranked = prediction_data["ranked"]
    analytics = prediction_data["analytics"]
    top = prediction_data["top"]
    top_risk_level = str(top["stock_out_risk"])
    top_risk_prob = float(top["stock_out_probability"])

    if warning_message:
        st.warning(warning_message)
    else:
        st.success(location.message)

    st.caption(f"Geocoding query used: `{location.query_used}`")
    if location.display_name:
        st.caption(f"Matched location: {location.display_name}")
    st.caption(f"Coordinates: {user_lat:.4f}, {user_lon:.4f} · Match type: {location.match_type}")

    st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown("**Medicine**\n\n💊 " + selected_medicine)
    s2.markdown("**Location**\n\n📍 " + (location.display_name or location.query_used))
    s3.markdown("**Lookup type**\n\n🔎 " + location.match_type.replace("_", " ").title())
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
    st.subheader("Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🩺 Nearby Pharmacies", analytics["total_pharmacies"], delta=None)
    k2.metric("📈 Availability Probability", f"{top['availability_probability']:.0%}")
    k3.metric(
        "⚠️ Stock-out Risk",
        top_risk_level,
        delta=f"{top_risk_prob:.0%} probability",
        delta_color="inverse",
    )
    k4.metric("🏥 Recommended Pharmacy", analytics["top_pharmacy"][:28])
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Prediction Results")
    st.markdown(
        f"Predicted status at top pharmacy: **{top['predicted_status']}** "
        f"({top['availability_probability']:.0%} confidence)"
    )

    st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
    st.markdown("**Top Recommendation**")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Rank", f"#{int(top['rank'])}")
    r2.metric("Recommended Pharmacy", str(top['pharmacy_name'])[:28])
    r3.metric("Distance", f"{top['distance_km']:.2f} km")
    r4.metric("Availability", f"{top['availability_probability']:.0%}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
    st.markdown("**Why Recommended**")
    explanations = generate_recommendation_explanation(top, ranked)
    for explanation in explanations:
        st.markdown(f"- {explanation}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Analytics Dashboard")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Pharmacies", analytics["total_pharmacies"])
    a2.metric("Average Availability", f"{analytics['average_availability']:.0%}")
    a3.metric("High-Risk Pharmacies", analytics["high_risk_count"])
    a4.metric("Top Recommendation", f"#{int(top['rank'])}")

    # Analytics Charts Section
    st.subheader("📊 Analytics Visualizations")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        trend_df = get_inventory_trend_series(
            inventory_df, str(top["pharmacy_name"]), selected_medicine
        )
        fig_trend = create_inventory_trend_chart(trend_df, str(top["pharmacy_name"]))
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with chart_col2:
        dist_df = get_availability_distribution(inventory_df, selected_medicine)
        fig_availability = create_availability_chart(dist_df)
        st.plotly_chart(fig_availability, use_container_width=True)
    
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        fig_risk = create_stock_out_risk_chart(ranked)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with chart_col4:
        fig_ranking = create_pharmacy_ranking_chart(ranked)
        st.plotly_chart(fig_ranking, use_container_width=True)


    st.subheader("Ranked Nearby Pharmacies")
    display_cols = [
        "rank",
        "pharmacy_name",
        "distance_km",
        "availability_probability",
        "stock_out_risk",
    ]
    table = ranked[display_cols].copy()
    table = table.sort_values(
        by=["availability_probability", "distance_km"],
        ascending=[False, True],
    ).reset_index(drop=True)
    table["rank"] = table.index + 1
    table["availability_probability"] = table["availability_probability"].apply(lambda x: f"{x:.0%}")
    table["distance_km"] = table["distance_km"].round(2)
    table = table.rename(
        columns={
            "rank": "Rank",
            "pharmacy_name": "Pharmacy Name",
            "distance_km": "Distance (km)",
            "availability_probability": "Availability Probability",
            "stock_out_risk": "Stock-out Risk",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Interactive Map")
    fmap = build_map(user_lat, user_lon, ranked)
    st_folium(fmap, width=1200, height=500)

    with st.expander("Risk breakdown for top recommendation"):
        level, prob = predict_stock_out_risk(
            float(top["availability_probability"]),
            float(top["inventory_quantity"]),
            float(top["report_recency_days"]),
        )
        css = risk_class(level)
        st.markdown(
            f'<span class="{css}">Stock-Out Risk: {level} ({prob:.0%})</span>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
