import streamlit as st
import plotly.express as px
from src.data_loader import load_master


# Data

@st.cache_data
def get_data():
    return load_master()


df = get_data()

# Page config
st.set_page_config(
    page_title="Data to Harvest: AI for Agricultural Resilience Hackathon",
    page_icon=":ear_of_corn:",
    layout="wide",
)
st.title("Data to Harvest: AI for Agricultural Resilience Hackathon")
st.caption("DSU Hackathon, October 17-18, 2026 * One row = one county, one crop, one year")


# Sidebar filters
st.sidebar.header("Filters")

states = st.sidebar.multiselect(
    "State",
    options=sorted(df["state_alpha"].unique()),
    default=sorted(df["state_alpha"].unique()),
)

crops = st.sidebar.multiselect(
    "Crop",
    options=sorted(df["crop"].unique()),
    default=["CORN"],
)

year_min, year_max = st.sidebar.slider(
    "Year range",
    min_value=int(df["year"].min()),
    max_value=int(df["year"].max()),
    value=(2000, 2025),
)

# Apply filters
filtered = df[
    (df["state_alpha"].isin(states))
    & (df["crop"].isin(crops))
    & (df["year"].between(year_min, year_max))
]

st.sidebar.metric("Rows selected", f"{len(filtered):,}")

# Tabs
tab_overview, tab_weather, tab_drought, tab_soil = st.tabs(
    ["Overview", "Weather & Yield", "Drought Analysis", "Soil & Risk"]
)

# Tab 1: Overview
# This is a working demo. Use it as a pattern for the other features to develop

with tab_overview:
    st.subheader("Dataset overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Counties", filtered["fips"].nunique())
    col2.metric("Crops", filtered["crop"].nunique())
    col3.metric("Year span", f"{year_min}–{year_max}")

    # Yield time series — one working chart as the template
    st.subheader("Yield over time")

    reported = filtered[filtered["yield_status"] == "reported"]

    if len(reported) > 0:
        state_yield = (reported.groupby(["state_alpha", "crop", "year"], as_index=False).agg(mean_yield=("yield_per_acre", "mean")))

        fig = px.line(
            state_yield,
            x="year",
            y="mean_yield",
            color="state_alpha",
            facet_col="crop",
            labels={
                "mean_yield": "Mean yield (bu/acre)",
                "year": "Year",
                "state_alpha": "State",
            },
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No reported yields for this selection.")


# Tab 2: Weather & Yield

with tab_weather:
    st.subheader("Weather & Yield")
    st.write("Explore how precipitation, temperature, GDD, and extreme heat relate to crop yield.")

    # TODO: Your weather analysis here


# Tab 3: Drought Analysis

with tab_drought:
    st.subheader("Drought Relief — Data Preview")

    # Names of the columns we want to inspect.
    preview_columns = [
        "county_name",
        "state_alpha",
        "crop",
        "year",
        "yield_status",
        "yield_anomaly_pct",
        "max_drought_severity",
        "weeks_in_d2_plus",
        "aws_100cm_mm",
    ]

    # List the counties available under the current sidebar filters.
    county_options = (
        filtered[["fips", "county_name", "state_alpha"]]
        .drop_duplicates()
        .sort_values(["state_alpha", "county_name"])
    )

    if county_options.empty:
        st.info("No counties match the current sidebar filters.")

    else:
        # Display readable names while selecting by county code.
        county_labels = {
            row.fips: f"{row.county_name}, {row.state_alpha}"
            for row in county_options.itertuples(index=False)
        }

        selected_fips = st.selectbox(
            "Choose a county",
            options=county_options["fips"].tolist(),
            format_func=lambda fips: county_labels[fips],
        )

        # Select only this county's records.
        county_data = filtered[
            filtered["fips"] == selected_fips
        ].sort_values(["crop", "year"])

        # Summarize yield-reporting coverage for each selected crop.
        st.subheader("County Report Card — Data Coverage")

        for crop_name in sorted(county_data["crop"].unique()):
            crop_data = county_data[
                county_data["crop"] == crop_name
            ]

            # Keep years with a reported yield.
            reported_data = crop_data[
                crop_data["yield_status"] == "reported"
            ]

            reported_years = reported_data["year"].nunique()
            total_years = crop_data["year"].nunique()

            st.metric(
                label=f"{crop_name}: years with reported yield",
                value=f"{reported_years} / {total_years}",
            )

            # Only evaluate years where drought information is available.
            drought_known = reported_data[
                reported_data["max_drought_severity"].notna()
            ]

            eligible_years = drought_known["year"].nunique()

            drought_years = drought_known.loc[
                drought_known["max_drought_severity"] >= 3,
                "year",
            ].nunique()

            if eligible_years > 0:
                drought_frequency = (
                    drought_years / eligible_years
                ) * 100

                st.metric(
                    label=f"{crop_name}: D2+ drought years with reported yields",
                    value=f"{drought_years} / {eligible_years}",
                )

                st.caption(
                    f"D2+ drought occurred in {drought_frequency:.1f}% "
                    "of years with both reported yield and known drought data."
                )
            else:
                st.info(
                    f"{crop_name}: No years have both reported yield "
                    "and known drought data for this selection."
                )
            # D2+ years with a valid yield anomaly.
            drought_anomalies = drought_known[
                (drought_known["max_drought_severity"] >= 3)
                & drought_known["yield_anomaly_pct"].notna()
            ]
        
            anomaly_years = drought_anomalies["year"].nunique()
        
            if anomaly_years > 0:
                mean_anomaly = drought_anomalies[
                    "yield_anomaly_pct"
                ].mean()
        
                st.metric(
                    label=f"{crop_name}: average yield anomaly in D2+ years",
                    value=f"{mean_anomaly:+.1f}%",
                )
        
                st.caption(
                    f"Based on {anomaly_years} D2+ drought years "
                    "with reported yield and a valid yield anomaly. "
                    "Negative means below trend; positive means above trend. "
                    "This describes an association, not drought's causal effect."
                )
            else:
                st.info(
                    f"{crop_name}: No D2+ drought years have a valid "
                    "yield anomaly for this selection."
                )
            
            # Years below D2 with a valid yield anomaly.
            below_d2_anomalies = drought_known[
                (drought_known["max_drought_severity"] < 3)
                & drought_known["yield_anomaly_pct"].notna()
            ]
            
            below_d2_years = below_d2_anomalies["year"].nunique()
            
            if below_d2_years > 0:
                below_d2_mean = below_d2_anomalies[
                    "yield_anomaly_pct"
                ].mean()
            
                st.metric(
                    label=f"{crop_name}: average yield anomaly in below-D2 years",
                    value=f"{below_d2_mean:+.1f}%",
                )
            
                st.caption(
                    f"Based on {below_d2_years} years with reported yield, "
                    "known drought data, and a valid yield anomaly. "
                    "Below D2 includes no drought, D0, and D1."
                )
            else:
                st.info(
                    f"{crop_name}: No below-D2 years have a valid "
                    "yield anomaly for this selection."
                )

            # Compare only when both groups have usable data.
            if anomaly_years > 0 and below_d2_years > 0:
                anomaly_gap = mean_anomaly - below_d2_mean

                st.metric(
                    label=f"{crop_name}: D2+ minus below-D2 yield anomaly",
                    value=f"{anomaly_gap:+.1f} percentage points",
                )

                st.caption(
                    "A negative difference means yield was further below "
                    "trend in D2+ years than in below-D2 years. "
                    "This is a descriptive comparison, not a causal estimate."
                )

        st.caption(
            "Coverage uses the selected year range. "
            "Years without a reported yield are not counted as zero yield."
        )

        st.dataframe(
            county_data[preview_columns],
            width="stretch",
            hide_index=True,
        )
        
        # Plot drought duration against yield anomaly.
        st.subheader("Drought Duration & Yield Performance")
        
        chart_data = county_data[
            (county_data["yield_status"] == "reported")
            & county_data["weeks_in_d2_plus"].notna()
            & county_data["yield_anomaly_pct"].notna()
        ].copy()
        
        if chart_data.empty:
            st.info(
                "No years have both drought-duration data and a valid "
                "yield anomaly for this selection."
            )
        else:
            drought_fig = px.scatter(
                chart_data,
                x="weeks_in_d2_plus",
                y="yield_anomaly_pct",
                color="crop",
                facet_col="crop",
                facet_col_wrap=2,
                hover_data={
                    "year": True,
                    "weeks_in_d2_plus": ":.0f",
                    "yield_anomaly_pct": ":.1f",
                },
                labels={
                    "weeks_in_d2_plus": "Weeks in D2+ drought",
                    "yield_anomaly_pct": "Yield anomaly (%)",
                    "crop": "Crop",
                    "year": "Year",
                },
            )
        
            drought_fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
            )
        
            drought_fig.update_traces(
                marker=dict(size=10, opacity=0.8)
            )
        
            crop_count = chart_data["crop"].nunique()
        
            drought_fig.update_layout(
                height=450 if crop_count <= 2 else 750,
                showlegend=False,
            )
        
            st.plotly_chart(
                drought_fig,
                width="stretch",
            )

            st.caption(
                f"Showing {len(chart_data)} county–crop–year records. "
                "Only reported yields with known drought duration and "
                "a valid yield anomaly are included. "
                "The dashed line marks zero yield anomaly. "
                "Patterns are descriptive and do not establish causation."
            )

    st.caption(
        "Yield anomaly: negative values mean below-trend yield. "
        "Drought severity: 3 = D2 (severe), 4 = D3 (extreme), "
        "5 = D4 (exceptional). Soil water storage is measured in mm."
    )

# Tab 4: Soil & Risk

with tab_soil:
    st.subheader("Soil & Risk")
    st.write("Separate what the soil determines from what the weather determines.")
    # TODO: Your soil and risk analysis here