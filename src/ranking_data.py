import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.data_loader import load_master


FEATURE_COLUMNS = [
    "precip_mm",
    "tavg_c",
    "extreme_heat_days",
    "gdd",
    "max_drought_severity",
    "weeks_in_d2_plus",
    "aws_100cm_mm",
    "nccpi_crop",
]


def prepare_ranking_data(crop="CORN"):
    """Separate development data from final testing data."""
    master = load_master()

    crop_data = master.loc[
        (master["crop"] == crop)
        & master["year"].between(2000, 2025)
    ].copy()

    if crop_data.empty:
        raise ValueError(f"No data found for crop: {crop}")

    # Remove the anomaly calculated using the full 2000–2025 period.
    crop_data = crop_data.drop(
        columns=["yield_anomaly_pct"],
        errors="ignore",
    )

    crop_data["has_reported_yield"] = (
        crop_data["yield_status"].eq("reported")
        & crop_data["yield_per_acre"].notna()
    )

    train_data = crop_data.loc[
        crop_data["year"].between(2000, 2015)
    ].copy()

    test_data = crop_data.loc[
        crop_data["year"].between(2016, 2025)
    ].copy()

    if train_data.empty or test_data.empty:
        raise ValueError(
            "Both the training and testing periods must contain records."
        )

    if crop_data.duplicated(["fips", "crop", "year"]).any():
        raise ValueError(
            "Duplicate county–crop–year records found. "
            "Resolve these before modeling."
        )

    return train_data, test_data


def add_training_only_trends(train_data, test_data, min_years=10):
    """Fit trends on training yields, then apply to both periods."""
    train_data = train_data.copy()
    test_data = test_data.copy()

    fitted_trends = []

    for (fips, crop), group in train_data.groupby(["fips", "crop"]):
        reported = group.loc[group["has_reported_yield"]]
        reported_years = reported["year"].nunique()

        slope = np.nan
        intercept = np.nan

        if reported_years >= min_years:
            slope, intercept = np.polyfit(
                reported["year"].to_numpy(dtype=float) - 2000,
                reported["yield_per_acre"].to_numpy(dtype=float),
                deg=1,
            )

        fitted_trends.append({
            "fips": fips,
            "crop": crop,
            "training_yield_years": reported_years,
            "trend_slope": slope,
            "trend_intercept": intercept,
        })

    trend_table = pd.DataFrame(fitted_trends)

    def apply_trends(data):
        data = data.merge(
            trend_table,
            on=["fips", "crop"],
            how="left",
            validate="many_to_one",
        )

        data["trend_eligible"] = data["trend_slope"].notna()

        data["expected_yield_train_trend"] = (
            data["trend_intercept"]
            + data["trend_slope"] * (data["year"] - 2000)
        )

        usable = (
            data["has_reported_yield"]
            & data["trend_eligible"]
            & (data["expected_yield_train_trend"] > 0)
        )

        data["yield_anomaly_train_pct"] = np.nan

        data.loc[usable, "yield_anomaly_train_pct"] = (
            (
                data.loc[usable, "yield_per_acre"]
                / data.loc[usable, "expected_yield_train_trend"]
            ) - 1
        ) * 100

        data["yield_loss_pct"] = (
            -data["yield_anomaly_train_pct"]
        ).clip(lower=0)

        return data

    return apply_trends(train_data), apply_trends(test_data)


def print_summary(data, label):
    reported = data.loc[data["has_reported_yield"]]

    print(f"\n{label}")
    print(f"Year range: {data['year'].min()}–{data['year'].max()}")
    print(f"Total rows: {len(data):,}")
    print(f"Counties: {data['fips'].nunique():,}")
    print(f"Rows with reported yield: {len(reported):,}")
    print(
        "Counties with at least one reported yield: "
        f"{reported['fips'].nunique():,}"
    )


if __name__ == "__main__":
    # Full development and reserved testing periods.
    train_data, test_data = prepare_ranking_data(crop="CORN")

    train_data, test_data = add_training_only_trends(
        train_data,
        test_data,
        min_years=10,
    )

    print_summary(train_data, "TRAINING DATA")
    print_summary(test_data, "TESTING DATA")

    for label, data in [
        ("TRAINING TREND COVERAGE", train_data),
        ("TESTING TREND COVERAGE", test_data),
    ]:
        valid = data.loc[
            data["yield_anomaly_train_pct"].notna()
        ]

        eligible_counties = data.loc[
            data["trend_eligible"], "fips"
        ].nunique()

        print(f"\n{label}")
        print(f"Counties eligible for a trend: {eligible_counties:,}")
        print(f"Rows with valid new anomalies: {len(valid):,}")
        print(
            "Counties with valid new anomalies: "
            f"{valid['fips'].nunique():,}"
        )

    # Fresh data for internal model fitting and validation.
    development_data, _ = prepare_ranking_data(crop="CORN")

    fit_data = development_data.loc[
        development_data["year"].between(2000, 2010)
    ].copy()

    validation_data = development_data.loc[
        development_data["year"].between(2011, 2015)
    ].copy()

    fit_data, validation_data = add_training_only_trends(
        fit_data,
        validation_data,
        min_years=10,
    )

    for label, data in [
        ("MODEL FIT COVERAGE", fit_data),
        ("VALIDATION COVERAGE", validation_data),
    ]:
        usable = data.loc[data["yield_loss_pct"].notna()]

        print(f"\n{label}")
        print(
            f"Year range: "
            f"{data['year'].min()}–{data['year'].max()}"
        )
        print(f"Rows with valid loss outcomes: {len(usable):,}")
        print(
            "Counties with valid loss outcomes: "
            f"{usable['fips'].nunique():,}"
        )

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in fit_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required input columns are missing: {missing_columns}"
        )

    model_fit_rows = fit_data.loc[
        fit_data["yield_loss_pct"].notna()
    ].copy()

    print("\nMODEL INPUT CHECK")
    print(f"Number of inputs: {len(FEATURE_COLUMNS)}")
    print(f"Fitting rows: {len(model_fit_rows):,}")
    print("Missing values per input:")
    print(
        model_fit_rows[FEATURE_COLUMNS]
        .isna()
        .sum()
        .to_string()
    )

    # Evaluate only records with a known loss outcome.
    model_validation_rows = validation_data.loc[
        validation_data["yield_loss_pct"].notna()
    ].copy()

    if model_fit_rows.empty or model_validation_rows.empty:
        raise ValueError(
            "Fitting and validation both need usable loss outcomes."
        )

    X_fit = model_fit_rows[FEATURE_COLUMNS]
    y_fit = model_fit_rows["yield_loss_pct"]

    X_validation = model_validation_rows[FEATURE_COLUMNS]
    y_validation = model_validation_rows["yield_loss_pct"]

    # Fit preprocessing and the model using fitting data only.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(X_fit, y_fit)

    validation_predictions = model.predict(X_validation)

    model_validation_rows["predicted_loss_pct"] = (
        validation_predictions
    )

    # Constant-median prediction-error baseline.
    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_fit, y_fit)

    baseline_predictions = baseline.predict(X_validation)

    model_mae = mean_absolute_error(
        y_validation,
        validation_predictions,
    )

    baseline_mae = mean_absolute_error(
        y_validation,
        baseline_predictions,
    )

    print("\nFIRST MODEL VALIDATION")
    print("Fitting years: 2000–2010")
    print("Validation years: 2011–2015")
    print(f"Validation rows: {len(model_validation_rows):,}")
    print(
        f"Random forest MAE: {model_mae:.2f} percentage points"
    )
    print(
        f"Median baseline MAE: {baseline_mae:.2f} percentage points"
    )

    # Prototype evaluation settings, not official funding thresholds.
    selection_fraction = 0.20
    severe_loss_threshold = 20.0
    random_repeats = 200

    # Rainfall reference uses only the model-fitting period.
    rainfall_reference = (
        fit_data.groupby(["fips", "crop"], as_index=False)
        .agg(fit_mean_precip_mm=("precip_mm", "mean"))
    )

    ranking_data = model_validation_rows.merge(
        rainfall_reference,
        on=["fips", "crop"],
        how="left",
        validate="many_to_one",
    )

    eligible = (
        ranking_data["precip_mm"].notna()
        & ranking_data["fit_mean_precip_mm"].notna()
        & (ranking_data["fit_mean_precip_mm"] > 0)
        & ranking_data["predicted_loss_pct"].notna()
    )

    ranking_data = ranking_data.loc[eligible].copy()

    ranking_data["rainfall_shortage_pct"] = (
        1
        - ranking_data["precip_mm"]
        / ranking_data["fit_mean_precip_mm"]
    ) * 100

    if ranking_data.empty:
        raise ValueError("No comparable rows for ranking evaluation.")

    ranking_results = []
    rng = np.random.default_rng(42)

    for (state, year), group in ranking_data.groupby(
        ["state_alpha", "year"]
    ):
        group = group.sort_values("fips").copy()

        slots = max(
            1,
            int(np.ceil(len(group) * selection_fraction)),
        )

        model_selected = group.sort_values(
            ["predicted_loss_pct", "fips"],
            ascending=[False, True],
        ).head(slots)

        rainfall_selected = group.sort_values(
            ["rainfall_shortage_pct", "fips"],
            ascending=[False, True],
        ).head(slots)

        for method, selected in [
            ("Model", model_selected),
            ("Rainfall only", rainfall_selected),
        ]:
            ranking_results.append({
                "state": state,
                "year": year,
                "method": method,
                "slots": slots,
                "selected_loss_sum": selected["yield_loss_pct"].sum(),
                "severe_count": (
                    selected["yield_loss_pct"] >= severe_loss_threshold
                ).sum(),
            })

        random_loss_sums = []
        random_severe_counts = []

        for _ in range(random_repeats):
            selected = group.sample(
                n=slots,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )

            random_loss_sums.append(
                selected["yield_loss_pct"].sum()
            )

            random_severe_counts.append(
                (
                    selected["yield_loss_pct"]
                    >= severe_loss_threshold
                ).sum()
            )

        ranking_results.append({
            "state": state,
            "year": year,
            "method": "Random",
            "slots": slots,
            "selected_loss_sum": np.mean(random_loss_sums),
            "severe_count": np.mean(random_severe_counts),
        })

    ranking_results = pd.DataFrame(ranking_results)

    summary = (
        ranking_results.groupby("method")[
            ["slots", "selected_loss_sum", "severe_count"]
        ]
        .sum()
    )

    summary["mean_selected_loss_pct"] = (
        summary["selected_loss_sum"] / summary["slots"]
    )

    summary["severe_share_pct"] = (
        summary["severe_count"] / summary["slots"]
    ) * 100

    print("\nVALIDATION RANKING COMPARISON")
    print("Years: 2011–2015")
    print(f"Comparable county–year rows: {len(ranking_data):,}")
    print("Selection: top 20% per state–year, rounded up")
    print("Severe loss: observed yield loss of at least 20%")
    print(
        f"Random baseline: average of "
        f"{random_repeats} draws per group"
    )
    print(
        summary[
            ["slots", "mean_selected_loss_pct", "severe_share_pct"]
        ]
        .round(2)
        .to_string()
    )

    # Print the detailed breakdown after ranking_results exists.
    ranking_results["mean_selected_loss_pct"] = (
        ranking_results["selected_loss_sum"]
        / ranking_results["slots"]
    )

    ranking_results["severe_share_pct"] = (
        ranking_results["severe_count"]
        / ranking_results["slots"]
    ) * 100

    print("\nRANKING RESULTS BY STATE AND YEAR")
    print(
        ranking_results[
            [
                "state",
                "year",
                "method",
                "slots",
                "mean_selected_loss_pct",
                "severe_share_pct",
            ]
        ]
        .sort_values(["state", "year", "method"])
        .round(2)
        .to_string(index=False)
    )
    
    # Sensitivity check: retain the original results above.
    without_2012 = ranking_results.loc[
        ranking_results["year"] != 2012
    ]

    sensitivity_summary = (
        without_2012.groupby("method")[
            ["slots", "selected_loss_sum", "severe_count"]
        ]
        .sum()
    )

    sensitivity_summary["mean_selected_loss_pct"] = (
        sensitivity_summary["selected_loss_sum"]
        / sensitivity_summary["slots"]
    )

    sensitivity_summary["severe_share_pct"] = (
        sensitivity_summary["severe_count"]
        / sensitivity_summary["slots"]
    ) * 100

    print("\nVALIDATION SENSITIVITY CHECK — EXCLUDING 2012")
    print("Diagnostic only; original validation results remain unchanged.")
    print(
        sensitivity_summary[
            ["slots", "mean_selected_loss_pct", "severe_share_pct"]
        ]
        .round(2)
        .to_string()
    )
    
    # Final evaluation: keep all prototype settings unchanged.
    final_train_rows = train_data.loc[
        train_data["yield_loss_pct"].notna()
    ].copy()

    final_test_rows = test_data.loc[
        test_data["yield_loss_pct"].notna()
    ].copy()

    if final_train_rows.empty or final_test_rows.empty:
        raise ValueError(
            "Final training and testing both need known loss outcomes."
        )

    X_final_train = final_train_rows[FEATURE_COLUMNS]
    y_final_train = final_train_rows["yield_loss_pct"]

    X_final_test = final_test_rows[FEATURE_COLUMNS]
    y_final_test = final_test_rows["yield_loss_pct"]

    # Clone creates fresh estimators with the same settings.
    final_model = clone(model)
    final_baseline = clone(baseline)

    final_model.fit(X_final_train, y_final_train)
    final_baseline.fit(X_final_train, y_final_train)

    final_test_rows["predicted_loss_pct"] = final_model.predict(
        X_final_test
    )

    final_baseline_predictions = final_baseline.predict(
        X_final_test
    )

    final_model_mae = mean_absolute_error(
        y_final_test,
        final_test_rows["predicted_loss_pct"],
    )

    final_baseline_mae = mean_absolute_error(
        y_final_test,
        final_baseline_predictions,
    )

    print("\nFINAL MODEL TEST")
    print("Training years: 2000–2015")
    print("Testing years: 2016–2025")
    print(f"Evaluated county–year rows: {len(final_test_rows):,}")
    print(
        f"Evaluated counties: "
        f"{final_test_rows['fips'].nunique():,}"
    )
    print(
        f"Random forest MAE: "
        f"{final_model_mae:.2f} percentage points"
    )
    print(
        f"Median baseline MAE: "
        f"{final_baseline_mae:.2f} percentage points"
    )

    # Recalculate rainfall references using 2000–2015 only.
    final_rainfall_reference = (
        train_data.groupby(["fips", "crop"], as_index=False)
        .agg(fit_mean_precip_mm=("precip_mm", "mean"))
    )

    final_ranking_data = final_test_rows.merge(
        final_rainfall_reference,
        on=["fips", "crop"],
        how="left",
        validate="many_to_one",
    )

    final_eligible = (
        final_ranking_data["precip_mm"].notna()
        & final_ranking_data["fit_mean_precip_mm"].notna()
        & (final_ranking_data["fit_mean_precip_mm"] > 0)
        & final_ranking_data["predicted_loss_pct"].notna()
    )

    final_ranking_data = final_ranking_data.loc[
        final_eligible
    ].copy()

    if final_ranking_data.empty:
        raise ValueError(
            "No comparable rows for final ranking evaluation."
        )

    final_ranking_data["rainfall_shortage_pct"] = (
        1
        - final_ranking_data["precip_mm"]
        / final_ranking_data["fit_mean_precip_mm"]
    ) * 100

    final_ranking_results = []
    final_rng = np.random.default_rng(42)

    for (state, year), group in final_ranking_data.groupby(
        ["state_alpha", "year"]
    ):
        group = group.sort_values("fips").copy()

        slots = max(
            1,
            int(np.ceil(len(group) * selection_fraction)),
        )

        model_selected = group.sort_values(
            ["predicted_loss_pct", "fips"],
            ascending=[False, True],
        ).head(slots)

        rainfall_selected = group.sort_values(
            ["rainfall_shortage_pct", "fips"],
            ascending=[False, True],
        ).head(slots)

        for method, selected in [
            ("Model", model_selected),
            ("Rainfall only", rainfall_selected),
        ]:
            final_ranking_results.append({
                "state": state,
                "year": year,
                "method": method,
                "slots": slots,
                "selected_loss_sum": selected["yield_loss_pct"].sum(),
                "severe_count": (
                    selected["yield_loss_pct"]
                    >= severe_loss_threshold
                ).sum(),
            })

        random_loss_sums = []
        random_severe_counts = []

        for _ in range(random_repeats):
            selected = group.sample(
                n=slots,
                random_state=int(
                    final_rng.integers(0, 2**31 - 1)
                ),
            )

            random_loss_sums.append(
                selected["yield_loss_pct"].sum()
            )

            random_severe_counts.append(
                (
                    selected["yield_loss_pct"]
                    >= severe_loss_threshold
                ).sum()
            )

        final_ranking_results.append({
            "state": state,
            "year": year,
            "method": "Random",
            "slots": slots,
            "selected_loss_sum": np.mean(random_loss_sums),
            "severe_count": np.mean(random_severe_counts),
        })

    final_ranking_results = pd.DataFrame(
        final_ranking_results
    )

    final_summary = (
        final_ranking_results.groupby("method")[
            ["slots", "selected_loss_sum", "severe_count"]
        ]
        .sum()
    )

    final_summary["mean_selected_loss_pct"] = (
        final_summary["selected_loss_sum"]
        / final_summary["slots"]
    )

    final_summary["severe_share_pct"] = (
        final_summary["severe_count"]
        / final_summary["slots"]
    ) * 100

    print("\nFINAL TEST RANKING COMPARISON")
    print("Years: 2016–2025")
    print(
        f"Comparable county–year rows: "
        f"{len(final_ranking_data):,}"
    )
    print("Selection: top 20% per state–year, rounded up")
    print("Severe loss: observed yield loss of at least 20%")
    print(
        f"Random baseline: average of "
        f"{random_repeats} draws per group"
    )
    print(
        final_summary[
            ["slots", "mean_selected_loss_pct", "severe_share_pct"]
        ]
        .round(2)
        .to_string()
    )

    final_ranking_results["mean_selected_loss_pct"] = (
        final_ranking_results["selected_loss_sum"]
        / final_ranking_results["slots"]
    )

    final_ranking_results["severe_share_pct"] = (
        final_ranking_results["severe_count"]
        / final_ranking_results["slots"]
    ) * 100

    print("\nFINAL TEST RESULTS BY STATE AND YEAR")
    print(
        final_ranking_results[
            [
                "state",
                "year",
                "method",
                "slots",
                "mean_selected_loss_pct",
                "severe_share_pct",
            ]
        ]
        .sort_values(["state", "year", "method"])
        .round(2)
        .to_string(index=False)
    )