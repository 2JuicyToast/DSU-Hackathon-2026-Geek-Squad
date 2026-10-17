from src.data_loader import load_master


def prepare_ranking_data(crop="CORN"):
    """Separate historical development data from final testing data."""
    master = load_master()

    crop_data = master.loc[
        (master["crop"] == crop)
        & master["year"].between(2000, 2025)
        ].copy()

    if crop_data.empty:
        raise ValueError(f"No data found for crop: {crop}")

    # Do not use the full-period anomaly in the modeling dataset.
    crop_data = crop_data.drop(
        columns=["yield_anomaly_pct"],
        errors="ignore",
    )

    # Flag usable observed yields without removing missing-yield records.
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
    train_data, test_data = prepare_ranking_data(crop="CORN")

    print_summary(train_data, "TRAINING DATA")
    print_summary(test_data, "TESTING DATA")