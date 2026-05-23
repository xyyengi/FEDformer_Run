from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_TZ = "Europe/Berlin"
UTC_TZ = "UTC"
BASE_DIR = Path(__file__).resolve().parent
GENERATION_FILE = BASE_DIR / "Actual_generation_202301010000_202603011700_Hour (1).csv"
CONSUMPTION_FILE = BASE_DIR / "Actual_consumption_202301010000_202603011700_Hour (2).csv"
OUTPUT_FILE = BASE_DIR / "Wind_Solar_Load_Processed.csv"

EXPECTED_START_UTC = pd.Timestamp("2022-12-31 23:00:00", tz=UTC_TZ)
EXPECTED_END_UTC = pd.Timestamp("2026-03-01 15:00:00", tz=UTC_TZ)
EXPECTED_LEN = int((EXPECTED_END_UTC - EXPECTED_START_UTC) / pd.Timedelta(hours=1)) + 1


def _parse_source_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", thousands=",")
    for col in df.columns:
        if col not in ["Start date", "End date"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _add_utc_axis(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    local_start = pd.to_datetime(df["Start date"], format="%b %d, %Y %I:%M %p")
    localized_start = local_start.dt.tz_localize(
        LOCAL_TZ,
        ambiguous="infer",
        nonexistent="raise",
    )

    result = df.copy()
    result["timestamp_utc"] = localized_start.dt.tz_convert(UTC_TZ)

    print(f"[{source_name}] rows: {len(result)}")
    print(f"[{source_name}] UTC start: {result['timestamp_utc'].iloc[0]}")
    print(f"[{source_name}] UTC end:   {result['timestamp_utc'].iloc[-1]}")
    return result


def _assert_time_axis(df: pd.DataFrame, name: str, expected_len: int = EXPECTED_LEN) -> None:
    ts = df["timestamp_utc"]
    assert str(ts.dt.tz) == UTC_TZ, f"{name}: timestamp_utc must be timezone-aware UTC"
    assert ts.is_monotonic_increasing, f"{name}: timestamp_utc is not monotonic increasing"
    assert not ts.duplicated().any(), f"{name}: duplicate timestamp_utc values found"
    deltas = ts.diff().dropna()
    assert (deltas == pd.Timedelta(hours=1)).all(), f"{name}: non-1h timestamp deltas found"
    assert len(df) == expected_len, f"{name}: length {len(df)} != expected {expected_len}"


def _assert_no_missing_or_inf(df: pd.DataFrame, name: str, columns=None) -> None:
    checked = df if columns is None else df[columns]
    assert not checked.isna().any().any(), f"{name}: NaN values found"
    numeric = checked.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all(), f"{name}: inf values found"


def build_processed_dataset() -> pd.DataFrame:
    generation = _add_utc_axis(_parse_source_csv(GENERATION_FILE), "generation")
    consumption = _add_utc_axis(_parse_source_csv(CONSUMPTION_FILE), "consumption")

    _assert_time_axis(generation, "generation")
    _assert_time_axis(consumption, "consumption")
    gen_cols = [
        "timestamp_utc",
        "Wind offshore [MWh] Calculated resolutions",
        "Wind onshore [MWh] Calculated resolutions",
        "Photovoltaics [MWh] Calculated resolutions",
    ]
    load_cols = [
        "timestamp_utc",
        "grid load [MWh] Calculated resolutions",
    ]
    _assert_no_missing_or_inf(generation, "generation used columns", gen_cols)
    _assert_no_missing_or_inf(consumption, "consumption used columns", load_cols)

    gen = generation[gen_cols].copy()
    gen["Wind"] = (
        gen["Wind offshore [MWh] Calculated resolutions"]
        + gen["Wind onshore [MWh] Calculated resolutions"]
    )
    gen["Solar"] = gen["Photovoltaics [MWh] Calculated resolutions"]
    gen = gen[["timestamp_utc", "Wind", "Solar"]]

    load = consumption[load_cols].copy()
    load = load.rename(columns={"grid load [MWh] Calculated resolutions": "Load"})

    merged = pd.merge(gen, load, on="timestamp_utc", how="inner", validate="one_to_one")
    _assert_time_axis(merged, "merged")
    _assert_no_missing_or_inf(merged, "merged")

    local_time = merged["timestamp_utc"].dt.tz_convert(LOCAL_TZ)
    night_mask = (local_time.dt.hour >= 19) | (local_time.dt.hour <= 5)
    merged.loc[night_mask, "Solar"] = 0.0

    processed = pd.DataFrame(
        {
            "date": merged["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "Wind": merged["Wind"],
            "Solar": merged["Solar"],
            "Load": merged["Load"],
            "month": local_time.dt.month,
            "day": local_time.dt.day,
            "weekday": local_time.dt.weekday,
            "hour": local_time.dt.hour,
        }
    )

    parsed_utc = pd.to_datetime(processed["date"], utc=True)
    validation = processed.copy()
    validation["timestamp_utc"] = parsed_utc
    _assert_time_axis(validation, "processed")
    _assert_no_missing_or_inf(processed, "processed")

    processed.to_csv(OUTPUT_FILE, index=False)

    print("\nProcessed data written:")
    print(f"  path: {OUTPUT_FILE}")
    print("  date policy: UTC timestamp with +0000 offset")
    print(f"  rows: {len(processed)}")
    print(f"  UTC start: {parsed_utc.iloc[0]}")
    print(f"  UTC end:   {parsed_utc.iloc[-1]}")
    print("  local time features: derived from UTC converted to Europe/Berlin")
    print("\nDelta counts:")
    print(parsed_utc.diff().dropna().value_counts().sort_index().to_string())
    print("\nHead:")
    print(processed.head(10).to_string(index=False))

    return processed


if __name__ == "__main__":
    build_processed_dataset()
