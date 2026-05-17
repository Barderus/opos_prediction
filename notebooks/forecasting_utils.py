from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import holidays
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "opos_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ID_PATTERN = r"^\d{4}-\d{5}$"
FOCUS_AGES = ("juvenile", "immature", "young_of_year", "neonate")

STATUS_MAP = {
    "R": "Released",
    "EAE": "EuthanizedAfterExam",
    "EOA": "EuthanizedAfterArrival",
    "D": "Dead",
    "E": "Euthanized",
    "D24": "DeadAfter24Hours",
    "DOA": "DeadOnArrival",
    "DBE": "DeadBeforeExam",
    "RVP": "TransferredToRehabber",
    "RTW": "ReturnToWild",
    "E24": "EuthanizedAfter24Hours",
    "T": "Transferred",
    "Reh": "Rehab",
    "ESC": "Escaped",
    "FOS": "FoundOnSite",
}

AGE_MAP = {
    "J": "juvenile",
    "J (juvenile)": "juvenile",
    "juvenile": "juvenile",
    "juvenille": "juvenile",
    "I": "immature",
    "I (immature)": "immature",
    "A": "adult",
    "A (adult)": "adult",
    "YoY": "young_of_year",
    "N": "neonate",
    "UNK": "unknown",
    "L": "unknown",
    "H": "unknown",
    "F": "unknown",
    "": "unknown",
}


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_opossum_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = load_raw_data(path).rename(
        columns={
            "Id": "id",
            "Name": "name",
            "Admission date": "admission_date",
            "Status": "status",
            "Final date": "final_date",
            "Age": "age_raw",
            "City": "city",
        }
    )

    valid_id = df["id"].astype("string").str.fullmatch(ID_PATTERN, na=False)
    df = df.loc[valid_id].copy()

    df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
    df["final_date"] = pd.to_datetime(df["final_date"], errors="coerce")
    df = df.dropna(subset=["admission_date"]).copy()

    df["age_raw"] = df["age_raw"].astype("string").fillna("").str.strip()
    df["age_group"] = df["age_raw"].replace(AGE_MAP).fillna("unknown")
    df["status"] = (
        df["status"].astype("string").str.strip().replace("", pd.NA).replace(STATUS_MAP)
    )
    df["city"] = (
        df["city"]
        .astype("string")
        .fillna("Unknown")
        .str.strip()
        .replace("", "Unknown")
    )
    df["year"] = df["admission_date"].dt.year.astype(int)
    df["month"] = df["admission_date"].dt.month.astype(int)
    df["day"] = df["admission_date"].dt.day.astype(int)
    df["day_of_week"] = df["admission_date"].dt.day_name()
    df["week_of_year"] = df["admission_date"].dt.isocalendar().week.astype(int)
    df["days_in_care"] = (df["final_date"] - df["admission_date"]).dt.days
    df["is_focus_age"] = df["age_group"].isin(FOCUS_AGES)

    return df.sort_values("admission_date").reset_index(drop=True)


def age_focus_data(df: pd.DataFrame, focus_ages: Iterable[str] = FOCUS_AGES) -> pd.DataFrame:
    return df.loc[df["age_group"].isin(tuple(focus_ages))].copy()


def build_daily_arrivals(
    df: pd.DataFrame,
    focus_ages: Iterable[str] = FOCUS_AGES,
    fill_missing_dates: bool = True,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    focus_df = age_focus_data(df, focus_ages)
    daily = (
        focus_df.groupby("admission_date")
        .size()
        .rename("arrivals")
        .to_frame()
        .sort_index()
    )

    if fill_missing_dates:
        overall_end = pd.to_datetime(df["admission_date"]).max()
        if end_date is not None:
            overall_end = max(overall_end, pd.Timestamp(end_date))
        full_range = pd.date_range(daily.index.min(), overall_end, freq="D")
        daily = daily.reindex(full_range, fill_value=0)
        daily.index.name = "admission_date"

    daily = daily.reset_index()
    return add_calendar_features(daily)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["admission_date"] = pd.to_datetime(out["admission_date"])
    us_holidays = holidays.US()
    iso = out["admission_date"].dt.isocalendar()

    out["year"] = out["admission_date"].dt.year
    out["quarter"] = out["admission_date"].dt.quarter
    out["month"] = out["admission_date"].dt.month
    out["day"] = out["admission_date"].dt.day
    out["day_of_week"] = out["admission_date"].dt.dayofweek
    out["day_of_year"] = out["admission_date"].dt.dayofyear
    out["week_of_year"] = iso.week.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_holiday"] = out["admission_date"].isin(us_holidays).astype(int)

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    return out


def add_lag_features(df: pd.DataFrame, target_col: str = "arrivals") -> pd.DataFrame:
    out = df.copy()
    for lag in (1, 7, 14, 28):
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    for window in (7, 14, 28):
        shifted = out[target_col].shift(1)
        out[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        out[f"rolling_std_{window}"] = shifted.rolling(window).std()
        out[f"rolling_max_{window}"] = shifted.rolling(window).max()

    return out


def apply_known_absence_window(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    target_col: str = "arrivals",
) -> pd.DataFrame:
    out = df.copy()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    mask = (pd.to_datetime(out["admission_date"]) >= start_ts) & (
        pd.to_datetime(out["admission_date"]) <= end_ts
    )
    out.loc[mask, target_col] = 0.0
    return out


def build_model_frame(
    df: pd.DataFrame,
    focus_ages: Iterable[str] = FOCUS_AGES,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    daily = build_daily_arrivals(df, focus_ages=focus_ages, end_date=end_date)
    model_df = add_lag_features(daily)
    return model_df.dropna().reset_index(drop=True)


def split_for_2026_forecast(
    df: pd.DataFrame,
    validation_start: str = "2025-01-01",
    test_start: str = "2026-01-01",
) -> SplitData:
    validation_start_ts = pd.Timestamp(validation_start)
    test_start_ts = pd.Timestamp(test_start)

    train = df.loc[df["admission_date"] < validation_start_ts].copy()
    validation = df.loc[
        (df["admission_date"] >= validation_start_ts)
        & (df["admission_date"] < test_start_ts)
    ].copy()
    test = df.loc[df["admission_date"] >= test_start_ts].copy()

    return SplitData(train=train, validation=validation, test=test)


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"admission_date", "arrivals"}
    return [column for column in df.columns if column not in exclude]


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> dict[str, float | str]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    positive_mask = y_true > 0
    if positive_mask.any():
        mape = (
            np.abs((y_true[positive_mask] - y_pred[positive_mask]) / y_true[positive_mask]).mean()
            * 100
        )
    else:
        mape = np.nan

    return {
        "model": model_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    }


def month_day_baseline(train_df: pd.DataFrame, forecast_df: pd.DataFrame) -> np.ndarray:
    reference = (
        train_df.assign(month=train_df["admission_date"].dt.month, day=train_df["admission_date"].dt.day)
        .groupby(["month", "day"])["arrivals"]
        .mean()
    )
    keyed = forecast_df.assign(
        month=forecast_df["admission_date"].dt.month,
        day=forecast_df["admission_date"].dt.day,
    )
    overall_mean = float(train_df["arrivals"].mean())
    return (
        keyed.set_index(["month", "day"]).index.to_series().map(reference).fillna(overall_mean).to_numpy()
    )


def forecast_date_range(
    last_observed_date: str | pd.Timestamp,
    horizon_days: int,
) -> pd.DatetimeIndex:
    last_observed_ts = pd.Timestamp(last_observed_date)
    if horizon_days <= 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    return pd.date_range(last_observed_ts + pd.Timedelta(days=1), periods=horizon_days, freq="D")


def recursive_forecast(
    model,
    history_df: pd.DataFrame,
    forecast_dates: Iterable[str | pd.Timestamp],
    features: list[str],
    target_col: str = "arrivals",
) -> pd.DataFrame:
    history = history_df[["admission_date", target_col]].copy()
    history["admission_date"] = pd.to_datetime(history["admission_date"])

    forecast_rows: list[pd.DataFrame] = []
    for forecast_date in pd.to_datetime(list(forecast_dates)):
        candidate = pd.concat(
            [
                history,
                pd.DataFrame({"admission_date": [forecast_date], target_col: [np.nan]}),
            ],
            ignore_index=True,
        )
        candidate = add_lag_features(add_calendar_features(candidate), target_col=target_col)
        row = candidate.iloc[[-1]].copy()
        prediction = float(model.predict(row[features])[0])
        row[target_col] = prediction
        history = pd.concat([history, row[["admission_date", target_col]]], ignore_index=True)
        forecast_rows.append(row)

    if not forecast_rows:
        empty_cols = ["admission_date", target_col] + features
        return pd.DataFrame(columns=empty_cols)

    return pd.concat(forecast_rows, ignore_index=True)


def zero_before_season_start(
    df: pd.DataFrame,
    prediction_cols: Iterable[str],
    start_month: int = 3,
    start_day: int = 14,
    date_col: str = "admission_date",
) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    before_season = (dates.dt.month < start_month) | (
        (dates.dt.month == start_month) & (dates.dt.day < start_day)
    )

    for column in prediction_cols:
        out.loc[before_season, column] = 0.0

    return out


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path
