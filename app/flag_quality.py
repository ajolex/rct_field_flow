from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class QualityCheckConfig:
    duration_column: str = "duration"
    min_duration_seconds: Optional[int] = None
    max_duration_seconds: Optional[int] = None
    speed_quantile: float = 0.25
    duplicate_keys: List[str] = field(default_factory=list)
    gps_columns: List[str] = field(default_factory=list)
    other_specify_suffix: str = "_other"
    intervention_columns: Dict[str, Iterable[str]] = field(default_factory=dict)
    numeric_columns: Optional[List[str]] = None
    enumerator_column: str = "enumerator"
    hfc_numeric_columns: Optional[List[str]] = None
    hfc_zscore_threshold: float = 2.5

    @classmethod
    def from_dict(cls, raw: Dict | None) -> "QualityCheckConfig":
        if not raw:
            return cls()
        return cls(
            duration_column=raw.get("duration_column", "duration"),
            min_duration_seconds=raw.get("min_duration_seconds"),
            max_duration_seconds=raw.get("max_duration_seconds"),
            speed_quantile=raw.get("speed_quantile", 0.25),
            duplicate_keys=raw.get("duplicate_keys", []),
            gps_columns=raw.get("gps_columns", []),
            other_specify_suffix=raw.get("other_specify_suffix", "_other"),
            intervention_columns=raw.get("intervention_columns", {}),
            numeric_columns=raw.get("numeric_columns"),
            enumerator_column=raw.get("enumerator_column", "enumerator"),
            hfc_numeric_columns=raw.get("hfc_numeric_columns"),
            hfc_zscore_threshold=raw.get("hfc_zscore_threshold", 2.5),
        )


@dataclass
class QualityResults:
    data: pd.DataFrame
    flags: pd.DataFrame
    flag_counts: pd.Series
    enumerator_summary: pd.DataFrame


def flag_all(df: pd.DataFrame, config: Dict | None = None) -> QualityResults:
    """Apply field quality checks (speeding, outliers, duplicates, fidelity)."""
    cfg = QualityCheckConfig.from_dict(config)
    flags = pd.DataFrame(index=df.index)
    cleaned = df.copy()

    _check_duration(cleaned, flags, cfg)
    _check_numeric_outliers(cleaned, flags, cfg)
    _check_duplicates(cleaned, flags, cfg)
    _merge_other_specify(cleaned, cfg)
    _check_intervention_fidelity(cleaned, flags, cfg)

    flag_counts = flags.sum()
    enumerator_summary = _enumerator_hfc(cleaned, flags, cfg)

    return QualityResults(
        data=cleaned,
        flags=flags,
        flag_counts=flag_counts,
        enumerator_summary=enumerator_summary,
    )


def _check_duration(df: pd.DataFrame, flags: pd.DataFrame, cfg: QualityCheckConfig) -> None:
    col = cfg.duration_column
    if col not in df.columns:
        return
    series = pd.to_numeric(df[col], errors="coerce")
    df[col] = series
    if cfg.min_duration_seconds is not None:
        flags["speeding"] = series < cfg.min_duration_seconds
    else:
        q = cfg.speed_quantile
        threshold = series.quantile(q)
        flags["speeding"] = series < threshold
    if cfg.max_duration_seconds is not None:
        flags["long_interview"] = series > cfg.max_duration_seconds


def _check_numeric_outliers(df: pd.DataFrame, flags: pd.DataFrame, cfg: QualityCheckConfig) -> None:
    if cfg.numeric_columns:
        numeric_cols = [c for c in cfg.numeric_columns if c in df.columns]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col == cfg.duration_column:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            flags[f"outlier_{col}"] = (series < lower) | (series > upper)


def _check_duplicates(df: pd.DataFrame, flags: pd.DataFrame, cfg: QualityCheckConfig) -> None:
    keys = [k for k in cfg.duplicate_keys if k in df.columns]
    if keys:
        flags["duplicate_case"] = df.duplicated(keys, keep=False)
    gps_cols = [c for c in cfg.gps_columns if c in df.columns]
    if len(gps_cols) == 2:
        flags["duplicate_gps"] = df.duplicated(gps_cols, keep=False)


def _merge_other_specify(df: pd.DataFrame, cfg: QualityCheckConfig) -> None:
    suffix = cfg.other_specify_suffix
    for col in df.columns:
        if col.endswith(suffix):
            base = col[: -len(suffix)]
            if base in df.columns:
                df[base] = df[base].fillna("").astype(str).str.strip()
                df[col] = df[col].fillna("").astype(str).str.strip()
                df[base] = (df[base] + " " + df[col]).str.strip()


def _check_intervention_fidelity(df: pd.DataFrame, flags: pd.DataFrame, cfg: QualityCheckConfig) -> None:
    for col, expected in cfg.intervention_columns.items():
        if col not in df.columns:
            continue
        values = expected if isinstance(expected, (list, tuple, set)) else [expected]
        flags[f"fidelity_{col}"] = ~df[col].isin(values)


def _enumerator_hfc(df: pd.DataFrame, flags: pd.DataFrame, cfg: QualityCheckConfig) -> pd.DataFrame:
    enumerator_col = cfg.enumerator_column
    if enumerator_col not in df.columns:
        return pd.DataFrame()

    risk_score = flags.sum(axis=1)
    summary = pd.DataFrame(
        {
            "enumerator": df[enumerator_col],
            "risk_score": risk_score,
        }
    ).groupby("enumerator").agg(
        total_cases=("risk_score", "size"),
        flagged_cases=("risk_score", lambda s: int((s > 0).sum())),
        avg_risk=("risk_score", "mean"),
    )
    summary["flagged_rate"] = summary["flagged_cases"] / summary["total_cases"]
    summary = summary.reset_index()

    numeric_cols = cfg.hfc_numeric_columns
    if numeric_cols:
        numeric_cols = [c for c in numeric_cols if c in df.columns]
    else:
        numeric_cols = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        z_scores = (series - series.mean()) / series.std(ddof=0) if series.std(ddof=0) else pd.Series(0, index=series.index)
        df[f"z_{col}"] = z_scores
    if numeric_cols:
        z_cols = [f"z_{col}" for col in numeric_cols]
        hfc = (
            df[[enumerator_col] + z_cols]
            .groupby(enumerator_col, dropna=True)
            .mean()
            .rename(columns={c: c.replace("z_", "mean_z_") for c in z_cols})
            .reset_index()
        )
        summary = summary.merge(hfc, on="enumerator", how="left")
    return summary
