from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class BackcheckConfig:
    sample_size: int = 50
    high_risk_quota: float = 0.6
    random_seed: int = 42
    id_column: str = "participant_id"
    contact_columns: List[str] = field(default_factory=lambda: ["phone"])
    location_columns: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "BackcheckConfig":
        if not raw:
            return cls()
        return cls(
            sample_size=raw.get("sample_size", 50),
            high_risk_quota=raw.get("high_risk_quota", 0.6),
            random_seed=raw.get("random_seed", 42),
            id_column=raw.get("id_column", "participant_id"),
            contact_columns=raw.get("contact_columns", ["phone"]),
            location_columns=raw.get("location_columns", []),
        )


def sample_backchecks(
    df: pd.DataFrame,
    flags: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Select a backcheck roster combining high-risk and random cases."""
    cfg = BackcheckConfig.from_dict(config)
    if df.empty:
        return pd.DataFrame()

    work_df = df.copy()
    risk_score = flags.sum(axis=1)
    work_df["risk_score"] = risk_score

    high_risk = work_df[risk_score > 0]
    high_n = min(round(cfg.sample_size * cfg.high_risk_quota), len(high_risk))
    if high_n > 0 and len(high_risk) > 0:
        high_sample = high_risk.sample(n=high_n, random_state=cfg.random_seed)
    else:
        high_sample = pd.DataFrame(columns=work_df.columns)

    remaining = work_df.drop(high_sample.index, errors="ignore")
    remaining_n = max(0, cfg.sample_size - len(high_sample))
    if remaining_n > 0 and len(remaining) > 0:
        random_sample = remaining.sample(
            n=min(remaining_n, len(remaining)), random_state=cfg.random_seed
        )
    else:
        random_sample = pd.DataFrame(columns=work_df.columns)

    backchecks = pd.concat([high_sample, random_sample]).drop_duplicates(cfg.id_column)

    columns = [cfg.id_column, "risk_score"] + cfg.contact_columns + cfg.location_columns
    columns = [c for c in columns if c in backchecks.columns]
    return backchecks[columns].reset_index(drop=True)
