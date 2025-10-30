from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


RandomizationMethod = Literal["simple", "stratified", "cluster"]


@dataclass(frozen=True)
class TreatmentArm:
    """Definition for a treatment arm."""

    name: str
    proportion: float


@dataclass(frozen=True)
class RandomizationConfig:
    """Configuration options for a randomization run."""

    id_column: str
    treatment_column: str = "treatment"
    method: RandomizationMethod = "simple"
    arms: List[TreatmentArm] = field(
        default_factory=lambda: [
            TreatmentArm("treatment", 0.5),
            TreatmentArm("control", 0.5),
        ]
    )
    strata: List[str] = field(default_factory=list)
    cluster: Optional[str] = None
    balance_covariates: List[str] = field(default_factory=list)
    iterations: int = 1
    seed: Optional[int] = None
    use_existing_assignment: bool = True


@dataclass
class RandomizationResult:
    """Outcome of a randomization run."""

    assignments: pd.DataFrame
    balance_table: pd.DataFrame
    best_min_pvalue: float
    iterations: int
    used_existing_assignment: bool


class Randomizer:
    """Implements rerandomization with balance checks."""

    def __init__(self, config: RandomizationConfig):
        self.config = config
        self._validate_config()

    def run(self, df: pd.DataFrame) -> RandomizationResult:
        cfg = self.config
        work_df = df.copy()

        if (
            cfg.use_existing_assignment
            and cfg.treatment_column in work_df.columns
            and work_df[cfg.treatment_column].notna().any()
        ):
            balance_table, min_p = self._balance_table(work_df, cfg.treatment_column)
            return RandomizationResult(
                assignments=work_df,
                balance_table=balance_table,
                best_min_pvalue=min_p,
                iterations=0,
                used_existing_assignment=True,
            )

        best_assignment = None
        best_min_pvalue = -np.inf
        best_balance_table = pd.DataFrame()
        total_iterations = max(1, cfg.iterations)

        for i in range(1, total_iterations + 1):
            seed = self._iteration_seed(i)
            assignment = self._assign(work_df, seed)
            work_df[cfg.treatment_column] = assignment
            balance_table, min_p = self._balance_table(work_df, cfg.treatment_column)
            if min_p > best_min_pvalue:
                best_min_pvalue = min_p
                best_assignment = assignment.copy()
                best_balance_table = balance_table.copy()

        if best_assignment is None:
            raise RuntimeError("Randomization failed to produce an assignment.")

        work_df[cfg.treatment_column] = best_assignment
        return RandomizationResult(
            assignments=work_df,
            balance_table=best_balance_table,
            best_min_pvalue=best_min_pvalue,
            iterations=total_iterations,
            used_existing_assignment=False,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _validate_config(self) -> None:
        cfg = self.config
        if not cfg.arms:
            raise ValueError("At least one treatment arm must be provided.")
        total_prop = sum(arm.proportion for arm in cfg.arms)
        if not np.isclose(total_prop, 1.0):
            raise ValueError(
                f"Treatment arm proportions must sum to 1. Got {total_prop:.4f}."
            )
        if cfg.method not in ("simple", "stratified", "cluster"):
            raise ValueError(f"Unsupported randomization method: {cfg.method}")
        if cfg.method == "cluster" and not cfg.cluster:
            raise ValueError("Cluster randomization requires 'cluster' to be set.")
        if cfg.iterations < 1 and not cfg.use_existing_assignment:
            raise ValueError("Iterations must be at least 1.")

    def _iteration_seed(self, iteration: int) -> Optional[int]:
        if self.config.seed is None:
            return None
        return self.config.seed + iteration - 1

    def _assign(self, df: pd.DataFrame, seed: Optional[int]) -> pd.Series:
        method = self.config.method
        if method == "simple":
            assignments = self._assign_simple(len(df), seed)
            return pd.Series(assignments, index=df.index, name=self.config.treatment_column)
        if method == "stratified":
            return self._assign_stratified(df, seed)
        if method == "cluster":
            return self._assign_cluster(df, seed)
        raise RuntimeError(f"Unknown randomization method: {method}")

    def _assign_simple(self, n: int, seed: Optional[int]) -> np.ndarray:
        rng = np.random.default_rng(seed)
        counts = self._target_counts(n)
        assignments = []
        for arm, count in counts.items():
            assignments.extend([arm] * count)
        assignments = np.array(assignments)
        rng.shuffle(assignments)
        return assignments

    def _assign_stratified(self, df: pd.DataFrame, seed: Optional[int]) -> pd.Series:
        strata_cols = self.config.strata
        if not strata_cols:
            return pd.Series(
                self._assign_simple(len(df), seed),
                index=df.index,
                name=self.config.treatment_column,
            )

        assignments = pd.Series(index=df.index, dtype=object, name=self.config.treatment_column)
        for offset, (_, group) in enumerate(
            df.groupby(strata_cols, dropna=False, sort=False)
        ):
            stratum_seed = None if seed is None else seed + offset
            assignments.loc[group.index] = self._assign_simple(len(group), stratum_seed)
        return assignments

    def _assign_cluster(self, df: pd.DataFrame, seed: Optional[int]) -> pd.Series:
        cluster_col = self.config.cluster
        strata_cols = self.config.strata

        clusters = df[[cluster_col] + strata_cols].drop_duplicates().set_index(cluster_col)
        assignments = {}

        for offset, (_, subset) in enumerate(
            clusters.groupby(strata_cols, dropna=False, sort=False)
        ):
            stratum_seed = None if seed is None else seed + offset
            cluster_ids = subset.index.to_list()
            cluster_assignments = self._assign_simple(len(cluster_ids), stratum_seed)
            assignments.update(dict(zip(cluster_ids, cluster_assignments)))

        assigned = df[cluster_col].map(assignments)
        if assigned.isna().any():
            missing = assigned[assigned.isna()].index.to_list()
            raise RuntimeError(f"Missing cluster assignments for rows: {missing[:5]}")
        return assigned.rename(self.config.treatment_column)

    def _target_counts(self, n: int) -> Dict[str, int]:
        props = np.array([arm.proportion for arm in self.config.arms], dtype=float)
        names = [arm.name for arm in self.config.arms]
        expected = props * n
        counts = np.floor(expected).astype(int)
        remainder = n - counts.sum()
        if remainder > 0:
            fractional = expected - counts
            order = np.argsort(-fractional)
            for idx in order[:remainder]:
                counts[idx] += 1
        return dict(zip(names, counts))

    def _balance_table(
        self,
        df: pd.DataFrame,
        treatment_col: str,
    ) -> Tuple[pd.DataFrame, float]:
        covariates = [c for c in self.config.balance_covariates if c in df.columns]
        if not covariates:
            empty = pd.DataFrame(columns=["covariate", "p_value", "min_p_value"])
            return empty, 1.0

        records = []
        min_p = 1.0
        groups = list(df[treatment_col].dropna().unique())
        for cov in covariates:
            cov_series = df[cov]
            if cov_series.dropna().empty:
                p_value = np.nan
                means = {g: np.nan for g in groups}
            else:
                samples = [cov_series[df[treatment_col] == g].dropna().to_numpy() for g in groups]
                samples = [s for s in samples if len(s) > 0]
                if len(samples) <= 1:
                    p_value = np.nan
                else:
                    _, p_value = stats.f_oneway(*samples)
                means = {
                    g: cov_series[df[treatment_col] == g].mean(skipna=True) for g in groups
                }
            if not np.isnan(p_value):
                min_p = min(min_p, p_value)
            records.append(
                {
                    "covariate": cov,
                    "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                    "means": means,
                }
            )
        balance_table = pd.DataFrame.from_records(records)
        balance_table["min_p_value"] = min_p if np.isfinite(min_p) else np.nan
        return balance_table, min_p if np.isfinite(min_p) else np.nan
