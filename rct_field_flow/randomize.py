from __future__ import annotations

import warnings
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
    """Outcome of a randomization run.
    
    Attributes:
        assignments: DataFrame with treatment assignments
        balance_table: DataFrame with balance test results
        best_min_pvalue: Highest minimum p-value across balance tests
        iterations: Number of rerandomization iterations run
        used_existing_assignment: Whether existing assignments were used
        diagnostics: Additional diagnostic information
    """

    assignments: pd.DataFrame
    balance_table: pd.DataFrame
    best_min_pvalue: float
    iterations: int
    used_existing_assignment: bool
    diagnostics: Dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Results from validating randomization fairness.
    
    Attributes:
        assignment_probabilities: DataFrame with average assignment probability for each unit and arm
        summary_stats: Dictionary with summary statistics about assignment probabilities
        is_valid: Boolean indicating if randomization appears fair
        warnings: List of warning messages if issues detected
    """

    assignment_probabilities: pd.DataFrame
    summary_stats: Dict
    is_valid: bool
    warnings: List[str] = field(default_factory=list)


class Randomizer:
    """Implements rerandomization with balance checks."""

    def __init__(self, config: RandomizationConfig):
        self.config = config
        self._validate_config()

    def run(self, df: pd.DataFrame, verbose: bool = False) -> RandomizationResult:
        """Run randomization with optional rerandomization.
        
        Args:
            df: DataFrame with units to randomize
            verbose: If True, print progress and diagnostic information
            
        Returns:
            RandomizationResult with assignments and diagnostics
            
        Note:
            When using rerandomization (iterations > 1), be aware that this can
            affect randomization inference assumptions. See Bruhn & McKenzie (2009)
            for details on the trade-offs between balance and inference validity.
        """
        cfg = self.config
        work_df = df.copy()

        # Validate input data
        if cfg.id_column not in work_df.columns:
            raise ValueError(f"ID column '{cfg.id_column}' not found in data")
        
        # Check for duplicates
        if work_df[cfg.id_column].duplicated().any():
            raise ValueError(f"Duplicate IDs found in column '{cfg.id_column}'")

        if (
            cfg.use_existing_assignment
            and cfg.treatment_column in work_df.columns
            and work_df[cfg.treatment_column].notna().any()
        ):
            if verbose:
                print(f"Using existing assignment in column '{cfg.treatment_column}'")
            balance_table, min_p = self._balance_table(work_df, cfg.treatment_column)
            self._validate_treatment_distribution(work_df, cfg.treatment_column, verbose)
            return RandomizationResult(
                assignments=work_df,
                balance_table=balance_table,
                best_min_pvalue=min_p,
                iterations=0,
                used_existing_assignment=True,
                diagnostics={"existing_assignment": True}
            )

        best_assignment = None
        best_min_pvalue = -np.inf
        best_balance_table = pd.DataFrame()
        total_iterations = max(1, cfg.iterations)
        
        # For validation: track assignment frequencies if doing rerandomization
        assignment_tracker = {} if total_iterations > 100 else None
        p_value_history = []

        if verbose and total_iterations > 1:
            print(f"Running {total_iterations} iterations to find best balance...")
            if total_iterations > 1000:
                warnings.warn(
                    "Rerandomization with many iterations can affect randomization inference. "
                    "Consider validating that no units are systematically favored.",
                    UserWarning
                )

        for i in range(1, total_iterations + 1):
            seed = self._iteration_seed(i)
            assignment = self._assign(work_df, seed)
            work_df[cfg.treatment_column] = assignment
            
            # Track assignments for validation
            if assignment_tracker is not None:
                for idx, arm in assignment.items():
                    if idx not in assignment_tracker:
                        assignment_tracker[idx] = []
                    assignment_tracker[idx].append(arm)
            
            balance_table, min_p = self._balance_table(work_df, cfg.treatment_column)
            p_value_history.append(min_p)
            
            if min_p > best_min_pvalue:
                best_min_pvalue = min_p
                best_assignment = assignment.copy()
                best_balance_table = balance_table.copy()
            
            if verbose and i % max(1, total_iterations // 10) == 0:
                print(f"  Iteration {i}/{total_iterations}, current best p-value: {best_min_pvalue:.4f}")

        if best_assignment is None:
            raise RuntimeError("Randomization failed to produce an assignment.")

        work_df[cfg.treatment_column] = best_assignment
        
        # Validate treatment distribution
        self._validate_treatment_distribution(work_df, cfg.treatment_column, verbose)
        
        # Compute diagnostics
        diagnostics = {
            "p_value_history": p_value_history,
            "mean_p_value": np.mean(p_value_history),
            "median_p_value": np.median(p_value_history),
        }
        
        # Check for systematic assignment bias if we tracked
        if assignment_tracker:
            diagnostics["assignment_check"] = self._check_assignment_probabilities(
                assignment_tracker, work_df, verbose
            )
        
        if verbose:
            print("\nRandomization complete!")
            print(f"Best minimum p-value: {best_min_pvalue:.4f}")
            print(f"Mean p-value across iterations: {diagnostics['mean_p_value']:.4f}")
        
        return RandomizationResult(
            assignments=work_df,
            balance_table=best_balance_table,
            best_min_pvalue=best_min_pvalue,
            iterations=total_iterations,
            used_existing_assignment=False,
            diagnostics=diagnostics
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _validate_config(self) -> None:
        cfg = self.config
        if not cfg.arms:
            raise ValueError("At least one treatment arm must be provided.")
        total_prop = sum(arm.proportion for arm in cfg.arms)
        # Allow tolerance of ±0.01 for rounding errors (e.g., 0.33 + 0.33 + 0.34 = 1.00)
        if not np.isclose(total_prop, 1.0, atol=0.01):
            raise ValueError(
                f"Treatment arm proportions must sum to 1 (±0.01 tolerance). Got {total_prop:.4f}."
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

        # If no strata, randomize all clusters together
        if not strata_cols:
            cluster_ids = clusters.index.to_list()
            cluster_assignments = self._assign_simple(len(cluster_ids), seed)
            assignments.update(dict(zip(cluster_ids, cluster_assignments)))
        else:
            # Randomize clusters within each stratum
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

    def _validate_treatment_distribution(
        self, 
        df: pd.DataFrame, 
        treatment_col: str,
        verbose: bool = False
    ) -> None:
        """Validate that treatment assignments match expected proportions.
        
        Raises:
            AssertionError: If assignments deviate significantly from expected proportions
        """
        treatment_counts = df[treatment_col].value_counts()
        total = len(df)
        
        for arm in self.config.arms:
            expected = arm.proportion * total
            actual = treatment_counts.get(arm.name, 0)
            # Allow some tolerance for rounding
            diff = abs(actual - expected)
            
            # For cluster randomization, allow more tolerance since we can't perfectly
            # balance individual counts when randomizing at the cluster level
            if self.config.method == "cluster":
                # Allow up to 10% deviation or at least 10 units, whichever is larger
                max_diff = max(10, total * 0.10)
            else:
                # For non-cluster randomization, strict tolerance
                max_diff = max(1, len(self.config.arms))
            
            if diff > max_diff:
                raise AssertionError(
                    f"Treatment arm '{arm.name}' has {actual} observations but expected "
                    f"~{expected:.0f} (proportion {arm.proportion}). Difference: {diff}"
                )
        
        if verbose:
            print("\nTreatment distribution validation passed:")
            for arm in self.config.arms:
                actual = treatment_counts.get(arm.name, 0)
                print(f"  {arm.name}: {actual} ({100*actual/total:.1f}%, expected {100*arm.proportion:.1f}%)")

    def _check_assignment_probabilities(
        self,
        assignment_tracker: Dict,
        df: pd.DataFrame,
        verbose: bool = False
    ) -> Dict:
        """Check if any units are systematically favored in treatment assignment.
        
        Following the reference guide's advice to check assignment probabilities
        across iterations to ensure fairness.
        
        Returns:
            Dictionary with statistics about assignment probability distribution
        """
        # Calculate empirical assignment probabilities
        arm_names = [arm.name for arm in self.config.arms]
        prob_stats = {arm: [] for arm in arm_names}
        
        for idx, assignments in assignment_tracker.items():
            for arm in arm_names:
                prob = assignments.count(arm) / len(assignments)
                prob_stats[arm].append(prob)
        
        # Compute statistics
        diagnostics = {}
        for arm in arm_names:
            probs = np.array(prob_stats[arm])
            diagnostics[arm] = {
                "mean": float(np.mean(probs)),
                "std": float(np.std(probs)),
                "min": float(np.min(probs)),
                "max": float(np.max(probs)),
            }
            
            # Check for concerning patterns
            expected = next(a.proportion for a in self.config.arms if a.name == arm)
            if abs(np.mean(probs) - expected) > 0.05:  # More than 5% deviation
                warnings.warn(
                    f"Units' average probability of '{arm}' assignment ({np.mean(probs):.3f}) "
                    f"deviates from expected ({expected:.3f}). Some units may be systematically favored.",
                    UserWarning
                )
        
        if verbose:
            print("\nAssignment probability check:")
            for arm, stats in diagnostics.items():
                expected = next(a.proportion for a in self.config.arms if a.name == arm)
                print(f"  {arm}: mean={stats['mean']:.3f} (expected={expected:.3f}), "
                      f"std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        return diagnostics

    def validate_randomization(
        self,
        df: pd.DataFrame,
        n_simulations: int = 500,
        base_seed: Optional[int] = None,
        verbose: bool = False
    ) -> ValidationResult:
        """Validate randomization fairness by running it multiple times with different seeds.
        
        Following the best practice from the RANDOMIZATION.md guide: run the randomization
        multiple times (e.g., 500) with different seeds and check if each observation has
        an approximately equal probability of being assigned to each treatment arm.
        
        Args:
            df: DataFrame with units to randomize
            n_simulations: Number of times to run randomization (default: 500)
            base_seed: Base seed for simulations. Each simulation uses base_seed + i
            verbose: If True, print progress and diagnostic information
            
        Returns:
            ValidationResult with assignment probabilities and validation diagnostics
            
        Note:
            This validation helps detect if any units are systematically favored for
            certain treatment arms, which would indicate a problem with the randomization code.
        """
        cfg = self.config
        work_df = df.copy()
        
        # Validate input data
        if cfg.id_column not in work_df.columns:
            raise ValueError(f"ID column '{cfg.id_column}' not found in data")
        
        # Check for duplicates
        if work_df[cfg.id_column].duplicated().any():
            raise ValueError(f"Duplicate IDs found in column '{cfg.id_column}'")
        
        if verbose:
            print(f"Running {n_simulations} randomization simulations to validate fairness...")
            print(f"Method: {cfg.method}")
            if cfg.strata:
                print(f"Stratification: {', '.join(cfg.strata)}")
            if cfg.cluster:
                print(f"Cluster: {cfg.cluster}")
            print("")
        
        # Track assignments for each unit across all simulations
        arm_names = [arm.name for arm in cfg.arms]
        assignment_counts = {arm: np.zeros(len(work_df)) for arm in arm_names}
        
        # Temporarily disable rerandomization iterations for this validation
        original_iterations = cfg.iterations
        original_use_existing = cfg.use_existing_assignment
        temp_config = RandomizationConfig(
            id_column=cfg.id_column,
            treatment_column=cfg.treatment_column,
            method=cfg.method,
            arms=cfg.arms,
            strata=cfg.strata,
            cluster=cfg.cluster,
            balance_covariates=[],  # Don't need balance checks for validation
            iterations=1,  # Single assignment per simulation
            seed=None,  # Will be set per iteration
            use_existing_assignment=False
        )
        temp_randomizer = Randomizer(temp_config)
        
        # Run simulations
        for i in range(n_simulations):
            if base_seed is not None:
                temp_config = RandomizationConfig(
                    id_column=cfg.id_column,
                    treatment_column=cfg.treatment_column,
                    method=cfg.method,
                    arms=cfg.arms,
                    strata=cfg.strata,
                    cluster=cfg.cluster,
                    balance_covariates=[],
                    iterations=1,
                    seed=base_seed + i,
                    use_existing_assignment=False
                )
                temp_randomizer = Randomizer(temp_config)
            
            # Run single randomization
            sim_df = work_df.copy()
            seed = None if base_seed is None else base_seed + i
            assignment = temp_randomizer._assign(sim_df, seed)
            
            # Count assignments for each unit
            for idx, (row_idx, arm) in enumerate(assignment.items()):
                assignment_counts[arm][idx] += 1
            
            if verbose and (i + 1) % max(1, n_simulations // 10) == 0:
                print(f"  Completed {i + 1}/{n_simulations} simulations")
        
        # Calculate probabilities
        prob_df = work_df[[cfg.id_column]].copy()
        for arm in arm_names:
            prob_df[f"prob_{arm}"] = assignment_counts[arm] / n_simulations
        
        # Calculate average probability (should be close to arm proportion for each unit)
        prob_df["avg_treatment_assignment"] = prob_df[[f"prob_{arm}" for arm in arm_names]].mean(axis=1)
        
        # Compute summary statistics
        summary_stats = {}
        warnings_list = []
        is_valid = True
        
        for arm in arm_names:
            probs = prob_df[f"prob_{arm}"].values
            expected = next(a.proportion for a in cfg.arms if a.name == arm)
            
            stats_dict = {
                "mean": float(np.mean(probs)),
                "std": float(np.std(probs)),
                "min": float(np.min(probs)),
                "max": float(np.max(probs)),
                "expected": expected,
                "mean_deviation": float(abs(np.mean(probs) - expected)),
            }
            summary_stats[arm] = stats_dict
            
            # Check for issues
            # 1. Mean should be close to expected proportion
            if abs(np.mean(probs) - expected) > 0.05:
                is_valid = False
                warnings_list.append(
                    f"Arm '{arm}': Mean probability ({np.mean(probs):.3f}) deviates from "
                    f"expected ({expected:.3f}) by more than 5%"
                )
            
            # 2. Check if some units are almost always in treatment or control
            # Standard deviation should be close to sqrt(p*(1-p)) for binomial
            expected_std = np.sqrt(expected * (1 - expected))
            
            # Count units with extreme probabilities (>90% or <10% when expecting 50%)
            if expected > 0.2 and expected < 0.8:  # Only check for balanced designs
                extreme_low = np.sum(probs < expected - 0.3)
                extreme_high = np.sum(probs > expected + 0.3)
                if extreme_low > len(probs) * 0.05 or extreme_high > len(probs) * 0.05:
                    is_valid = False
                    warnings_list.append(
                        f"Arm '{arm}': {extreme_low + extreme_high} units ({100*(extreme_low + extreme_high)/len(probs):.1f}%) "
                        f"have extreme assignment probabilities (>30% deviation from expected)"
                    )
        
        if verbose:
            print("\nValidation Results:")
            print(f"  Valid: {'✓ Yes' if is_valid else '✗ No'}")
            print("\n  Summary Statistics by Arm:")
            for arm, stats in summary_stats.items():
                print(f"    {arm}:")
                print(f"      Mean probability: {stats['mean']:.4f} (expected: {stats['expected']:.4f})")
                print(f"      Std deviation: {stats['std']:.4f}")
                print(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            
            if warnings_list:
                print("\n  Warnings:")
                for warning in warnings_list:
                    print(f"    - {warning}")
            else:
                print("\n  ✓ No issues detected. Randomization appears fair.")
        
        return ValidationResult(
            assignment_probabilities=prob_df,
            summary_stats=summary_stats,
            is_valid=is_valid,
            warnings=warnings_list
        )
