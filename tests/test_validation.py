"""Tests for randomization validation feature."""
import numpy as np
import pandas as pd
import pytest

from rct_field_flow.randomize import (
    RandomizationConfig,
    Randomizer,
    TreatmentArm,
    ValidationResult,
)


@pytest.fixture
def simple_dataframe():
    """Create a simple test dataframe."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(100),
        "age": np.random.randint(18, 65, 100),
        "income": np.random.normal(50000, 15000, 100),
    })


@pytest.fixture
def stratified_dataframe():
    """Create a stratified test dataframe."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(200),
        "age": np.random.randint(18, 65, 200),
        "region": np.random.choice(["North", "South"], 200),
        "gender": np.random.choice(["M", "F"], 200),
    })


def test_validate_simple_randomization(simple_dataframe):
    """Test validation with simple randomization."""
    config = RandomizationConfig(
        id_column="id",
        treatment_column="treatment",
        method="simple",
        arms=[
            TreatmentArm("treatment", 0.5),
            TreatmentArm("control", 0.5),
        ],
        seed=12345,
    )
    
    randomizer = Randomizer(config)
    result = randomizer.validate_randomization(
        simple_dataframe,
        n_simulations=100,
        base_seed=12345,
        verbose=False
    )
    
    # Check result structure
    assert isinstance(result, ValidationResult)
    assert "prob_treatment" in result.assignment_probabilities.columns
    assert "prob_control" in result.assignment_probabilities.columns
    assert "avg_treatment_assignment" in result.assignment_probabilities.columns
    
    # Check summary statistics
    assert "treatment" in result.summary_stats
    assert "control" in result.summary_stats
    
    # For a fair randomization, mean should be close to expected
    treatment_stats = result.summary_stats["treatment"]
    assert abs(treatment_stats["mean"] - 0.5) < 0.05
    assert treatment_stats["expected"] == 0.5
    
    control_stats = result.summary_stats["control"]
    assert abs(control_stats["mean"] - 0.5) < 0.05
    
    # Should be valid
    assert result.is_valid


def test_validate_stratified_randomization(stratified_dataframe):
    """Test validation with stratified randomization."""
    config = RandomizationConfig(
        id_column="id",
        treatment_column="treatment",
        method="stratified",
        strata=["region", "gender"],
        arms=[
            TreatmentArm("treatment", 0.5),
            TreatmentArm("control", 0.5),
        ],
        seed=12345,
    )
    
    randomizer = Randomizer(config)
    result = randomizer.validate_randomization(
        stratified_dataframe,
        n_simulations=100,
        base_seed=12345,
        verbose=False
    )
    
    # Check result validity
    assert isinstance(result, ValidationResult)
    assert result.is_valid
    
    # Check that all observations have probabilities
    assert len(result.assignment_probabilities) == len(stratified_dataframe)
    
    # Probabilities should sum to approximately 1.0 for each observation
    prob_cols = ["prob_treatment", "prob_control"]
    prob_sums = result.assignment_probabilities[prob_cols].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=0.01)


def test_validate_three_arm_randomization(simple_dataframe):
    """Test validation with three treatment arms."""
    config = RandomizationConfig(
        id_column="id",
        treatment_column="treatment",
        method="simple",
        arms=[
            TreatmentArm("treatment1", 0.33),
            TreatmentArm("treatment2", 0.33),
            TreatmentArm("control", 0.34),
        ],
        seed=12345,
    )
    
    randomizer = Randomizer(config)
    result = randomizer.validate_randomization(
        simple_dataframe,
        n_simulations=100,
        base_seed=12345,
        verbose=False
    )
    
    # Check all arms are present
    assert "prob_treatment1" in result.assignment_probabilities.columns
    assert "prob_treatment2" in result.assignment_probabilities.columns
    assert "prob_control" in result.assignment_probabilities.columns
    
    # Check expected proportions
    assert result.summary_stats["treatment1"]["expected"] == 0.33
    assert result.summary_stats["treatment2"]["expected"] == 0.33
    assert result.summary_stats["control"]["expected"] == 0.34
    
    # Check means are close to expected
    assert abs(result.summary_stats["treatment1"]["mean"] - 0.33) < 0.05
    assert abs(result.summary_stats["treatment2"]["mean"] - 0.33) < 0.05
    assert abs(result.summary_stats["control"]["mean"] - 0.34) < 0.05


def test_validate_with_different_seeds(simple_dataframe):
    """Test that validation with different base seeds gives consistent results."""
    config = RandomizationConfig(
        id_column="id",
        treatment_column="treatment",
        method="simple",
        arms=[
            TreatmentArm("treatment", 0.5),
            TreatmentArm("control", 0.5),
        ],
        seed=12345,
    )
    
    randomizer = Randomizer(config)
    
    # Run validation with two different base seeds
    result1 = randomizer.validate_randomization(
        simple_dataframe,
        n_simulations=100,
        base_seed=12345,
        verbose=False
    )
    
    result2 = randomizer.validate_randomization(
        simple_dataframe,
        n_simulations=100,
        base_seed=54321,
        verbose=False
    )
    
    # Both should be valid
    assert result1.is_valid
    assert result2.is_valid
    
    # Means should be similar (within 0.1)
    mean1 = result1.summary_stats["treatment"]["mean"]
    mean2 = result2.summary_stats["treatment"]["mean"]
    assert abs(mean1 - mean2) < 0.1


def test_validate_cluster_randomization():
    """Test validation with cluster randomization."""
    # Create clustered data
    np.random.seed(42)
    df = pd.DataFrame({
        "id": range(100),
        "cluster": np.repeat(range(10), 10),  # 10 clusters, 10 units each
        "age": np.random.randint(18, 65, 100),
    })
    
    config = RandomizationConfig(
        id_column="id",
        treatment_column="treatment",
        method="cluster",
        cluster="cluster",
        arms=[
            TreatmentArm("treatment", 0.5),
            TreatmentArm("control", 0.5),
        ],
        seed=12345,
    )
    
    randomizer = Randomizer(config)
    result = randomizer.validate_randomization(
        df,
        n_simulations=100,
        base_seed=12345,
        verbose=False
    )
    
    # Check result validity
    assert isinstance(result, ValidationResult)
    assert result.is_valid
    
    # For cluster randomization, units in the same cluster should have
    # the same assignment probability
    for cluster_id in df["cluster"].unique():
        cluster_mask = df["cluster"] == cluster_id
        cluster_probs = result.assignment_probabilities.loc[cluster_mask, "prob_treatment"]
        # All units in cluster should have same probability
        assert cluster_probs.std() < 0.01  # Very small std within cluster


def test_validation_detects_minimum_simulations():
    """Test that validation works with minimum number of simulations."""
    np.random.seed(42)
    df = pd.DataFrame({
        "id": range(50),
        "age": np.random.randint(18, 65, 50),
    })
    
    config = RandomizationConfig(
        id_column="id",
        treatment_column="treatment",
        method="simple",
        arms=[
            TreatmentArm("treatment", 0.5),
            TreatmentArm("control", 0.5),
        ],
        seed=12345,
    )
    
    randomizer = Randomizer(config)
    
    # Should work with small number of simulations
    result = randomizer.validate_randomization(
        df,
        n_simulations=10,
        base_seed=12345,
        verbose=False
    )
    
    assert isinstance(result, ValidationResult)
    assert len(result.assignment_probabilities) == 50


def test_validation_probabilities_sum_to_one(simple_dataframe):
    """Test that assignment probabilities sum to approximately 1.0 for each observation."""
    config = RandomizationConfig(
        id_column="id",
        treatment_column="treatment",
        method="simple",
        arms=[
            TreatmentArm("treatment", 0.5),
            TreatmentArm("control", 0.5),
        ],
        seed=12345,
    )
    
    randomizer = Randomizer(config)
    result = randomizer.validate_randomization(
        simple_dataframe,
        n_simulations=100,
        base_seed=12345,
        verbose=False
    )
    
    # For each observation, probabilities should sum to ~1.0
    prob_cols = ["prob_treatment", "prob_control"]
    prob_sums = result.assignment_probabilities[prob_cols].sum(axis=1)
    
    # Check all sums are close to 1.0
    assert np.allclose(prob_sums, 1.0, atol=0.01)


def test_validation_with_unequal_proportions(simple_dataframe):
    """Test validation with unequal treatment proportions."""
    config = RandomizationConfig(
        id_column="id",
        treatment_column="treatment",
        method="simple",
        arms=[
            TreatmentArm("treatment", 0.3),
            TreatmentArm("control", 0.7),
        ],
        seed=12345,
    )
    
    randomizer = Randomizer(config)
    result = randomizer.validate_randomization(
        simple_dataframe,
        n_simulations=100,
        base_seed=12345,
        verbose=False
    )
    
    # Check expected proportions
    assert result.summary_stats["treatment"]["expected"] == 0.3
    assert result.summary_stats["control"]["expected"] == 0.7
    
    # Means should be close to expected proportions
    assert abs(result.summary_stats["treatment"]["mean"] - 0.3) < 0.05
    assert abs(result.summary_stats["control"]["mean"] - 0.7) < 0.05
    
    # Should be valid
    assert result.is_valid
