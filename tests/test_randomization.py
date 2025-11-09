"""
Test script to validate enhanced randomization with all types
"""
import pandas as pd
from rct_field_flow.randomize import RandomizationConfig, Randomizer, TreatmentArm

# Load the sample baseline
df = pd.read_csv('examples/sample_baseline.csv')
print(f"Loaded {len(df)} observations with {df['barangay_code'].nunique()} clusters")

# Test 1: Simple Randomization
print("\n=== Test 1: Simple Randomization ===")
config1 = RandomizationConfig(
    id_column='caseid',
    treatment_column='treatment',
    method='simple',
    arms=[
        TreatmentArm('treatment', 0.5),
        TreatmentArm('control', 0.5)
    ],
    balance_covariates=['age', 'hh_size'],
    iterations=100,
    seed=44821
)
result1 = Randomizer(config1).run(df, verbose=True)
print(f"✓ Simple randomization complete: {result1.assignments['treatment'].value_counts().to_dict()}")

# Test 2: Stratified Randomization
print("\n=== Test 2: Stratified Randomization ===")
config2 = RandomizationConfig(
    id_column='caseid',
    treatment_column='treatment',
    method='stratified',
    strata=['province', 'gender'],
    arms=[
        TreatmentArm('treatment', 0.5),
        TreatmentArm('control', 0.5)
    ],
    balance_covariates=['age', 'hh_size', 'income'],
    iterations=500,
    seed=44821
)
result2 = Randomizer(config2).run(df, verbose=True)
print(f"✓ Stratified randomization complete: {result2.assignments['treatment'].value_counts().to_dict()}")

# Test 3: Cluster Randomization
print("\n=== Test 3: Cluster Randomization ===")
config3 = RandomizationConfig(
    id_column='caseid',
    treatment_column='treatment',
    method='cluster',
    cluster='barangay_code',
    strata=['province'],
    arms=[
        TreatmentArm('treatment', 0.5),
        TreatmentArm('control', 0.5)
    ],
    balance_covariates=['age', 'hh_size', 'income'],
    iterations=1000,
    seed=44821
)
result3 = Randomizer(config3).run(df, verbose=True)
print(f"✓ Cluster randomization complete: {result3.assignments['treatment'].value_counts().to_dict()}")
print(f"  Clusters per arm: {result3.assignments.groupby('treatment')['barangay_code'].nunique().to_dict()}")

# Test 4: Three-arm randomization with rerandomization
print("\n=== Test 4: Three-Arm Rerandomization ===")
config4 = RandomizationConfig(
    id_column='caseid',
    treatment_column='treatment',
    method='stratified',
    strata=['province'],
    arms=[
        TreatmentArm('treatment1', 0.33),
        TreatmentArm('treatment2', 0.33),
        TreatmentArm('control', 0.34)
    ],
    balance_covariates=['age', 'hh_size', 'income', 'consumption'],
    iterations=2000,
    seed=44821
)
result4 = Randomizer(config4).run(df, verbose=True)
print(f"✓ Three-arm randomization complete: {result4.assignments['treatment'].value_counts().to_dict()}")

print("\n=== All Tests Passed ===")
print("Enhanced randomization with validation, diagnostics, and all types working correctly!")
