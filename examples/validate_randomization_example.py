"""
Example: Validate Randomization Fairness

This script demonstrates how to use the randomization validation feature
to verify that no observations are systematically favored for treatment
or control assignment.

Following the best practice from RANDOMIZATION.md: "Run your randomization 
a few hundred times with different seeds and compare the outcomes."
"""

import pandas as pd
import matplotlib.pyplot as plt
from rct_field_flow.randomize import RandomizationConfig, Randomizer, TreatmentArm

# Load sample baseline data
print("Loading sample baseline data...")
df = pd.read_csv('examples/sample_baseline.csv')
print(f"Loaded {len(df)} observations")

# Configure randomization
config = RandomizationConfig(
    id_column='caseid',
    treatment_column='treatment',
    method='stratified',
    strata=['province', 'gender'],
    arms=[
        TreatmentArm('treatment', 0.5),
        TreatmentArm('control', 0.5)
    ],
    balance_covariates=['age', 'hh_size', 'income'],
    iterations=100,  # For actual randomization
    seed=12345
)

# Create randomizer
randomizer = Randomizer(config)

# Validate randomization fairness
print("\nValidating randomization fairness...")
print("Running 500 simulations with different seeds...")
validation_result = randomizer.validate_randomization(
    df,
    n_simulations=500,
    base_seed=12345,
    verbose=True
)

# Check validation status
print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)

if validation_result.is_valid:
    print("✓ PASS: Randomization appears fair")
else:
    print("✗ FAIL: Issues detected with randomization")
    print("\nWarnings:")
    for warning in validation_result.warnings:
        print(f"  - {warning}")

# Display summary statistics
print("\nAssignment Probability Statistics:")
for arm, stats in validation_result.summary_stats.items():
    print(f"\n{arm}:")
    print(f"  Expected:    {stats['expected']:.3f}")
    print(f"  Mean:        {stats['mean']:.4f}")
    print(f"  Std Dev:     {stats['std']:.4f}")
    print(f"  Range:       [{stats['min']:.4f}, {stats['max']:.4f}]")

# Create visualization
print("\nGenerating histogram...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, arm in enumerate(validation_result.summary_stats.keys()):
    prob_col = f"prob_{arm}"
    probs = validation_result.assignment_probabilities[prob_col].values
    expected = validation_result.summary_stats[arm]["expected"]
    
    axes[idx].hist(probs, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[idx].axvline(expected, color='red', linestyle='--', linewidth=2, 
                     label=f'Expected ({expected:.2f})')
    axes[idx].set_xlabel(f'Probability of {arm} assignment')
    axes[idx].set_ylabel('Number of observations')
    axes[idx].set_title(f'{arm.title()} Assignment Probabilities')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Randomization Validation: Assignment Probability Distribution\n(500 simulations)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('validation_histogram.png', dpi=150, bbox_inches='tight')
print("✓ Histogram saved to: validation_histogram.png")

# Save detailed probabilities
validation_result.assignment_probabilities.to_csv('validation_probabilities.csv', index=False)
print("✓ Assignment probabilities saved to: validation_probabilities.csv")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("""
The histogram should look like a binomial distribution centered around
the expected proportion for each treatment arm.

✓ GOOD: Most observations cluster around the expected proportion with
        a symmetric, bell-shaped distribution

✗ BAD:  Some observations are almost always in treatment or almost
        always in control (bimodal or skewed distribution)

If the distribution looks problematic, check your randomization code
for issues that might cause systematic bias.
""")

print("\nValidation complete!")
