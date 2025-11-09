"""Test the enhanced Stata code generation with rerandomization loop."""
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rct_field_flow.randomize import RandomizationConfig, TreatmentArm
from rct_field_flow.app import generate_stata_randomization_code

# Test with rerandomization
config = RandomizationConfig(
    id_column='barangay_id',
    treatment_column='treatment',
    method='stratified',
    arms=[
        TreatmentArm('control', 0.3333),
        TreatmentArm('treatment_A', 0.3333),
        TreatmentArm('treatment_B', 0.3334)
    ],
    strata=['province', 'coastal_dummy'],
    balance_covariates=['barangay_area', 'population_2020', 'pop_dens_2020'],
    seed=20250128,
    iterations=10000
)

print("=" * 80)
print("STATA CODE WITH RERANDOMIZATION LOOP")
print("=" * 80)

stata_code = generate_stata_randomization_code(config, 'stratified')

# Save to file
with open("test_stata_rerand.do", "w", encoding='utf-8') as f:
    f.write(stata_code)

print(stata_code)
print("\n" + "=" * 80)
print(f"✅ Code saved to test_stata_rerand.do ({len(stata_code)} characters)")
