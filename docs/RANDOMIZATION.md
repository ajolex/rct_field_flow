# Randomization Guide

This document explains the randomization methods implemented in RCT Field Flow, following best practices from Bruhn & McKenzie (2009) and IPA/J-PAL guidelines.

## Overview

RCT Field Flow supports multiple randomization methods designed to balance statistical rigor with practical field requirements:

1. **Simple Randomization** - Each unit has equal, independent probability
2. **Stratified Randomization** - Randomize within pre-defined strata
3. **Cluster Randomization** - Randomize at group level (villages, schools, etc.)
4. **Cluster + Stratified** - Combine both approaches
5. **Rerandomization** - Run many iterations and select the most balanced

## Randomization Methods

### Simple Randomization

Each unit is independently assigned with equal probability to treatment/control.

**Pros:**
- "Truly random" - strongest econometric assumptions
- Simple to implement and explain

**Cons:**
- Risk of imbalance on important covariates
- Rarely used unless no baseline data available

**Example config:**
```yaml
randomization:
  id_column: "participant_id"
  method: "simple"
  arms:
    - name: "treatment"
      proportion: 0.5
    - name: "control"
      proportion: 0.5
```

### Stratified Randomization

Split sample into strata based on important variables, then randomize within each stratum.

**Pros:**
- Enforces balance on stratification variables
- Transparent and widely accepted
- Works well with randomization inference
- Can have multiple treatment arms

**Cons:**
- Need to bin continuous variables
- Limited number of stratification variables (to keep strata sizes manageable)
- Must handle odd-sized strata

**Example config:**
```yaml
randomization:
  id_column: "caseid"
  method: "stratified"
  strata: ["province", "gender"]
  arms:
    - name: "treatment"
      proportion: 0.5
    - name: "control"
      proportion: 0.5
```

**Handling Uneven Strata:**
The code uses a "largest remainder" method to distribute units when strata sizes aren't perfectly divisible by the number of arms. This ensures treatment group sizes are as equal as possible.

### Cluster Randomization

Randomize at the cluster level (e.g., villages, schools) rather than individual level.

**Pros:**
- Avoids spillover effects
- Often easier logistically
- Can combine with stratification

**Cons:**
- Reduces statistical power
- Need larger samples or accept less precision
- Must account for intra-cluster correlation in analysis

**Example config:**
```yaml
randomization:
  id_column: "caseid"
  method: "cluster"
  cluster: "barangay_code"
  strata: ["province"]  # optional: stratify clusters
  arms:
    - name: "treatment"
      proportion: 0.5
    - name: "control"
      proportion: 0.5
```

### Rerandomization

Run the randomization thousands of times, test each for balance, and keep the most balanced assignment.

**Pros:**
- Achieves excellent balance on many variables
- Leads to tighter standard errors
- Produces reviewer-friendly balance tables
- Can balance on continuous variables without binning

**Cons:**
- **Complicates randomization inference** - can invalidate standard inference methods
- May skew assignment probabilities for some units
- Computationally intensive

**Example config:**
```yaml
randomization:
  id_column: "caseid"
  method: "stratified"
  strata: ["province"]
  balance_covariates: ["age", "hh_size", "income"]
  iterations: 10000  # Test 10,000 randomizations
  seed: 44821
  arms:
    - name: "treatment"
      proportion: 0.5
    - name: "control"
      proportion: 0.5
```

**Important Notes on Rerandomization:**

1. **Randomization Inference**: When using >1000 iterations, standard randomization inference may not be valid. Consider using robust standard errors or alternative inference methods.

2. **Assignment Probability**: Always validate that no units are systematically favored. Use the `--verbose` flag to see diagnostic checks:
   ```bash
   rct-field-flow randomize --baseline data.csv --verbose
   ```

3. **P-value Selection**: The code maximizes the *minimum* p-value across all balance covariates. This is more conservative than maximizing a joint test statistic.

## Best Practices

### 1. Use Assertions Liberally

The code includes built-in validation:
- Checks for duplicate IDs
- Validates treatment group sizes match expected proportions
- Monitors assignment probability distributions

### 2. Test Your Randomization

After randomization, always:
- Check balance table for all important covariates
- Verify treatment group sizes
- If using rerandomization, examine the diagnostic output

### 3. Document Your Process

Always record:
- Seed used (for reproducibility)
- Number of iterations
- Final minimum p-value
- Any units excluded and why

### 4. Handling Missing Data

- Missing values in stratification variables create a separate stratum
- Missing values in balance covariates are excluded from that covariate's test
- Consider imputing or excluding observations with extensive missingness

## Command Line Usage

### Basic Randomization
```bash
rct-field-flow randomize \
  --baseline examples/sample_baseline.csv \
  --output randomized.csv
```

### With Verbose Diagnostics
```bash
rct-field-flow randomize \
  --baseline examples/sample_baseline.csv \
  --output randomized.csv \
  --verbose \
  --balance-table balance_results.csv
```

### Key CLI Options

- `--verbose, -v`: Show detailed progress and diagnostic information
- `--balance-table PATH`: Save balance test results to CSV
- `--config-path PATH`: Use custom configuration file

## Validation Checks

The randomization includes several automatic validation checks:

### 1. Treatment Distribution
Ensures that actual treatment group sizes match expected proportions within a tolerance of 1 unit per arm.

### 2. Assignment Probability (Rerandomization Only)
When running >100 iterations, tracks how often each unit is assigned to each arm. Warns if:
- Average probability deviates >5% from expected
- Some units are systematically favored

### 3. Balance Tests
For each balance covariate:
- Performs ANOVA F-test across treatment arms
- Reports p-value and group means
- Tracks minimum p-value across all covariates

## Troubleshooting

### "Duplicate IDs found"
- Check that your ID column has unique values
- Remove duplicate rows or fix data entry errors

### "Treatment arm sizes don't match expected"
- May occur with small samples and exact proportions
- Check if stratification is creating very small strata
- Consider collapsing strata or using fewer stratification variables

### "Some units systematically favored"
- Appears when rerandomization creates biased assignment probabilities
- Solutions:
  - Reduce number of iterations
  - Add stratification to enforce balance
  - Consider if rerandomization is necessary

### Low p-values after many iterations
- If best p-value is still low (<0.10) after 10,000 iterations:
  - Sample may be genuinely imbalanced
  - Consider adding stratification
  - Check for data issues or outliers

## References

- Bruhn, Miriam, and David McKenzie. 2009. "In Pursuit of Balance: Randomization in Practice in Development Field Experiments." *American Economic Journal: Applied Economics*, 1(4): 200–232.
- Glennerster, Rachel, and Kudzai Takavarasha. 2013. *Running Randomized Evaluations: A Practical Guide*. Princeton University Press.
- IPA Research Resources: Randomization Guide (2018)

## Examples

See `examples/` directory for:
- `sample_baseline.csv` - Generated test dataset with 450 clusters
- `config_example.yaml` - Example configuration for different randomization types
- Test scripts demonstrating each method

## Technical Details

### Algorithm for Rerandomization

1. Generate random seed sequence from base seed
2. For each iteration i:
   - Set seed[i]
   - Generate assignment using selected method
   - Compute balance tests for all covariates
   - Calculate minimum p-value
   - If min_p > best_min_p, save this assignment
3. Return assignment with highest minimum p-value

### Handling Odd-Sized Strata

Uses the "largest remainder" method:
1. Calculate exact target counts: n * proportion
2. Assign floor(target) to each arm
3. Compute remainders: target - floor(target)
4. Distribute remaining units to arms with largest remainders

This ensures fair distribution and reproducibility.
