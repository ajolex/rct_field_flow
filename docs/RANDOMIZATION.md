# Randomization Guide: How to Achieve Balance

**Authors:** Martin Sweeney (updated by Matt White and Kelsey Larson)  
**Date:** October 15, 2013 (updated July 27, 2018)

---

## Table of Contents

1. [General Randomization Tips](#general-randomization-tips)
2. [Types of Randomization](#types-of-randomization)
3. [The Rerandomization Technique](#the-rerandomization-technique)
4. [Implementation Examples](#implementation-examples)
5. [References](#references)

---

## General Randomization Tips

An error in randomization coding can badly harm the quality of a randomized control trial – the presumption of random assignment is the **most important presumption** that RCT analysis makes. When coding a randomization, you should follow these best practices:

### 1. **Use Prewritten Commands When Possible**

Use established programs like `randtreat` (available on the IPA GitHub) to handle most types of randomization commonly used in field experiments. Working with an established program can help **reduce the risk of coding errors**, so please use `randtreat` unless you have an extraordinary circumstance that makes using the program impossible.

### 2. **Use "Assert" and "Confirm" Liberally**

Check each step to make sure the code is producing exactly what you wanted. `assert` and `confirm` will throw an error if the code ever breaks your checks.

**Example:**

```stata
count if treatment == 1
assert `r(N)' == 50 // the treatment group should contain exactly 50 people at this stage.
```

### 3. **Run Your Randomization Multiple Times**

Run your randomization a few hundred times with different seeds and compare the outcomes. In most randomizations, all units should have an approximately similar chance of being assigned into the treatment and control group.

**Validation approach:**

- If 30% of the sample will be assigned to treatment, each observation should be assigned to the treatment group in approximately 30% of the randomizations you run
- Rerun the treatment assignment 500 times, creating a variable that is the mean of randomization 1-500 for each observation
- Graph it with `hist avg_treat_assignment`
- If that histogram does NOT look like a binomial distribution – if some observations are almost always in treatment or almost always in control – check your randomization code

---

## Types of Randomization

Economists debate the best methods for randomization. Here's an overview of the main types. See Chapter 4 of *Running Randomized Evaluations: A Practical Guide* by Glennerster and Takavarasha for an in-depth discussion.

### Simple Randomization

Each unit has an equal and independently assigned probability of being assigned to each treatment arm.

**Implementation:** Generate a random number, sort by that number, and then assign the first n observations to treatment and the rest to control.

| **Pros** | **Cons** |
|----------|----------|
| "Truly random" | Risk of substantial differences in randomly sorted groups |
| Strong statistical assumptions | Undermines ability to attribute changes to treatment |
| | Almost never used unless baseline data is lacking |

### Stratified Randomization

The sample is split into strata defined by important variables, and we randomize within each stratum.

| **Pros** | **Cons** |
|----------|----------|
| Enforces balance along key variables | Requires binning continuous variables |
| Transparent and easy to explain | Can only handle limited number of variables |
| Works well with randomization inference | Need to handle odd-sized strata |
| Widely accepted | |

### Cluster Randomization

Randomization at the group level (e.g., schools, villages).

| **Pros** | **Cons** |
|----------|----------|
| Avoids spillovers | Reduces statistical power |
| Often easier logistically | Need larger sample or accept less precision |

### Clustered and Stratified Randomization

Combines both cluster and stratified approaches for complex designs.

### Rerandomization

Run your randomization 10,000+ times, test each for balance, and keep the most balanced one.

| **Pros** | **Cons** |
|----------|----------|
| Achieves excellent balance on many variables | Complicates randomization inference |
| Leads to tighter standard errors | May skew assignment probabilities for some units |
| Reviewer-friendly balance tables | **Always check** that no unit is systematically favored |

---

## The Rerandomization Technique

This guide highlights a method featured in **Bruhn & McKenzie (2009)** that is widely used in field experiments.

### Key Concept

Instead of accepting a single "good enough" randomization, we run **thousands (10,000–100,000)** and select the one with the **best balance** (highest p-value or highest minimum p-value across multiple variables).

### ⚠️ Important Caveat

**Rerandomization can break the assumptions needed for randomization inference.** If PIs want to use randomization inference (RI), prefer simple stratified randomization.

### Recommended Approach

**First choice:** Use the `randtreat` command — it's battle-tested and does rerandomization automatically.

If you must write it yourself, see the [Implementation Examples](#implementation-examples) below.

---

## Implementation Examples

Here are three fully working Stata examples for different scenarios:

### Example 1: One Continuous Balance Variable (Maximize p-value)

**Scenario:** Stratifies on `highaccess` and `marketed2013`; balances on `compcount`

```stata
// stratifies on: highaccess, marketed2013 
// balances on: compcount 

isid Communi2                        // assert unique ID
sort Communi2                        // sort on unique ID
version 12.1: set seed 528713  
gen double rand = .                  
gen double rand2 = . 
gen treatment_num = .
gen besttreatment_num = .
loc p 0                              
loc bestp 0 

quietly forv x=1/10000 { 
    sort Communi2  
    replace rand = runiform() 
    replace rand2 = runiform() 
    isid rand rand2                  // assert no duplicates
    sort highaccess marketed2013 rand rand2 

    by highaccess marketed2013 (rand rand2): ///
    replace treatment_num = ///
        cond(_n <  .32*_N, 1, ///
        cond(_n >= .32*_N & _n < .51*_N, 2, ///
        cond(_n >= .51*_N & _n < .70*_N, 3, ///
        cond(_n >= .70*_N & _n <= _N, 4, .))))

    // Test balance
    reg compcount i.treatment_num 
    testparm i.treatment_num 
    loc p = r(p) 

    if `p' > `bestp' {  
        loc bestp = `p'  
        replace besttreatment_num = treatment_num 
    } 
    noisily display `x' 
}
display "Best p-value: `bestp'"
// Final assignments in besttreatment_num
```

---

### Example 2: Multiple Continuous Variables (Maximize Minimum p-value)

**Scenario:** Stratifies on `highaccess` and `marketed2013`; balances on `compcount` and `hhsize_av`

```stata
// stratifies on: highaccess, marketed2013 
// balances on: compcount, hhsize_av  

isid Communi2  
sort Communi2  
version 12.1: set seed 528713  
gen double rand = .  
gen double rand2 = . 
gen treatment_num = .
gen mintreatment_num = .
loc p1 0 
loc p2 0 
loc minp 0 
loc minp1 0 
loc minp2 0 

quietly forv x=1/10000 { 
    sort Communi2  
    replace rand = runiform() 
    replace rand2 = runiform() 
    isid rand rand2  
    sort highaccess marketed2013 rand rand2 

    by highaccess marketed2013 (rand rand2): ///
    replace treatment_num = ///
        cond(_n <  .32*_N, 1, ///
        cond(_n >= .32*_N & _n < .51*_N, 2, ///
        cond(_n >= .51*_N & _n < .70*_N, 3, ///
        cond(_n >= .70*_N & _n <= _N, 4, .))))

    reg compcount i.treatment_num 
    testparm i.treatment_num 
    loc p1 = r(p) 

    reg hhsize_av i.treatment_num 
    testparm i.treatment_num 
    loc p2 = r(p) 

    if min(`p1', `p2') > `minp' { 
        loc minp1 = `p1' 
        loc minp2 = `p2' 
        loc minp = min(`minp1', `minp2') 
        replace mintreatment_num = treatment_num 
    } 
    noisily display `x' 
}
display "p1: `minp1'" 
display "p2: `minp2'"
```

---

### Example 3: Many Strata + Uneven Cell Sizes (Equal-sized Treatment Groups)

**Scenario:** 3 arms (0=control, 1=T1, 2=T2), 60 strata

```stata
// Example: 3 arms (0=control, 1=T1, 2=T2), 60 strata
egen strata = group(access c4d_treatment_num low_behavior low_knowledge) 
tab strata, missing   // 60 strata

gen m4d_treatment = .
gen best_treatment = .
loc minp 0

quietly forv x = 1/10000 {
    sort household_id
    gen rand  = runiform()
    gen rand2 = runiform()
    isid rand rand2
    sort strata rand rand2

    foreach q of numlist 1/60 {
        loc assign = floor(runiform()*100000)
        by strata (rand rand2): replace m4d_treatment = mod(`assign',3)     if strata==`q' & _n <= _N/3
        by strata (rand rand2): replace m4d_treatment = mod(`assign'+1,3) if strata==`q' & _n > _N/3 & _n <= 2*_N/3
        by strata (rand rand2): replace m4d_treatment = mod(`assign'+2,3) if strata==`q' & _n > 2*_N/3
    }

    // Balance checks (example)
    reg compcount i.m4d_treatment
    testparm i.m4d_treatment
    loc p1 = r(p)
    reg hhsize_av i.m4d_treatment
    testparm i.m4d_treatment
    loc p2 = r(p)

    if min(`p1',`p2') > `minp' {
        loc minp = min(`p1',`p2')
        replace best_treatment = m4d_treatment
    }
    noisily display `x'
}
```

**Key Feature:** This method ensures **perfectly equal group sizes** even when strata sizes aren't divisible by 3.

---

## References

Bruhn, Miriam, and David McKenzie. 2009. "In Pursuit of Balance: Randomization in Practice in Development Field Experiments." *American Economic Journal: Applied Economics*, 1(4): 200–232.

Glennerster, Rachel, and Kudzai Takavarasha. 2013. *Running Randomized Evaluations: A Practical Guide*. Princeton: Princeton University Press. [Chapter 4 recommended]

---

## Quick Checklist

- [ ] Using `randtreat` or other established randomization program?
- [ ] Using `assert` and `confirm` statements liberally?
- [ ] Testing randomization with 100s of different seeds?
- [ ] Checking that observations have ~equal probability across arms?
- [ ] If using rerandomization, verified no systematic bias?
- [ ] Balance table prepared for publication?
- [ ] Randomization code documented and reproducible?

