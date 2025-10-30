import pandas as pd

from rct_field_flow.randomize import RandomizationConfig, Randomizer, TreatmentArm


def _demo_df():
    return pd.DataFrame(
        {
            "participant_id": range(1, 9),
            "district": ["north"] * 4 + ["south"] * 4,
            "gender": ["M", "F"] * 4,
            "age": [20, 21, 22, 23, 40, 41, 42, 43],
        }
    )


def test_stratified_assignments_respect_proportions():
    df = _demo_df()
    cfg = RandomizationConfig(
        id_column="participant_id",
        treatment_column="assignment",
        method="stratified",
        arms=[TreatmentArm("cash", 0.5), TreatmentArm("control", 0.5)],
        strata=["district"],
        balance_covariates=["age"],
        iterations=1,
        seed=123,
        use_existing_assignment=False,
    )
    result = Randomizer(cfg).run(df)
    assigned = result.assignments
    for _, group in assigned.groupby("district"):
        counts = group["assignment"].value_counts().to_dict()
        assert counts["cash"] == counts["control"] == len(group) // 2


def test_rerandomization_improves_min_pvalue():
    df = _demo_df()
    cfg_1 = RandomizationConfig(
        id_column="participant_id",
        treatment_column="assignment",
        method="simple",
        arms=[TreatmentArm("cash", 0.5), TreatmentArm("control", 0.5)],
        balance_covariates=["age"],
        iterations=1,
        seed=2024,
        use_existing_assignment=False,
    )
    cfg_10 = RandomizationConfig(
        id_column=cfg_1.id_column,
        treatment_column=cfg_1.treatment_column,
        method=cfg_1.method,
        arms=list(cfg_1.arms),
        strata=cfg_1.strata,
        cluster=cfg_1.cluster,
        balance_covariates=cfg_1.balance_covariates,
        iterations=10,
        seed=cfg_1.seed,
        use_existing_assignment=cfg_1.use_existing_assignment,
    )
    result_one = Randomizer(cfg_1).run(df)
    result_many = Randomizer(cfg_10).run(df)
    assert result_many.best_min_pvalue >= result_one.best_min_pvalue


def test_existing_assignment_skips_randomization():
    df = _demo_df()
    df["treatment"] = ["cash", "control"] * 4
    cfg = RandomizationConfig(
        id_column="participant_id",
        treatment_column="treatment",
        method="simple",
        arms=[TreatmentArm("cash", 0.5), TreatmentArm("control", 0.5)],
        balance_covariates=["age"],
        iterations=5,
        seed=100,
        use_existing_assignment=True,
    )
    result = Randomizer(cfg).run(df)
    assert result.used_existing_assignment
    assert result.iterations == 0
