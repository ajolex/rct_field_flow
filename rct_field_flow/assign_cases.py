from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class TeamRule:
    name: str
    match: Dict[str, Iterable[str]]
    quota: Optional[int] = None

    def matches(self, row: pd.Series) -> bool:
        for column, raw_values in self.match.items():
            values = list(raw_values) if isinstance(raw_values, (list, tuple, set)) else [raw_values]
            if row.get(column) not in values:
                return False
        return True


@dataclass
class CaseAssignmentConfig:
    case_id_column: str
    label_template: str
    team_column: Optional[str] = None
    default_team: str = "unassigned"
    team_rules: List[TeamRule] = field(default_factory=list)
    form_ids: Dict[str, Iterable[str]] = field(default_factory=dict)
    additional_columns: List[str] = field(default_factory=list)
    treatment_column: str = "treatment"
    form_separator: str = ","

    @classmethod
    def from_dict(cls, data: Dict) -> "CaseAssignmentConfig":
        rules = [
            TeamRule(name=rule["name"], match=rule.get("match", {}), quota=rule.get("quota"))
            for rule in data.get("team_rules", [])
        ]
        return cls(
            case_id_column=data["case_id_column"],
            label_template=data.get("label_template", "{id}"),
            team_column=data.get("team_column"),
            default_team=data.get("default_team", "unassigned"),
            team_rules=rules,
            form_ids=data.get("form_ids", {}),
            additional_columns=data.get("additional_columns", []),
            treatment_column=data.get("treatment_column", "treatment"),
            form_separator=data.get("form_separator", ","),
        )

    def form_list_for(self, treatment: str) -> List[str]:
        if treatment in self.form_ids:
            raw = self.form_ids[treatment]
        else:
            raw = self.form_ids.get("default", [])
        if isinstance(raw, str):
            return [raw]
        return list(raw)


def assign_cases(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Assign SurveyCTO cases to enumerator teams."""
    cfg = CaseAssignmentConfig.from_dict(config)
    work_df = df.copy()

    if cfg.case_id_column not in work_df.columns:
        raise KeyError(f"{cfg.case_id_column} not found in dataframe.")

    def resolve_team(row: pd.Series) -> str:
        if cfg.team_column and pd.notna(row.get(cfg.team_column)):
            return str(row[cfg.team_column])
        for rule in cfg.team_rules:
            if rule.matches(row):
                return rule.name
        return cfg.default_team

    def build_label(row: pd.Series) -> str:
        context = row.to_dict()
        context.setdefault("id", row.get(cfg.case_id_column))
        try:
            return cfg.label_template.format(**context)
        except KeyError:
            return str(row.get(cfg.case_id_column))

    def build_formids(row: pd.Series) -> str:
        treatment_value = row.get(cfg.treatment_column)
        forms = cfg.form_list_for(str(treatment_value) if treatment_value is not None else "default")
        return cfg.form_separator.join(forms)

    work_df = work_df.assign(
        id=work_df[cfg.case_id_column],
        label=work_df.apply(build_label, axis=1),
        users=work_df.apply(resolve_team, axis=1),
        formids=work_df.apply(build_formids, axis=1),
    )

    columns = ["id", "label", "users", "formids"] + cfg.additional_columns
    existing_columns = [c for c in columns if c in work_df.columns]
    case_roster = work_df[existing_columns].copy()

    return case_roster
