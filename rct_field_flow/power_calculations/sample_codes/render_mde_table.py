"""Render LaTeX MDE table from power_summary.csv."""
from __future__ import annotations

import os
from typing import Iterable

import pandas as pd

HERE = os.path.dirname(__file__)
CSV_PATH = os.path.join(HERE, "power_summary.csv")
OUT_TEX = os.path.join(HERE, "mde_table.tex")

ROW_END = " \\\\"


def write_lines(lines: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def placeholder_lines() -> list[str]:
    return [
        "% power_summary.csv not found; placeholder table written. Run power step first.",
        "\\begin{table}[h!]",
        "  \\centering",
        "  \\caption{Baseline moments, ICCs, and MDEs (pending)}",
        "  \\label{tab:mde}",
        "  \\begin{tabular}{lrrrrr}",
        "    \\toprule",
        f"    Outcome & Mean & ICC & DE & MDE & Notes{ROW_END}",
        "    \\midrule",
        f"    Labour participation & -- & -- & -- & -- & 30-day paid work (0/1){ROW_END}",
        "    \\midrule",
        f"    Entrepreneurial engagement & -- & -- & -- & -- & 30-day self-employment (0/1){ROW_END}",
        "    \\midrule",
        f"    Sense of agency (index) & -- & -- & -- & -- & SD units{ROW_END}",
        "    \\midrule",
        f"    Aspirations gap (continuous) & -- & -- & -- & -- & SD units{ROW_END}",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]


def format_row(row: pd.Series):
    outcome_map = {
        "labor_part": "Labour participation",
        "entrep_engage": "Entrepreneurial engagement",
        "agency_prop": "Sense of agency (index)",
        "agency_index": "Sense of agency (index)",
        "aspirations_gap_w": "Aspirations gap (continuous)",
        "aspirations_gap": "Aspirations gap (continuous)",
    }
    name = outcome_map.get(row.get("outcome"), row.get("outcome", ""))
    mean = row.get("mean")
    icc = row.get("icc")
    de = row.get("DE")
    mde = row.get("MDE")
    notes = "SD units"
    if row.get("outcome") in ("labor_part", "entrep_engage"):
        try:
            mde_val = 100.0 * float(mde)
            mde_str = f"{mde_val:.2f}"
        except Exception:
            mde_str = "--"
        notes = "percentage points"
    else:
        try:
            mde_str = f"{float(mde):.2f}"
        except Exception:
            mde_str = "--"

    def fmt(value, template):
        try:
            return template.format(float(value))
        except Exception:
            return "--"

    mean_str = fmt(mean, "{0:0.2f}")
    icc_str = fmt(icc, "{0:0.3f}")
    de_str = fmt(de, "{0:0.2f}")
    return name, mean_str, icc_str, de_str, mde_str, notes


def main() -> None:
    if not os.path.exists(CSV_PATH):
        write_lines(placeholder_lines())
        print(f"power_summary.csv not found. Wrote placeholder to {OUT_TEX}")
        return

    df = pd.read_csv(CSV_PATH)
    order = ["labor_part", "entrep_engage", "agency_prop", "aspirations_gap_w", "aspirations_gap"]
    df["order"] = df["outcome"].apply(lambda x: order.index(x) if x in order else 99)
    df = df.sort_values("order")

    lines = [
        "% Auto-generated from power_summary.csv",
        "\\begin{table}[h!]",
        "  \\centering",
        "  \\caption{Baseline moments, ICCs, and MDEs (PSPS-based)}",
        "  \\label{tab:mde}",
        "  \\begin{tabular}{lrrrrr}",
        "    \\toprule",
        f"    Outcome & Mean & ICC & DE & MDE & Notes{ROW_END}",
        "    \\midrule",
    ]

    for _, row in df.iterrows():
        name, mean_str, icc_str, de_str, mde_str, notes = format_row(row)
        lines.append(f"    {name} & {mean_str} & {icc_str} & {de_str} & {mde_str} & {notes}{ROW_END}")
        lines.append("    \\midrule")

    if lines[-1].strip() == "\\midrule":
        lines.pop()

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])

    write_lines(lines)
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
