"""Render a short LaTeX paragraph summarising MDEs."""
from __future__ import annotations

import os
import pandas as pd

HERE = os.path.dirname(__file__)
CSV_PATH = os.path.join(HERE, "power_summary.csv")
OUT_TEX = os.path.join(HERE, "mde_paragraph.tex")

ORDER = ["labor_part", "entrep_engage", "agency_prop", "aspirations_gap_w", "aspirations_gap"]
LABELS = {
    "labor_part": "labour participation",
    "entrep_engage": "entrepreneurial engagement",
    "agency_prop": "sense of agency (SD)",
    "aspirations_gap_w": "aspirations gap (SD)",
    "aspirations_gap": "aspirations gap (SD)",
}


def write_text(text: str) -> None:
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")


def placeholder() -> None:
    write_text("MDEs pending computation; run the power analysis to populate exact values.")
    print(f"power_summary.csv not found. Wrote placeholder paragraph to {OUT_TEX}")


def main() -> None:
    if not os.path.exists(CSV_PATH):
        placeholder()
        return

    df = pd.read_csv(CSV_PATH)
    df = df[df["outcome"].isin(ORDER)].copy()
    df["order"] = df["outcome"].apply(lambda x: ORDER.index(x))
    df = df.sort_values("order")

    parts: list[str] = []
    for _, row in df.iterrows():
        label = LABELS.get(row["outcome"], row["outcome"])
        if row["outcome"] in ("labor_part", "entrep_engage"):
            mde_pp = 100.0 * float(row["MDE"])
            parts.append(f"{label}: {mde_pp:.1f} pp")
        else:
            parts.append(f"{label}: {float(row['MDE']):.2f}")

    paragraph = (
        "Based on PSPS baseline moments and barangay-level ICCs, the design (8 clusters per arm, $m\\approx 18$, 80\% power, $\\alpha=0.05$) yields minimum detectable differences of "
        + "; ".join(parts[:-1])
        + "; and "
        + parts[-1]
        + "."
    )

    write_text(paragraph)
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
