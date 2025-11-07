from .analyze import attrition_table, estimate_ate, heterogeneity_analysis, one_click_analysis
from .assign_cases import assign_cases
from .backcheck import sample_backchecks
from .flag_quality import flag_all
from .randomize import (
    RandomizationConfig,
    RandomizationResult,
    Randomizer,
    TreatmentArm,
)

# Optional import - report module requires weasyprint with native dependencies
try:
    from .report import generate_weekly_report
except ImportError as e:
    generate_weekly_report = None
    import warnings
    warnings.warn(
        f"Report module unavailable: {e}. Install GTK libraries for PDF generation: "
        "https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows",
        ImportWarning
    )

__all__ = [
    "RandomizationConfig",
    "RandomizationResult",
    "Randomizer",
    "TreatmentArm",
    "assign_cases",
    "flag_all",
    "sample_backchecks",
    "generate_weekly_report",
    "estimate_ate",
    "heterogeneity_analysis",
    "attrition_table",
    "one_click_analysis",
]
__version__ = "0.1.0"
