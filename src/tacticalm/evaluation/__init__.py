"""Evaluation framework for TacticaLM.

Provides automated metrics for factual accuracy, numerical accuracy,
unsupported claim detection, and tactical specificity scoring.

Note: run_baseline requires torch and is imported lazily to avoid
DLL/import issues on systems without a working GPU torch install.
"""

from .factuality import compute_factuality_score
from .numerical import compute_numerical_accuracy
from .unsupported_claims import compute_unsupported_rate
from .specificity import score_tactical_specificity
from .report import aggregate_results, generate_comparison_table, export_report

__all__ = [
    "compute_factuality_score",
    "compute_numerical_accuracy",
    "compute_unsupported_rate",
    "score_tactical_specificity",
    "aggregate_results",
    "generate_comparison_table",
    "export_report",
]
