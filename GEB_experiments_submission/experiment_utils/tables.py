"""Generate the two density-estimation LaTeX tables from saved CSV summaries."""

from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils.config import DENSITY_RESULTS_ROOT, N_REPETITIONS


SIMULATIONS = (
    ("hyperrectangle_structure", "Axis-aligned", 2, 3),
    ("gaussian_mixture_low_d", "Smooth", 2, 3),
    ("gaussian_mixture_high_d_sparse", "Sparse", 1, 2),
)
REAL_DATA = (
    ("household_power", "POWER", 3, 3),
    ("dry_bean", "Dry Bean", 2, 2),
    ("isolet", "ISOLET", 2, 2),
)
SIMULATION_METHODS = (
    ("GEB-M", "GEB", "mixture"),
    ("HMPT-M", "HMPT", "mixture"),
    ("PT-M", "PT", "mixture"),
    ("GEB-G", "GEB", "normal"),
    ("HMPT-G", "HMPT", "normal"),
    ("PT-G", "PT", "normal"),
    ("DPMM", "DPMM", "raw"),
)
REAL_METHODS = tuple((method, method, None) for method in ("GEB", "HMPT", "PT", "DPMM"))


def _format(value, digits, bold=False, parentheses=False):
    if not np.isfinite(value):
        return "--"
    text = f"{value:.{digits}f}"
    if parentheses:
        text = f"({text})"
    return rf"\textbf{{{text}}}" if bold else text


def _render_table(folder, datasets, methods, label):
    shape = (len(methods), len(datasets))
    means, errors, times = (np.full(shape, np.nan) for _ in range(3))
    complete = np.zeros(shape, dtype=bool)
    pending = []
    for col, (dataset, _, _, _) in enumerate(datasets):
        path = folder / f"{dataset}_results.csv"
        table = pd.read_csv(path).set_index("repetition") if path.exists() else pd.DataFrame()
        numeric_rows = pd.to_numeric(table.index, errors="coerce").notna()
        for row, (display, method, transform) in enumerate(methods):
            if transform is None:
                transform = ("pca" if dataset == "isolet" else "raw") if method == "DPMM" else (
                    "mixture" if dataset == "household_power" else "normal"
                )
            score_column = f"{method}_{transform}_log_score"
            time_column = f"{method}_{transform}_running_time"
            if score_column not in table:
                pending.append(f"{dataset}/{display}: 0/{N_REPETITIONS}")
                continue
            count = table.loc[numeric_rows, score_column].count()
            timing_count = table.loc[numeric_rows, time_column].count()
            if method == "GEB" and "1" in table.index and pd.notna(table.at["1", time_column]):
                timing_count -= 1
            complete[row, col] = count == N_REPETITIONS and timing_count == N_REPETITIONS - (method == "GEB")
            if not complete[row, col]:
                pending.append(f"{dataset}/{display}: {count}/{N_REPETITIONS} scores")
            if "Mean" in table.index:
                means[row, col] = table.at["Mean", score_column]
                times[row, col] = table.at["Mean", time_column]
            if "Standard error" in table.index:
                errors[row, col] = table.at["Standard error", score_column]

    best_score = np.zeros(shape, dtype=bool)
    best_time = np.zeros(shape, dtype=bool)
    for col in range(len(datasets)):
        if complete[:, col].all():
            best_score[:, col] = means[:, col] == means[:, col].max()
            best_time[:, col] = times[:, col] == times[:, col].min()

    lines = ["% Generated from density-estimation CSV summaries."]
    if pending:
        lines.append("% Incomplete results; winners are bold only for fully completed columns.")
        lines.extend("% " + item for item in pending)
    lines.extend(r"""\begin{table}[!htbp]
\centering
\small
\setlength{\tabcolsep}{3pt}
\captionsetup{font=small}
\caption{Average test log scores and wall-clock run-times.
Scores are reported as mean (standard error); best results are in bold.}""".splitlines())
    lines.extend([rf"\label{{tab:{label}_logscore}}", rf"\label{{tab:{label}_time}}"])
    lines.extend(r"""\begin{tabular}{@{}l@{\quad} r@{\ }l r@{\ }l r@{\ }l@{\hspace{1.5em}} c c c@{}}
\toprule
& \multicolumn{6}{c@{\hspace{1.5em}}}{Test log score}
& \multicolumn{3}{c}{Run-time (seconds)} \\
\cmidrule(lr){2-7}\cmidrule{8-10}
Method""".splitlines())
    for col, (_, title, _, _) in enumerate(datasets):
        spec = r"c@{\hspace{1.5em}}" if col == 2 else "c"
        lines.append(rf"& \multicolumn{{2}}{{{spec}}}{{{title}}}")
    lines.extend(["& " + " & ".join(item[1] for item in datasets) + r" \\", r"\midrule"])
    for row, (display, method, _) in enumerate(methods):
        if label == "sim" and row in (3, 6):
            lines.append(r"\midrule")
        lines.append(display)
        for col, (_, _, score_digits, se_digits) in enumerate(datasets):
            score = _format(means[row, col], score_digits, best_score[row, col])
            se = _format(errors[row, col], se_digits, best_score[row, col], parentheses=True)
            lines.append(f"& {score} & {se}")
        timings = []
        for col in range(len(datasets)):
            digits = (2 if col == 2 else 0) if method == "DPMM" else (0 if method == "HMPT" else 2)
            timings.append(_format(times[row, col], digits, best_time[row, col]))
        lines.append("& " + " & ".join(timings) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_density_tables(updated_csv=None):
    """Refresh both tables, or only the table affected by one saved CSV."""
    folder = Path(updated_csv).parent if updated_csv is not None else DENSITY_RESULTS_ROOT
    dataset = Path(updated_csv).stem.removesuffix("_results") if updated_csv is not None else None
    outputs = []
    for datasets, methods, label, filename in (
        (SIMULATIONS, SIMULATION_METHODS, "sim", "simulation_table.tex"),
        (REAL_DATA, REAL_METHODS, "real", "real_data_table.tex"),
    ):
        if dataset is not None and dataset not in {item[0] for item in datasets}:
            continue
        text = _render_table(folder, datasets, methods, label)
        path = folder / filename
        folder.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tex.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
        outputs.append(str(path))
    return outputs
