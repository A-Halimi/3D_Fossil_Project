#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fossil Model Comparison Report Generator (v2)
=============================================

This module aggregates the saved *classification_report*.txt files produced by
your Fossil Classifier training runs (v5 rig or similar), parses the overall
and per-class metrics, and generates a comprehensive comparative report across
all models you’ve trained.

Major Features
--------------
- Automatically discovers classification report files under a results root dir.
- Skips `.ipynb_checkpoints` copies so each model is parsed once.
- Robust parser for scikit-learn `classification_report` text *with indentation*
  (handles leading spaces + multi-word class names).
- Reads optional header block metrics ("Test Accuracy", "Top-3 Accuracy", "Test AUC")
  when your training rig saved them; otherwise falls back to the accuracy row in
  the sklearn table.
- Builds an **overall per-model metrics table** (accuracy, top-3, AUC, macro/weighted).
- Builds a **wide per-class table** (precision, recall, F1, support) across all models.
- Computes **strengths & weaknesses** per model relative to the per-class best
  (configurable recall & precision gap thresholds).
- Produces **leaderboards** by accuracy, macro-F1, weighted-F1, macro precision, macro recall.
- Derives **per-class best & worst** model tables (F1 fallback to recall).
- Generates a rich **Markdown report** + CSV exports + JSON summary for programmatic use.

Typical Directory Layout Expected
---------------------------------
results/
  effv2l/
    reports/classification_report.txt
  effv2s/
    reports/classification_report.txt
  mobilenet/
    reports/classification_report.txt
  convnext/
    reports/classification_report.txt
  ...
(Additional subfolders OK; the script recurses.)

Quick Start (Notebook / Python REPL)
------------------------------------
>>> import fossil_model_compare_v2 as fmc2
>>> res = fmc2.build_full_report(results_root="results", recall_gap=0.15, precision_gap=0.15)
>>> res["overall_table"]

Command Line
------------
$ python fossil_model_compare_v2.py --results_root results --recall_gap 0.15 --precision_gap 0.15

Outputs (default: results/_comparison/)
--------------------------------------
- fossil_model_comparison_report_v2.md            (full markdown report)
- fossil_model_comparison_report_v2.csv           (overall per-model metrics)
- fossil_model_per_class_metrics_v2.csv           (wide per-class metrics)
- fossil_model_per_class_best_v2.csv              (best & worst model per class)
- fossil_model_strengths_v2.json                  (structured strengths/weaknesses dict)

Author: ChatGPT (OpenAI), generated for Fossil Classifier project
"""

from __future__ import annotations

import os
import re
import json
import pathlib
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _safe_float(x, default=np.nan):
    """Convert to float; return default on failure."""
    try:
        return float(x)
    except Exception:
        return default


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------
def parse_sklearn_report_block(text: str) -> tuple[pd.DataFrame, dict]:
    """
    Parse the *table portion* of a scikit-learn `classification_report` string.

    We assume the block begins with a header line containing the tokens
    "precision", "recall", and "support" in any spacing. All subsequent
    non-empty lines are parsed as:

        <label>  <precision>  <recall>  <f1-score>  <support>

    where <label> may contain spaces. The last 4 tokens must be numeric; all
    preceding tokens are joined to form the label.

    Special summary rows recognized (case-insensitive):
        accuracy
        macro avg
        weighted avg

    Returns
    -------
    per_class_df : pd.DataFrame
        Index: class label
        Columns: precision, recall, f1, support
    summary : dict
        {
          "macro_avg":  {precision, recall, f1, support} or {},
          "weighted_avg": {...} or {},
          "accuracy_row": {"accuracy": float, "support": int} or {}
        }
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() != ""]
    # find header
    start_idx: Optional[int] = None
    header_pat = re.compile(r"precision\s+recall", re.IGNORECASE)
    for i, ln in enumerate(lines):
        if header_pat.search(ln) and "support" in ln.lower():
            start_idx = i + 1
            break
    if start_idx is None:
        return pd.DataFrame(), {}

    rows: List[tuple] = []
    macro: dict = {}
    weighted: dict = {}
    acc_row: dict = {}

    for ln in lines[start_idx:]:
        low = ln.lower().strip()
        toks = ln.split()

        # accuracy
        if low.startswith("accuracy"):
            # expected: accuracy 0.9235 55679
            if len(toks) >= 3:
                acc_row = {
                    "accuracy": _safe_float(toks[-2]),
                    "support": int(float(toks[-1])),
                }
            continue

        # macro avg
        if low.startswith("macro avg"):
            if len(toks) >= 5:
                macro = dict(
                    precision=_safe_float(toks[-4]),
                    recall=_safe_float(toks[-3]),
                    f1=_safe_float(toks[-2]),
                    support=int(float(toks[-1])),
                )
            continue

        # weighted avg
        if low.startswith("weighted avg"):
            if len(toks) >= 5:
                weighted = dict(
                    precision=_safe_float(toks[-4]),
                    recall=_safe_float(toks[-3]),
                    f1=_safe_float(toks[-2]),
                    support=int(float(toks[-1])),
                )
            continue

        # normal class row
        if len(toks) < 5:
            continue
        try:
            sup = int(float(toks[-1]))
            f1 = float(toks[-2])
            rec = float(toks[-3])
            pre = float(toks[-4])
        except Exception:
            continue

        label = " ".join(toks[:-4]).strip()
        rows.append((label, pre, rec, f1, sup))

    if rows:
        per_class_df = pd.DataFrame(
            rows, columns=["label", "precision", "recall", "f1", "support"]
        ).set_index("label")
    else:
        per_class_df = pd.DataFrame(
            columns=["precision", "recall", "f1", "support"], dtype=float
        )

    summary = {
        "macro_avg": macro,
        "weighted_avg": weighted,
        "accuracy_row": acc_row,
    }
    return per_class_df, summary


def parse_full_classification_report_text(txt: str) -> tuple[dict, pd.DataFrame, dict]:
    """
    Parse a *complete* classification report file produced by the Fossil v5 rig.

    The file typically contains a header block:

        Overall Test Metrics
        ====================
        Test Accuracy : 0.9235
        Top-3 Accuracy: 0.9742
        Test AUC      : 0.9901

    followed by the scikit-learn table. We extract header metrics when present,
    then delegate table parsing to `parse_sklearn_report_block`.

    Returns
    -------
    header : dict
        { "test_accuracy": ..., "top-3_accuracy": ..., "test_auc": ... } (if found)
    per_class_df : pd.DataFrame
    summary : dict (macro_avg, weighted_avg, accuracy_row)
    """
    header = {}
    # header metrics: regex search tolerant of spacing
    m = re.search(r"Test\s+Accuracy\s*:\s*([0-9.]+)", txt, re.IGNORECASE)
    if m:
        header["test_accuracy"] = float(m.group(1))
    m = re.search(r"Top-3\s+Accuracy\s*:\s*([0-9.]+)", txt, re.IGNORECASE)
    if m:
        header["top-3_accuracy"] = float(m.group(1))
    m = re.search(r"Test\s+AUC\s*:\s*([0-9.]+)", txt, re.IGNORECASE)
    if m:
        header["test_auc"] = float(m.group(1))

    # locate the metrics table (first occurrence of the word "precision")
    idx = txt.lower().find("precision")
    if idx == -1:
        return header, pd.DataFrame(), {}
    table_block = txt[idx:]
    per_class_df, summary = parse_sklearn_report_block(table_block)
    return header, per_class_df, summary


# ------------------------------------------------------------------
# Model discovery
# ------------------------------------------------------------------
def discover_reports(
    results_root: str | pathlib.Path,
    patterns=("classification_report*.txt",),
) -> list[pathlib.Path]:
    """
    Recursively find report files. Skips `.ipynb_checkpoints` directories.
    """
    root = pathlib.Path(results_root)
    files: list[pathlib.Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    files = [p for p in files if ".ipynb_checkpoints" not in str(p)]
    return sorted(set(files))


def guess_model_name_from_path(path: pathlib.Path) -> str:
    """
    Infer the model key from the file path.

    Priority:
      1. Filename stem minus 'classification_report'
      2. Parent-of-parent directory name if file is under .../<model>/reports/<file>.txt
      3. Parent directory name fallback
    """
    stem = path.stem.lower()
    stem = stem.replace("classification_report", "")
    stem = stem.strip("_-")

    if not stem:
        # try results/<model>/reports/file
        if path.parent.name == "reports":
            stem = path.parent.parent.name.lower()
        else:
            stem = path.parent.name.lower()

    return stem


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------
def aggregate_reports(
    results_root="results",
    recall_gap=0.15,
    precision_gap=0.15,
    verbose=True,
):
    """
    Discover & parse all reports under `results_root`, returning:

        models: dict[model_key] -> {
            'path', 'header', 'per_class', 'summary'
        }
        overall_df: pd.DataFrame (index=model)
        per_class_wide: pd.DataFrame (MultiIndex cols: model, metric)
        strengths: dict[model_key] -> strengths/weaknesses/notes
    """
    files = discover_reports(results_root)
    if verbose:
        print(f"Discovered {len(files)} report files:")
        for f in files:
            print(" -", f)

    models: Dict[str, dict] = {}
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="ignore")
        header, df, summary = parse_full_classification_report_text(txt)
        if df.empty:
            if verbose:
                print(f"[WARN] Could not parse per-class metrics: {f}")
            continue
        model_key = guess_model_name_from_path(f)
        models[model_key] = dict(
            path=str(f),
            header=header,
            per_class=df,
            summary=summary,
        )

    if not models:
        raise RuntimeError("No parsable reports found.")

    # overall table
    records = []
    for m, d in models.items():
        sm = d["summary"]
        macro = sm.get("macro_avg", {})
        weighted = sm.get("weighted_avg", {})
        acc_row = sm.get("accuracy_row", {})

        row = {"model": m}
        # accuracy: prefer header (test split) else accuracy row
        row["accuracy"] = d["header"].get("test_accuracy", acc_row.get("accuracy", np.nan))
        row["top3"] = d["header"].get("top-3_accuracy", np.nan)
        row["auc"] = d["header"].get("test_auc", np.nan)

        row["macro_precision"] = macro.get("precision", np.nan)
        row["macro_recall"] = macro.get("recall", np.nan)
        row["macro_f1"] = macro.get("f1", np.nan)

        row["weighted_precision"] = weighted.get("precision", np.nan)
        row["weighted_recall"] = weighted.get("recall", np.nan)
        row["weighted_f1"] = weighted.get("f1", np.nan)

        records.append(row)

    overall_df = pd.DataFrame.from_records(records).set_index("model")

    # per-class wide
    all_classes = sorted({cls for d in models.values() for cls in d["per_class"].index})
    metrics = ["precision", "recall", "f1", "support"]
    col_tuples = [(m, met) for m in models.keys() for met in metrics]
    per_class_wide = pd.DataFrame(
        index=all_classes,
        columns=pd.MultiIndex.from_tuples(col_tuples, names=["model", "metric"]),
        dtype=float,
    )

    for m, d in models.items():
        df = d["per_class"]
        for cls in df.index:
            for met in metrics:
                per_class_wide.loc[cls, (m, met)] = df.loc[cls, met]

    # strengths
    strengths = compute_strengths(
        per_class_wide, overall_df, recall_gap=recall_gap, precision_gap=precision_gap
    )

    return models, overall_df, per_class_wide, strengths


# ------------------------------------------------------------------
# Strengths / Weaknesses vs. Best
# ------------------------------------------------------------------
def compute_strengths(
    per_class_wide: pd.DataFrame,
    overall_df: pd.DataFrame,
    recall_gap=0.15,
    precision_gap=0.15,
):
    """
    For each model, compare its per-class precision & recall to the *best* value
    among all models for that class. If within +/- gap => "strength"; if the
    gap exceeds threshold => "weakness".
    """
    models = per_class_wide.columns.levels[0]

    recalls = per_class_wide.xs("recall", axis=1, level="metric")
    precisions = per_class_wide.xs("precision", axis=1, level="metric")
    best_recall = recalls.max(axis=1)
    best_precision = precisions.max(axis=1)

    report: Dict[str, dict] = {}
    for m in models:
        s_list: List[str] = []
        w_list: List[str] = []
        for cls in per_class_wide.index:
            r = recalls.loc[cls, m]
            p = precisions.loc[cls, m]
            br = best_recall.loc[cls]
            bp = best_precision.loc[cls]

            if pd.notna(r):
                if br - r <= recall_gap:
                    s_list.append(f"{cls} recall={r:.4f} (best {br:.4f})")
                else:
                    w_list.append(f"{cls} recall={r:.4f} < best {br:.4f} (-{br - r:.4f})")

            if pd.notna(p):
                if bp - p <= precision_gap:
                    s_list.append(f"{cls} precision={p:.4f} (best {bp:.4f})")
                else:
                    w_list.append(f"{cls} precision={p:.4f} < best {bp:.4f} (-{bp - p:.4f})")

        notes = []
        if "macro_f1" in overall_df.columns and not pd.isna(overall_df.loc[m, "macro_f1"]):
            notes.append(f"Macro-F1={overall_df.loc[m, 'macro_f1']:.4f}")
        if "accuracy" in overall_df.columns and not pd.isna(overall_df.loc[m, "accuracy"]):
            notes.append(f"Accuracy={overall_df.loc[m, 'accuracy']:.4f}")
        if "auc" in overall_df.columns and not pd.isna(overall_df.loc[m, "auc"]):
            notes.append(f"AUC={overall_df.loc[m, 'auc']:.4f}")

        report[m] = dict(strengths=s_list, weaknesses=w_list, notes=notes)

    return report


# ------------------------------------------------------------------
# Leaderboards
# ------------------------------------------------------------------
def build_leaderboards(overall_df: pd.DataFrame):
    """
    Build leaderboards for key metrics.
    Returns dict[str, pd.Series].
    """
    out = {}
    for col in ["accuracy", "macro_f1", "weighted_f1", "macro_recall", "macro_precision"]:
        if col in overall_df.columns:
            out[col] = overall_df.sort_values(col, ascending=False)[col]
    return out


# ------------------------------------------------------------------
# Markdown formatting helpers
# ------------------------------------------------------------------
def df_to_md(df: pd.DataFrame, title=None, floatfmt=".4f") -> str:
    lines = []
    if title:
        lines.append(f"## {title}")
        lines.append("")
    lines.append(df.to_markdown(floatfmt=floatfmt))
    lines.append("")
    return "\n".join(lines)


def strengths_to_md(strengths: Dict[str, Dict]) -> str:
    lines = ["## Model Strengths & Weaknesses", ""]
    for m, d in strengths.items():
        lines.append(f"### {m}")
        if d["notes"]:
            lines.append("*" + "; ".join(d["notes"]) + "*")
        if d["strengths"]:
            lines.append("**Strengths:**")
            lines.extend([f"- {s}" for s in d["strengths"]])
        if d["weaknesses"]:
            lines.append("**Weaknesses:**")
            lines.extend([f"- {w}" for w in d["weaknesses"]])
        lines.append("")
    return "\n".join(lines)


def per_class_best_worst(per_class_wide: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (best_df, worst_df) tables using F1 where available (fallback: recall).
    """
    # F1 if available
    f1s = per_class_wide.xs("f1", axis=1, level="metric")
    recalls = per_class_wide.xs("recall", axis=1, level="metric")
    # fallback fill
    use_df = f1s.fillna(recalls)

    best_model = use_df.idxmax(axis=1)
    best_score = use_df.max(axis=1)
    worst_model = use_df.idxmin(axis=1)
    worst_score = use_df.min(axis=1)

    best_df = pd.DataFrame({"best_model": best_model, "best_score": best_score})
    worst_df = pd.DataFrame({"worst_model": worst_model, "worst_score": worst_score})
    return best_df, worst_df


def final_summary_md(overall_df: pd.DataFrame, per_class_wide: pd.DataFrame) -> str:
    """
    Produce a brief final summary, naming top models and class specialists.
    """
    metric = "macro_f1" if "macro_f1" in overall_df.columns else "accuracy"
    best_model = overall_df[metric].idxmax()
    runner_up = (
        overall_df.drop(index=best_model)[metric].idxmax()
        if overall_df.shape[0] > 1
        else None
    )

    lines = ["## Final Summary", ""]
    lines.append(
        f"- **Best single model:** {best_model} "
        f"(highest {metric}: {overall_df.loc[best_model, metric]:.4f})."
    )
    if runner_up:
        lines.append(
            f"- **Strong runner-up:** {runner_up} "
            f"({metric}: {overall_df.loc[runner_up, metric]:.4f})."
        )

    # Class specialists by recall
    recalls = per_class_wide.xs("recall", axis=1, level="metric")
    best_each = recalls.idxmax(axis=1)
    lines.append("")
    lines.append("### Class Specialists (Best Recall per Taxon)")
    for cls, mdl in best_each.items():
        val = recalls.loc[cls, mdl]
        lines.append(f"- {cls}: {mdl} (recall {val:.4f})")

    lines.append("")
    lines.append(
        "Consider a class-aware ensemble that weights each model more heavily for the "
        "classes above where it leads in recall; this often boosts macro-F1 on imbalanced tasks."
    )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main high-level build function
# ------------------------------------------------------------------
def build_full_report(
    results_root="results",
    output_dir=None,
    recall_gap=0.15,
    precision_gap=0.15,
    verbose=True,
):
    """
    High-level one-call pipeline:
        1. Discover & parse all classification reports.
        2. Aggregate overall & per-class metrics.
        3. Compute strengths & weaknesses vs. per-class best.
        4. Build leaderboards & best/worst tables.
        5. Write markdown + CSV + JSON bundle.

    Returns
    -------
    dict with paths & in-memory DataFrames.
    """
    results_root = pathlib.Path(results_root)
    if output_dir is None:
        output_dir = results_root / "_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    models, overall_df, per_class_wide, strengths = aggregate_reports(
        results_root,
        recall_gap=recall_gap,
        precision_gap=precision_gap,
        verbose=verbose,
    )

    # leaderboards
    lboards = build_leaderboards(overall_df)

    # per-class best / worst
    best_df, worst_df = per_class_best_worst(per_class_wide)

    # assemble markdown
    md_lines = [
        "# Fossil Classifier Comparative Report (v2)",
        "",
        f"_Generated: {datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        "This report aggregates classification reports generated by the fossil "
        "classifier training rig, compares overall and per-class metrics across "
        "all models, and highlights strengths, weaknesses, and ensemble opportunities.",
        "",
        df_to_md(
            overall_df.sort_values("macro_f1", ascending=False),
            "Overall Metrics",
        ),
        "## Leaderboards",
        "",
    ]
    for k, ser in lboards.items():
        title = k.replace("_", " ").title()
        md_lines.append(df_to_md(ser.to_frame(), f"{title} Leaderboard"))

    md_lines.append(df_to_md(per_class_wide, "Per-class Metrics by Model"))
    md_lines.append(df_to_md(best_df, "Per-class Best (F1→Recall Fallback)"))
    md_lines.append(df_to_md(worst_df, "Per-class Worst (F1→Recall Fallback)"))
    md_lines.append(strengths_to_md(strengths))
    md_lines.append(final_summary_md(overall_df, per_class_wide))

    md_text = "\n".join(md_lines)

    # save artifacts
    md_path = output_dir / "fossil_model_comparison_report_v2.md"
    csv_overall_path = output_dir / "fossil_model_comparison_report_v2.csv"
    csv_perclass_path = output_dir / "fossil_model_per_class_metrics_v2.csv"
    best_csv_path = output_dir / "fossil_model_per_class_best_v2.csv"
    worst_csv_path = output_dir / "fossil_model_per_class_worst_v2.csv"
    json_strengths_path = output_dir / "fossil_model_strengths_v2.json"

    md_path.write_text(md_text, encoding="utf-8")
    overall_df.to_csv(csv_overall_path)
    per_class_wide.to_csv(csv_perclass_path)
    best_df.to_csv(best_csv_path)
    worst_df.to_csv(worst_csv_path)
    with open(json_strengths_path, "w") as f:
        json.dump(strengths, f, indent=2)

    if verbose:
        print("\nSaved:")
        print(" -", md_path)
        print(" -", csv_overall_path)
        print(" -", csv_perclass_path)
        print(" -", best_csv_path)
        print(" -", worst_csv_path)
        print(" -", json_strengths_path)

    return dict(
        md_path=md_path,
        overall_csv=csv_overall_path,
        per_class_csv=csv_perclass_path,
        best_csv=best_csv_path,
        worst_csv=worst_csv_path,
        strengths_json=json_strengths_path,
        overall_table=overall_df,
        per_class_wide=per_class_wide,
        strengths=strengths,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Aggregate fossil classifier reports (v2)."
    )
    ap.add_argument(
        "--results_root", type=str, default="results",
        help="Root directory containing model result subfolders."
    )
    ap.add_argument(
        "--recall_gap", type=float, default=0.15,
        help="Recall gap threshold (>= gap => weakness)."
    )
    ap.add_argument(
        "--precision_gap", type=float, default=0.15,
        help="Precision gap threshold."
    )
    ap.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress logging."
    )
    args = ap.parse_args()

    build_full_report(
        results_root=args.results_root,
        recall_gap=args.recall_gap,
        precision_gap=args.precision_gap,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    _main()
