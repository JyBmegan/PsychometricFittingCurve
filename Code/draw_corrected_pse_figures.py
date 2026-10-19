#!/usr/bin/env python3
"""Rebuild corrected manuscript Figure 1d and Figure 3 from result tables.

Figure contract
---------------
Conclusion: signed PSE deviations depend on pretrained weights and SE insertion
position; several face-pretrained models are close to the human mean.
Archetype: quantitative grouped bar chart.
Evidence: PSE_Difference_ModelMinusHuman from the complete-mask model fits,
reported in percentage points. Every trained model contributes one fixed PSE,
so no inferential error bars are added.
Panels: Figure 1d = the L3 experimental label; Figure 3a = L1 (early);
Figure 3b = L2 (middle).
Baseline: the horizontal zero line is the human-participant mean PSE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = [
    "VGG-16",
    "AlexNet",
    "SE-AlexNet-R2",
    "SE-AlexNet-R4",
    "SE-AlexNet-R8",
    "SE-AlexNet-R16",
    "SE-AlexNet-R32",
]
MODEL_COLORS = [
    "#32B894",
    "#8CC9C5",
    "#67B2C9",
    "#4E98BD",
    "#327DAE",
    "#315F91",
    "#2D4B61",
]
BASE_TYPES = ["ObjectBased", "FaceBased"]
GROUP_TITLES = {
    "ObjectBased": "物体预训练权重模型",
    "FaceBased": "面孔预训练权重模型",
}
Y_LIMITS = (-25, 15)
Y_TICKS = np.arange(-25, 16, 5)


def configure_matplotlib() -> None:
    """Set manuscript-safe typography and editable PostScript text."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial Unicode MS",
                "Songti SC",
                "Heiti TC",
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 0.8,
            "patch.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "Architecture",
        "Location",
        "BaseType",
        "MaskType",
        "Reduction",
        "PSE_Difference_ModelMinusHuman",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return df.loc[df["MaskType"].eq("Full")].copy()


def panel_values(df: pd.DataFrame, location: str, base_type: str) -> np.ndarray:
    """Return baseline and SE-model signed differences in the paper's order."""
    values: list[float] = []
    for architecture in ("VGG16", "AlexNet"):
        rows = df.loc[
            df["Architecture"].eq(architecture)
            & df["Location"].eq("Baseline")
            & df["BaseType"].eq(base_type)
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one {architecture}/{base_type} baseline row; got {len(rows)}"
            )
        values.append(float(rows.iloc[0]["PSE_Difference_ModelMinusHuman"]) * 100)

    for reduction in (2, 4, 8, 16, 32):
        rows = df.loc[
            df["Architecture"].eq("SE-AlexNet")
            & df["Location"].eq(location)
            & df["BaseType"].eq(base_type)
            & df["Reduction"].eq(reduction)
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one SE-AlexNet/{location}/{base_type}/R{reduction} row; "
                f"got {len(rows)}"
            )
        values.append(float(rows.iloc[0]["PSE_Difference_ModelMinusHuman"]) * 100)
    return np.asarray(values)


def draw_panel(ax: plt.Axes, df: pd.DataFrame, location: str) -> None:
    n_models = len(MODEL_LABELS)
    group_gap = 1.0
    x_object = np.arange(n_models, dtype=float)
    x_face = x_object + n_models + group_gap

    for base_type, positions in zip(BASE_TYPES, (x_object, x_face)):
        values = panel_values(df, location, base_type)
        bars = ax.bar(
            positions,
            values,
            width=0.78,
            color=MODEL_COLORS,
            edgecolor="#151515",
            linewidth=0.75,
            zorder=3,
        )
        if base_type == "ObjectBased":
            for bar in bars:
                bar.set_linestyle((0, (4, 3)))

    all_positions = np.concatenate([x_object, x_face])
    ax.set_xticks(all_positions)
    ax.set_xticklabels(MODEL_LABELS * 2, rotation=48, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.85, x_face[-1] + 0.85)
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks(Y_TICKS)
    ax.axhline(0, color="#111111", linewidth=0.85, zorder=4)

    ax.text(
        float(x_object.mean()),
        13.0,
        GROUP_TITLES["ObjectBased"],
        ha="center",
        va="center",
        fontsize=7.5,
    )
    ax.text(
        float(x_face.mean()),
        13.0,
        GROUP_TITLES["FaceBased"],
        ha="center",
        va="center",
        fontsize=7.5,
    )
    ax.text(
        x_face[-1],
        0.7,
        "人类被试主观相等点均值",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        zorder=5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", direction="out", pad=2)
    ax.tick_params(axis="y", direction="out", pad=2)


def export_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.eps", format="eps")
    fig.savefig(output_dir / f"{stem}.svg", format="svg")
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf")
    fig.savefig(output_dir / f"{stem}_preview.png", format="png", dpi=600)
    fig.savefig(output_dir / f"{stem}.tiff", format="tiff", dpi=600)
    plt.close(fig)


def make_figure1d(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2047, 3.3858))
    draw_panel(ax, df, "L3")
    ax.set_ylabel("Location-3模型 PSE 与人类被试均值的差异（百分点）", labelpad=4)
    ax.text(
        -0.095,
        0.985,
        "(d)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.115, right=0.992, bottom=0.30, top=0.97)
    export_figure(fig, output_dir, "Figure1d_corrected")


def make_figure3(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2047, 6.6929),
        sharey=True,
    )
    panel_specs = [
        (axes[0], "L1", "(a)", "前序插入模型 PSE 与人类被试均值的差异（百分点）"),
        (axes[1], "L2", "(b)", "中序插入模型 PSE 与人类被试均值的差异（百分点）"),
    ]
    for ax, location, label, ylabel in panel_specs:
        draw_panel(ax, df, location)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.text(
            -0.095,
            0.985,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.115, right=0.992, bottom=0.145, top=0.985, hspace=0.52)
    export_figure(fig, output_dir, "Figure3_corrected")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=repo_root / "Results" / "Model_Psychometric_Results.csv",
        help="Model psychometric result table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "Results" / "Corrected_Figures",
        help="Directory for EPS figures and PNG previews.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    results = load_results(args.results_csv)
    make_figure1d(results, args.output_dir)
    make_figure3(results, args.output_dir)
    print(f"Wrote corrected figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
