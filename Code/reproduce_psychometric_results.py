#!/usr/bin/env python3
"""Reproduce the psychometric summaries from the archived 21-point predictions.

The manuscript analyses use nine stimulus levels selected from the 21-point
prediction series: 0, 20, 30, 40, 50, 60, 70, 80, and 100 percent happy.
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import f as f_distribution
from scipy.stats import t as t_distribution
from scipy.stats import ttest_rel


STIMULUS_LEVELS = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
SELECTED_ROWS = np.array([0, 4, 6, 8, 10, 12, 14, 16, 20])
LOWER_BOUNDS = np.array([0.0, 0.0, 0.0, 0.1])
UPPER_BOUNDS = np.array([0.05, 0.05, 1.0, 100.0])
START_VALUES = np.array([0.01, 0.02, 0.5, 5.0])

FC_PATTERN = re.compile(
    r"^Location-(?P<Location>[123])-(?P<BaseType>FaceBased|ObjectBased)-"
    r"(?P<MaskType>Full|E|M|N)-(?P<Reduction>2|4|8|16|32)\.csv$"
)
BASELINE_PATTERN = re.compile(
    r"^(?P<Model>AlexNet|VGG-16)-(?P<BaseType>FaceBased|ObjectBased)-"
    r"(?P<MaskType>Full|E|M|N)-raw\.csv$"
)


def psychometric_function(x, guess, lapse, pse, scale):
    return guess + (1.0 - guess - lapse) * 0.5 * (
        1.0 + erf((x - pse) / np.sqrt(2.0 * scale**2))
    )


def read_prediction(path: Path) -> np.ndarray:
    table = pd.read_csv(path, header=None)
    if table.shape[1] < 2:
        raise ValueError(f"{path.name}: expected at least two columns")
    values = pd.to_numeric(table.iloc[:, 1], errors="coerce").dropna().to_numpy(float)
    if len(values) != 21:
        raise ValueError(f"{path.name}: expected 21 predictions, found {len(values)}")
    if np.any((values < 0) | (values > 1)):
        raise ValueError(f"{path.name}: probabilities must be between 0 and 1")
    return values


def fit_curve(y: np.ndarray) -> dict[str, float]:
    parameters, _ = curve_fit(
        psychometric_function,
        STIMULUS_LEVELS,
        y,
        p0=START_VALUES,
        bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
        maxfev=100_000,
    )
    guess, lapse, pse, scale = parameters
    fitted = psychometric_function(STIMULUS_LEVELS, *parameters)
    residual = y - fitted
    rmse_fit = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - np.sum(residual**2) / denominator) if denominator else np.nan
    slope = float((1.0 - guess - lapse) / (np.sqrt(2.0 * np.pi) * scale))
    return {
        "GuessRate": float(guess),
        "LapseRate": float(lapse),
        "PSE": float(pse),
        "Scale": float(scale),
        "SlopeAtPSE": slope,
        "R2_Fit": r_squared,
        "RMSE_Fit": rmse_fit,
    }


def parse_condition(path: Path) -> dict[str, object]:
    match = FC_PATTERN.match(path.name)
    if match:
        values = match.groupdict()
        return {
            "Architecture": "SE-AlexNet",
            "Location": f"L{values['Location']}",
            "BaseType": values["BaseType"],
            "MaskType": values["MaskType"],
            "Reduction": int(values["Reduction"]),
            "ModelVariant": f"SE-AlexNet-R{values['Reduction']}",
        }
    match = BASELINE_PATTERN.match(path.name)
    if match:
        values = match.groupdict()
        architecture = "VGG16" if values["Model"] == "VGG-16" else "AlexNet"
        return {
            "Architecture": architecture,
            "Location": "Baseline",
            "BaseType": values["BaseType"],
            "MaskType": values["MaskType"],
            "Reduction": np.nan,
            "ModelVariant": architecture,
        }
    raise ValueError(f"Unrecognized prediction filename: {path.name}")


def effect_component(values: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Return subject-specific factorial effect components.

    values has shape subject x pretraining x location x reduction.
    """
    component = values
    for axis in range(3):
        if axis not in axes:
            component = component.mean(axis=axis + 1, keepdims=True)
    for axis in axes:
        component = component - component.mean(axis=axis + 1, keepdims=True)
    return component


def repeated_measures_anova(delta: np.ndarray) -> pd.DataFrame:
    factor_names = ["Pretraining", "Location", "Reduction"]
    factor_levels = delta.shape[1:]
    rows = []
    for order in range(1, 4):
        for axes in itertools.combinations(range(3), order):
            subject_effect = effect_component(delta, axes)
            group_effect = subject_effect.mean(axis=0, keepdims=True)
            complement = int(
                np.prod([factor_levels[i] for i in range(3) if i not in axes])
            )
            ss_effect = float(len(delta) * complement * np.sum(group_effect**2))
            ss_error = float(complement * np.sum((subject_effect - group_effect) ** 2))
            df_effect = int(np.prod([factor_levels[i] - 1 for i in axes]))
            df_error = int((len(delta) - 1) * df_effect)
            ms_effect = ss_effect / df_effect
            ms_error = ss_error / df_error
            f_value = ms_effect / ms_error
            p_value = float(f_distribution.sf(f_value, df_effect, df_error))
            partial_eta_squared = ss_effect / (ss_effect + ss_error)
            rows.append(
                {
                    "Effect": " × ".join(factor_names[i] for i in axes),
                    "SS_Effect": ss_effect,
                    "SS_Error": ss_error,
                    "df_Effect": df_effect,
                    "df_Error": df_error,
                    "F": f_value,
                    "p": p_value,
                    "Partial_Eta_Squared": partial_eta_squared,
                }
            )
    return pd.DataFrame(rows)


def paired_posthoc(
    delta: np.ndarray,
    compare_axis: int,
    family_axes: tuple[int, ...],
    factor_labels: list[list[str]],
) -> pd.DataFrame:
    rows = []
    compare_labels = factor_labels[compare_axis]
    family_shape = [delta.shape[axis + 1] for axis in family_axes]
    comparisons_per_family = len(list(itertools.combinations(compare_labels, 2)))
    for family_indices in np.ndindex(*family_shape):
        selector: list[object] = [slice(None), slice(None), slice(None), slice(None)]
        family_values = {}
        for axis, index in zip(family_axes, family_indices):
            selector[axis + 1] = index
            family_values[["Pretraining", "Location", "Reduction"][axis]] = factor_labels[axis][index]
        for first, second in itertools.combinations(range(len(compare_labels)), 2):
            first_selector = selector.copy()
            second_selector = selector.copy()
            first_selector[compare_axis + 1] = first
            second_selector[compare_axis + 1] = second
            first_values = delta[tuple(first_selector)]
            second_values = delta[tuple(second_selector)]
            test = ttest_rel(first_values, second_values)
            difference = first_values - second_values
            sd_difference = float(np.std(difference, ddof=1))
            row = {
                **family_values,
                "ComparisonFactor": ["Pretraining", "Location", "Reduction"][compare_axis],
                "Level1": compare_labels[first],
                "Level2": compare_labels[second],
                "MeanDifference_Level1MinusLevel2": float(np.mean(difference)),
                "t": float(test.statistic),
                "df": len(difference) - 1,
                "p_Raw": float(test.pvalue),
                "p_Bonferroni": min(1.0, float(test.pvalue) * comparisons_per_family),
                "Cohens_dz": float(np.mean(difference) / sd_difference) if sd_difference else np.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=repository_root / "Data" / "model_predictions_21point",
    )
    parser.add_argument(
        "--human-curve",
        type=Path,
        default=repository_root / "Data" / "human-data-9point.csv",
    )
    parser.add_argument(
        "--human-pse",
        type=Path,
        default=repository_root / "Code" / "Human_PSE_40_Subjects.csv",
    )
    parser.add_argument("--output", type=Path, default=repository_root / "Results")
    args = parser.parse_args()

    prediction_files = sorted(args.predictions.glob("*.csv"))
    expected_names = {
        f"Location-{location}-{pretraining}-{mask}-{reduction}.csv"
        for location in [1, 2, 3]
        for pretraining in ["FaceBased", "ObjectBased"]
        for mask in ["Full", "E", "M", "N"]
        for reduction in [2, 4, 8, 16, 32]
    }
    expected_names.update(
        f"{model}-{pretraining}-{mask}-raw.csv"
        for model in ["AlexNet", "VGG-16"]
        for pretraining in ["FaceBased", "ObjectBased"]
        for mask in ["Full", "E", "M", "N"]
    )
    actual_names = {path.name for path in prediction_files}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"Prediction archive mismatch; missing={missing}, extra={extra}")

    human_curve = pd.read_csv(args.human_curve, header=None).iloc[:, 0].to_numpy(float)
    if len(human_curve) != 9:
        raise ValueError("The human mean curve must contain exactly nine values")
    human_pse_table = pd.read_csv(args.human_pse)
    human_pse = human_pse_table["PSE_u"].to_numpy(float)
    if len(human_pse) != 40:
        raise ValueError("The individual human PSE file must contain exactly 40 rows")

    human_fit = fit_curve(human_curve)
    human_mean = float(np.mean(human_pse))
    human_sd = float(np.std(human_pse, ddof=1))
    human_sem = human_sd / np.sqrt(len(human_pse))

    result_rows = []
    for path in prediction_files:
        condition = parse_condition(path)
        prediction_21 = read_prediction(path)
        prediction_9 = prediction_21[SELECTED_ROWS]
        fit = fit_curve(prediction_9)
        t_value = (human_mean - fit["PSE"]) / human_sem
        p_value = float(2.0 * t_distribution.sf(abs(t_value), len(human_pse) - 1))
        pse_difference = fit["PSE"] - human_mean
        result_rows.append(
            {
                "SourceFile": path.name,
                **condition,
                **fit,
                "PSE_Difference_ModelMinusHuman": float(pse_difference),
                "t_Absolute": float(abs(t_value)),
                "df": len(human_pse) - 1,
                "p_TwoSided": p_value,
                "Cohens_d_Absolute": float(abs(pse_difference) / human_sd),
                "RMSE_vs_HumanMeanCurve": float(
                    np.sqrt(np.mean((prediction_9 - human_curve) ** 2))
                ),
            }
        )

    results = pd.DataFrame(result_rows).sort_values(
        ["Location", "BaseType", "MaskType", "Reduction", "Architecture"],
        na_position="last",
    )

    fc_full = results[
        (results["Architecture"] == "SE-AlexNet") & (results["MaskType"] == "Full")
    ].copy()
    pse_lookup = {
        (row.BaseType, row.Location, int(row.Reduction)): row.PSE
        for row in fc_full.itertuples()
    }
    pretraining_labels = ["FaceBased", "ObjectBased"]
    location_labels = ["L1", "L2", "L3"]
    reduction_labels = ["R2", "R4", "R8", "R16", "R32"]
    delta = np.empty((len(human_pse), 2, 3, 5), dtype=float)
    long_rows = []
    for subject_index, subject_pse in enumerate(human_pse):
        for pretraining_index, pretraining in enumerate(pretraining_labels):
            for location_index, location in enumerate(location_labels):
                for reduction_index, reduction_label in enumerate(reduction_labels):
                    reduction = int(reduction_label[1:])
                    value = abs(pse_lookup[(pretraining, location, reduction)] - subject_pse)
                    delta[subject_index, pretraining_index, location_index, reduction_index] = value
                    long_rows.append(
                        {
                            "SubjectID": human_pse_table.iloc[subject_index]["SubjectID"],
                            "Pretraining": pretraining,
                            "Location": location,
                            "Reduction": reduction_label,
                            "DeltaPSE": value,
                        }
                    )

    wide_columns = {}
    for pretraining_index, pretraining in enumerate(pretraining_labels):
        prefix = "Face" if pretraining == "FaceBased" else "Object"
        for location_index, location in enumerate(location_labels):
            for reduction_index, reduction in enumerate(reduction_labels):
                wide_columns[f"{prefix}_{location}_{reduction}"] = delta[
                    :, pretraining_index, location_index, reduction_index
                ]
    wide = pd.DataFrame(wide_columns)
    wide.insert(0, "SubjectID", human_pse_table["SubjectID"].to_numpy())

    factor_labels = [pretraining_labels, location_labels, reduction_labels]
    posthoc_pretraining = paired_posthoc(delta, 0, (1, 2), factor_labels)
    posthoc_locations = paired_posthoc(delta, 1, (0, 2), factor_labels)
    posthoc_reductions = paired_posthoc(delta, 2, (0, 1), factor_labels)

    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / "Model_Psychometric_Results.csv", index=False)

    table1 = results[
        (results["MaskType"] == "Full")
        & (
            (results["Architecture"].isin(["AlexNet", "VGG16"]))
            | (
                (results["Architecture"] == "SE-AlexNet")
                & (results["Location"] == "L3")
            )
        )
    ].copy()
    table1["PaperModel"] = table1["Architecture"]
    se_rows = table1["Architecture"] == "SE-AlexNet"
    table1.loc[se_rows, "PaperModel"] = (
        "SE-AlexNet-R"
        + table1.loc[se_rows, "Reduction"].astype(int).astype(str)
    )
    model_order = {
        "VGG16": 0,
        "AlexNet": 1,
        "SE-AlexNet-R2": 2,
        "SE-AlexNet-R4": 3,
        "SE-AlexNet-R8": 4,
        "SE-AlexNet-R16": 5,
        "SE-AlexNet-R32": 6,
    }
    table1["ModelOrder"] = table1["PaperModel"].map(model_order)
    table1 = table1.sort_values(["ModelOrder", "BaseType"])
    table1[
        [
            "PaperModel",
            "BaseType",
            "SourceFile",
            "PSE",
            "PSE_Difference_ModelMinusHuman",
            "t_Absolute",
            "df",
            "p_TwoSided",
            "Cohens_d_Absolute",
            "RMSE_vs_HumanMeanCurve",
        ]
    ].to_csv(args.output / "Paper_Table1_PSE_RMSE.csv", index=False)
    table1[
        [
            "PaperModel",
            "BaseType",
            "SourceFile",
            "R2_Fit",
            "RMSE_Fit",
        ]
    ].to_csv(args.output / "Paper_Table2_Fit_Quality.csv", index=False)

    table3 = results[
        (results["Architecture"].isin(["AlexNet", "VGG16"]))
        | (
            (results["Architecture"] == "SE-AlexNet")
            & (results["Reduction"] == 16)
        )
    ].copy()
    table3["PaperModel"] = np.where(
        table3["Architecture"] == "SE-AlexNet",
        table3["Location"].map(
            {
                "L1": "SE-AlexNet-L1-R16",
                "L2": "SE-AlexNet-L2-R16",
                "L3": "SE-AlexNet-L3-R16",
            }
        ),
        table3["Architecture"],
    )
    mask_order = {"Full": 0, "E": 1, "N": 2, "M": 3}
    table3_model_order = {
        "VGG16": 0,
        "AlexNet": 1,
        "SE-AlexNet-L1-R16": 2,
        "SE-AlexNet-L2-R16": 3,
        "SE-AlexNet-L3-R16": 4,
    }
    table3["MaskOrder"] = table3["MaskType"].map(mask_order)
    table3["ModelOrder"] = table3["PaperModel"].map(table3_model_order)
    table3 = table3.sort_values(["MaskOrder", "ModelOrder", "BaseType"])
    table3[
        [
            "MaskType",
            "PaperModel",
            "BaseType",
            "SourceFile",
            "PSE",
            "PSE_Difference_ModelMinusHuman",
            "t_Absolute",
            "df",
            "p_TwoSided",
            "Cohens_d_Absolute",
            "RMSE_vs_HumanMeanCurve",
        ]
    ].to_csv(args.output / "Paper_Table3_Occlusion.csv", index=False)

    pd.DataFrame(
        [
            {
                "N_Individuals": len(human_pse),
                "Individual_PSE_Mean": human_mean,
                "Individual_PSE_SD": human_sd,
                "Individual_PSE_SEM": human_sem,
                "Group_Curve_PSE": human_fit["PSE"],
                "Group_Curve_R2_Fit": human_fit["R2_Fit"],
                "Group_Curve_RMSE_Fit": human_fit["RMSE_Fit"],
            }
        ]
    ).to_csv(args.output / "Human_Psychometric_Summary.csv", index=False)
    pd.DataFrame(long_rows).to_csv(args.output / "DeltaPSE_Long.csv", index=False)
    wide.to_csv(args.output / "DeltaPSE_Wide.csv", index=False)
    repeated_measures_anova(delta).to_csv(
        args.output / "Repeated_Measures_ANOVA.csv", index=False
    )
    posthoc_locations.to_csv(args.output / "PostHoc_Locations_within_Condition.csv", index=False)
    posthoc_reductions.to_csv(args.output / "PostHoc_Reductions_within_Condition.csv", index=False)
    posthoc_pretraining.to_csv(args.output / "PostHoc_Pretraining_within_Condition.csv", index=False)

    print(f"Validated {len(prediction_files)} prediction files.")
    print(f"Wrote reproducible result tables to {args.output}")


if __name__ == "__main__":
    main()
