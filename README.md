# Psychometric-Function Analysis for SE-AlexNet

Reproduce the paper's PSE, fit-quality, ΔPSE, and ANOVA tables from the 21-point model predictions and the human behavioral data.

## Reproduce

```bash
pip install -r requirements.txt
python3 Code/reproduce_psychometric_results.py
```

The script validates 136 prediction files and writes every result table into `Results/`.

## Inputs

| Path | Content |
| --- | --- |
| `Data/model_predictions_21point/` | 136 prediction files: 120 SE-AlexNet conditions + 16 AlexNet/VGG-16 baselines; each file holds 21 response probabilities |
| `Data/human-data-9point.csv` | Human mean response probability at 9 stimulus intensities |
| `Code/Human_PSE_40_Subjects.csv` | Individual PSE and curve parameters for the 40 participants |

The analysis takes the 1st, 5th, 7th, 9th, 11th, 13th, 15th, 17th, and 21st points of each 21-point series, corresponding to stimulus intensities `0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0`.

Mask labels are `Full` (intact), `E` (eyes occluded), `N` (nose occluded), `M` (mouth occluded). `Location-1/2/3` are the three fully-connected SE insertion positions (pre/mid/post). Reduction ratios are 2, 4, 8, 16, 32.

## Outputs

`Results/` holds 11 tables, three of which the paper uses directly:

| Output | Content |
| --- | --- |
| `Paper_Table1_PSE_RMSE.csv` | Paper Table 1: per-model PSE, t, p, Cohen's d, and model-human curve RMSE |
| `Paper_Table2_Fit_Quality.csv` | Paper Table 2: fit R² and RMSE |
| `Paper_Table3_Occlusion.csv` | Paper Table 3: per-model PSE statistics under occlusion |

The rest are intermediate data: `Model_Psychometric_Results.csv` (complete statistics for all 136 models), `Human_Psychometric_Summary.csv`, `DeltaPSE_Long/Wide.csv`, `Repeated_Measures_ANOVA.csv`, and the three `PostHoc_*.csv` files.

## Psychometric function and statistics

Cumulative-Gaussian psychometric function (fitted to the 9 points):

$$
y = g + (1-g-l)\cdot\frac12\left[1+\operatorname{erf}\left(\frac{x-u}{\sqrt{2v^2}}\right)\right]
$$

- $g$: lower asymptote (guess); $l$: upper-asymptote loss (lapse); $u$: point of subjective equality (PSE); $v$: curve scale.
- Slope at the PSE: $\displaystyle\frac{1-g-l}{\sqrt{2\pi}\,v}$.

Fit bounds and starting values:

```text
Upper bounds [0.05, 0.05, 1.0, 100.0]
Start values [0.01, 0.02, 0.5,   5.0]
Lower bounds [0.00, 0.00, 0.0,   0.1]
```

Two-sided one-sample t-test of each model PSE against the 40 human PSEs:

$$
t=\frac{\bar{PSE}_{human}-PSE_{model}}{SD_{human}/\sqrt{N}},
\qquad
|d|=\frac{|PSE_{model}-\bar{PSE}_{human}|}{SD_{human}}
$$

Individual-level ΔPSE:

$$
\Delta PSE_{ij}=|PSE_{model,i}-PSE_{human,j}|
$$

ΔPSE is analyzed with a 2 (pretraining type) × 3 (insertion location) × 5 (reduction ratio) repeated-measures ANOVA, plus Bonferroni-corrected post-hoc paired t-tests.
