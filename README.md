# AI-Assisted Multi-Well Reservoir Characterization

**From Well Logs to PHIF Prediction and Explainable Reservoir Insights**

## 1. Project Overview

This project investigates whether PHIF (Final Porosity), as provided in the Volve petrophysical dataset, can be predicted from commonly available well-log measurements and how the predictions can support cross-well reservoir characterization.

The workflow combines data-quality auditing, geoscience-aware feature engineering, machine learning, leave-one-well-out validation, prediction-error analysis, and depth-based visualization.

The central challenge is not simply obtaining a low prediction error. It is determining whether a model trained on some wells transfers reliably to a different well and understanding where its predictions require caution.

### Project Objectives

- Audit input and output well-log data and align measurements by depth.
- Build a reproducible PHIF modeling dataset.
- Evaluate generalization to wells excluded from model training.
- Investigate the contribution of local vertical log variability.
- Characterize predicted PHIF patterns and model reliability across wells.
- Examine model behavior through feature-importance analysis.
- Communicate findings and limitations in a form useful for further geological review.

## 2. Dataset and Scope

The source data comes from the Volve Field Dataset distributed by Equinor:

[Equinor — Volve Field Data](https://www.equinor.com/energy/volve-field-data)

The original repository includes well folders, LAS/DLIS files, petrophysical reports, an inventory workbook, and formation-top information.

The current PHIF modeling workflow uses nine wells with aligned input and output measurements.

### Dataset Summary

| Item | Value |
|---|---:|
| ML-ready rows | 47,746 |
| Wells | 9 |
| Target | `PHIF` |
| Primary input curves | `GR`, `RT` |
| Derived feature | `LOG10_RT` |
| Local feature window | 11 samples |

The modeling dataset is stored locally at:

`data/processed/phif_ml_dataset_v1.csv`

Raw well folders and source log files are excluded from the intended Git working tree through `.gitignore`. Users must obtain the source dataset independently and reproduce the preprocessing workflow if they want to rebuild the dataset from raw files.

### Modeling Scope

The current modeling scope is deliberately limited to predicting the supplied PHIF target using Gamma Ray and Resistivity-derived information.

The project does not claim to independently measure porosity, permeability, hydrocarbon saturation, or reservoir productivity.

The PHIF target is treated as a supplied petrophysical reference curve. Consequently, model evaluation measures agreement with that curve rather than independent laboratory validation of porosity.

## 3. Methodology

The project follows a data-quality-first, business- and geoscience-aware workflow.

The main stages are:

1. Inventory and audit the source repository.
2. Inspect LAS headers, curves, depth ranges, and target availability.
3. Align input and output curves by rounded depth.
4. Audit target quality, missing values, and anomalous values.
5. Build the PHIF machine-learning dataset.
6. Define well-based development and holdout partitions.
7. Conduct exploratory data analysis and feature-distribution checks.
8. Establish baseline models.
9. Compare candidate models using well-based validation.
10. Investigate generalization failures and prediction bias.
11. Engineer local vertical variability features.
12. Select the final model configuration using development-well evaluation.
13. Generate leave-one-well-out predictions.
14. Characterize PHIF patterns and prediction errors by well and depth.
15. Examine model feature importance and document limitations.
16. Prepare reproducible outputs and project documentation.

The well-based evaluation strategy helps reduce the risk of data leakage between training and evaluation wells. It also makes cross-well performance assessment more representative of the intended application.

However, it does not eliminate all potential sources of leakage or evaluation bias. The data preparation process, feature engineering, model selection, and historical use of evaluation wells must also be considered when interpreting results.

## 4. Feature Engineering and Model

### Base Features

The initial model uses the following input features:

- `GR`: Gamma Ray.
- `RT`: Resistivity.
- `LOG10_RT`: Base-10 logarithm of positive RT values.

The logarithmic transformation is used to represent the resistivity distribution on a more manageable scale.

### Selected Local Features

The selected candidate adds two within-well rolling variability features:

- `GR_LOCAL_STD_11`
- `LOGRT_LOCAL_STD_11`

These features describe local variation in the input log curves using an 11-sample rolling window.

They are derived from input logs rather than the PHIF target. Their purpose is to provide information about local vertical changes that may not be represented adequately by individual log measurements.

The rolling window is expressed in samples, not physical thickness. Its geological interpretation therefore depends on the original depth sampling interval.

### Current Model Candidate

The selected candidate is `HistGradientBoostingRegressor` from scikit-learn.

| Parameter | Value |
|---|---:|
| `learning_rate` | 0.05 |
| `max_iter` | 300 |
| `max_leaf_nodes` | 15 |
| `min_samples_leaf` | 30 |
| `l2_regularization` | 1.0 |
| `random_state` | 42 |

The selected configuration is a model-development decision, not proof that it is universally optimal for other fields or wells.

### Final Feature Selection

The final feature configuration was selected using seven development wells.

The selected configuration was `LOCAL_VARIABILITY` with an 11-sample rolling window.

| Metric | Result |
|---|---:|
| Mean MAE | 0.047241 |
| Median MAE | 0.038480 |
| Mean RMSE | 0.062123 |
| Median RMSE | 0.054695 |
| Mean R² | 0.129152 |
| Median R² | 0.242625 |
| Positive R² | 5 of 7 wells |

The selection prioritized MAE while considering the overall error profile and cross-well behavior.

The selected configuration was retained as the final candidate for PHIF characterization.

Further model and feature experimentation was paused after this selection to preserve a consistent final evaluation workflow.

## 5. Evaluation Strategy

### Leave-One-Well-Out Validation

The main cross-well evaluation uses leave-one-well-out (LOO) validation.

For each fold, one well is held out while the model is trained on the other wells. This is more representative of transferring a model to an unseen well than randomly splitting individual depth samples.

The evaluation uses the following metrics:

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **R²:** Coefficient of Determination.
- **Mean Residual:** Predicted PHIF minus target PHIF.

Negative R² indicates that the predictions perform worse than a constant prediction based on the held-out well's target mean. This is an important warning about cross-well generalization.

### Historical Baseline Model Comparison

Before selecting the final local-variability feature configuration, three baseline algorithms were compared using the same input features and leave-one-well-out evaluation folds.

All three models used the same two input features:

- `GR`
- `LOG10_RT`

This comparison provides a consistent historical baseline across algorithms. It is separate from the final four-feature HistGradientBoosting candidate described in Section 4.

| Model | Mean MAE | Mean RMSE | Mean R² | Positive R² |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 0.049533 | 0.063745 | 0.023814 | 4 of 9 wells |
| XGBoost | 0.050251 | 0.064764 | -0.004358 | 4 of 9 wells |
| Linear Regression | 0.052988 | 0.065010 | -0.028520 | 4 of 9 wells |

HistGradientBoosting achieved the strongest aggregate results among the three baseline models and ranked best on 7 of 9 wells for each of MAE, RMSE, and R².

However, all three models produced positive R² on only 4 of 9 wells. Five wells had negative R², indicating that cross-well generalization remained challenging regardless of the baseline algorithm.

This comparison supported the selection of HistGradientBoosting as the baseline algorithm for subsequent feature development. The final local-variability candidate was evaluated separately through the feature-selection and historical holdout confirmation processes.

### Historical Holdout Confirmation

Feature selection was performed on seven development wells. F-1 and F1C were then used for a historical holdout confirmation.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Base HGB | 0.054147 | 0.069308 | 0.148268 |
| Local variability HGB, window 11 | 0.050076 | 0.065491 | 0.239487 |

The local variability candidate improved the aggregate historical holdout metrics:

- MAE improvement: approximately 7.52%.
- RMSE improvement: approximately 5.51%.

**Important qualification:** F-1 and F1C had already appeared in earlier leave-one-well-out experiments. These results are therefore historical holdout confirmation, not a pristine, previously untouched final test estimate.

The results should be interpreted as supporting evidence for the selected candidate rather than as an unbiased estimate of performance on completely unseen wells.

## 6. OOF Prediction Results

The current candidate was evaluated through nine leave-one-well-out folds.

### Per-Well Evaluation

| Well | Samples | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| 15_9-F-1 | 2,872 | 0.027883 | 0.034151 | 0.434662 |
| 15_9-F-1A | 2,281 | 0.042536 | 0.057616 | 0.086288 |
| 15_9-F-1B | 2,105 | 0.035278 | 0.050370 | 0.323818 |
| 15_9-F11 T2 | 2,017 | 0.041270 | 0.056178 | 0.158092 |
| 15_9-F11A | 1,533 | 0.051941 | 0.065937 | 0.105435 |
| 15_9-F11B | 13,933 | 0.043912 | 0.059311 | 0.426181 |
| 15_9-F12 | 2,647 | 0.070789 | 0.088380 | -0.505645 |
| 15_9-F15D | 11,936 | 0.040131 | 0.051171 | 0.527944 |
| 15_9-F1C | 8,422 | 0.059649 | 0.074609 | -0.227784 |

The OOF prediction dataset contains 47,746 rows across nine wells, with no missing predictions.

### Main Findings

- Performance varies substantially between wells.
- F15D has the strongest R² among the nine evaluated wells.
- F12 has the highest MAE and the strongest negative mean prediction bias.
- F1C also exhibits negative R² and underprediction.
- Predicted vertical variability is generally lower than the target variability, consistent with smoothing in most wells.

The negative R² values for F12 and F1C indicate that the model does not yet provide reliable cross-well generalization for these wells.

These results show why aggregate model performance alone is insufficient for reservoir interpretation.

### Prediction Interpretation

The OOF predictions represent predictions generated when each well was excluded from the corresponding model training fold.

They are therefore more suitable for examining cross-well generalization than predictions generated by a model fitted and evaluated on the same samples.

However, OOF predictions do not eliminate the limitations associated with the number of wells, target provenance, and geological variability in the dataset.

## 7. PHIF Characterization and Explainability

The project generates well-level summaries, fixed-width depth-window summaries, and well-log profile figures.

The characterization workflow examines:

- Actual versus predicted PHIF distributions.
- Mean prediction bias by well.
- Prediction error along depth.
- Local variability in the target and predictions.
- Differences in model performance across wells.

### Well-Level Characterization

The well-level characterization summarizes the relationship between actual and predicted PHIF for each evaluated well.

The analysis indicates that predicted PHIF variability is generally lower than actual PHIF variability across most wells.

This suggests that the model tends to produce smoother predictions than the supplied target curve.

F12 and F1C show notable underprediction patterns, while other wells exhibit different levels and directions of prediction bias.

These differences indicate that model reliability is not uniform across the evaluated wells.

### Depth-Window Characterization

The project also aggregates characterization metrics into fixed-width depth windows.

The current window width is 10 depth units.

This analysis is intended to identify local variations in prediction behavior and compare actual versus predicted PHIF patterns along each well.

The window width is an exploratory aggregation scale, not a validated geological boundary or reservoir thickness. The source depth units must be confirmed before interpreting these windows in metres.

### Model Explainability

Permutation importance outputs are generated to examine the contribution of input features to model performance.

The analysis includes:

- Overall feature importance.
- Well-specific feature importance.

These outputs help examine the extent to which the model relies on individual input features.

Feature importance describes model dependence on the available input features. It should not be interpreted as proof of a causal relationship or direct geological significance.

The results should be considered alongside the model's cross-well evaluation and geological context.

### Selected Visualization Outputs

**Cross-Well Comparison**

`reports/figures/phif_oof_well_comparison.png`

**Individual Well Profiles**

The nine profiles are stored in:

`reports/figures/phif_oof_profiles/`

Each profile compares Gamma Ray, actual versus OOF-predicted PHIF, and prediction residuals along depth.

### Interpretation Safeguards

PHIF values in this project originate from the supplied petrophysical target curve. The evaluation measures how well the model reproduces that target.

Predicted PHIF alone does not establish:

- Independent laboratory porosity.
- Permeability or flow capacity.
- Hydrocarbon presence or saturation.
- Commercial productivity.
- Validated reservoir boundaries or net pay.

Geological interpretation requires additional evidence, including suitable formation context, other petrophysical measurements, and independent validation.

The model outputs should therefore be treated as analytical support for further investigation rather than definitive reservoir engineering conclusions.

## 8. Repository Structure

The following structure presents the main project components. It is a high-level overview rather than a complete listing of every repository file.

```text id="q2r8pt"
spwla_volve/
├── .gitignore
├── README.md
├── project_history.md
├── requirements.txt
├── VOLVE_INVENTORY.xlsx
├── Well_picks_Volve_v1.dat
├── data/
│   └── processed/
│       ├── phif_ml_dataset_v1.csv
│       ├── phif_train.csv
│       ├── phif_validation.csv
│       ├── phif_test.csv
│       └── phif_well_split_metadata.csv
├── scripts/
│   └── Python scripts for auditing, preprocessing,
│       EDA, modeling, evaluation, and characterization
└── reports/
    ├── data_quality/
    │   ├── baseline_manifest_pre_stabilization_2026-11-06.csv
    │   └── CSV artifacts for data quality,
    │       LAS audits, feature analysis,
    │       and generalization diagnostics
    ├── figures/
    │   └── phif_oof_profiles/
    └── models/
        └── CSV artifacts for model evaluation,
            explainability, and characterization
```

The raw well directories and source LAS files are maintained locally and excluded from normal Git tracking.

The structure above presents the main project components rather than every individual experiment artifact.

## 9. Key Artifacts

### Processed Datasets

- `data/processed/phif_ml_dataset_v1.csv`
- `data/processed/phif_train.csv`
- `data/processed/phif_validation.csv`
- `data/processed/phif_test.csv`
- `data/processed/phif_well_split_metadata.csv`

### Data Quality and Integrity

- `reports/data_quality/phif_ml_dataset_audit_v1.csv`
- `reports/data_quality/las_dataset_readiness.csv`
- `reports/data_quality/las_target_readiness.csv`
- `reports/data_quality/well_generalization_diagnostic.csv`
- `reports/data_quality/crosswell_failure_regimes.csv`
- `reports/data_quality/baseline_manifest_pre_stabilization_2026-11-06.csv`

The baseline manifest records SHA256 hashes of 30 selected
experimental artifacts before repository stabilization.
It provides a reference for verifying artifact integrity
during subsequent maintenance.

### Final Model Selection and Evaluation

- `reports/models/final_phif_development_selection.csv`
- `reports/models/final_phif_development_selection_folds.csv`
- `reports/models/final_phif_holdout_confirmation.csv`
- `reports/models/final_phif_model_comparison.csv`
- `reports/models/final_phif_oof_metrics_by_well.csv`

### OOF Prediction and Characterization

- `reports/models/phif_oof_characterization_by_well.csv`
- `reports/models/phif_oof_characterization_by_depth_window.csv`

### Explainability

- `reports/models/phif_hgb_permutation_importance.csv`
- `reports/models/phif_hgb_permutation_importance_by_well.csv`

### EDA Visualizations

- `reports/figures/gr_vs_phif_train.png`
- `reports/figures/gr_vs_phif_by_well_train.png`
- `reports/figures/log10rt_vs_phif_train.png`
- `reports/figures/log10rt_vs_phif_by_well_train.png`

### Final OOF Visualizations

- `reports/figures/phif_oof_well_comparison.png`
- `reports/figures/phif_oof_profiles/`

The profile directory contains nine individual well-log visualization files.

### Generated Prediction Files

Per-sample prediction CSV files are generated locally by the modeling and evaluation scripts.

These files are excluded from normal Git tracking through `.gitignore` to avoid unnecessarily increasing repository size.

The summary metrics and characterization outputs remain separate from the large prediction-level datasets.

## 10. Environment and Reproducibility

### Development Environment

The current development environment uses:

| Component | Version |
|---|---|
| Python | 3.12.10 |
| pandas | 3.0.6 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| lasio | 0.32 |
| XGBoost | 3.4.1 |

XGBoost is included for model comparison experiments. The selected final candidate uses scikit-learn's `HistGradientBoostingRegressor`.

### Dependency Installation

The project dependencies are pinned in `requirements.txt`.

Create and activate a virtual environment:

```powershell id="h8y4np"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell id="r5f3wv"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the installed dependencies:

```powershell id="e9z7ua"
pip check
```

### Rebuilding the Processed Dataset

The main processed dataset is:

`data/processed/phif_ml_dataset_v1.csv`

It can be regenerated from the original Volve LAS source files using:
> **Overwrite warning:** This script writes the processed dataset and its
> data-quality audit output directly to the project directories. Running
> it may overwrite existing artifacts. Preserve and verify current outputs
> before execution. Run this script only when dataset regeneration is
> explicitly intended.

```powershell id="n3m5kb"
python scripts/build_phif_ml_dataset.py
```

The original LAS source files must be available locally before running the dataset-building script.

Raw LAS files are not included in normal Git tracking. Users must obtain the source data separately from the official Volve dataset.

After rebuilding the dataset, verify the following expected conditions:

- The processed dataset contains 47,746 rows.
- All nine wells are represented.
- Required model features and the PHIF target are available.
- There are no missing values in the final ML-ready feature and target columns.

### Reproducing the Fixed Well-Based Split

The script writes the train, validation, test, and split metadata CSV files
directly to `data/processed/`. Running it may overwrite existing partition
artifacts. Preserve and verify the current files before execution. Do not
run it merely to inspect an existing project state.

**Overwrite warning:** Execute this script only when regeneration of the
well-based split is explicitly intended.

Run:

```powershell id="x7b2qc"
python scripts/create_well_split.py
```

This script reads `phif_ml_dataset_v1.csv` directly and produces:

- `data/processed/phif_train.csv`
- `data/processed/phif_validation.csv`
- `data/processed/phif_test.csv`
- `data/processed/phif_well_split_metadata.csv`

The configured partitions are:

| Partition | Wells | Rows |
|---|---:|---:|
| Train | 6 | 24,516 |
| Validation | 1 | 11,936 |
| Test | 2 | 11,294 |
| **Total** | **9** | **47,746** |

The script validates well assignments, partition overlap, and total row counts before saving the outputs.

After execution, verify that:

- The three partitions contain no overlapping wells.
- All nine expected wells are represented across the partitions.
- The generated split metadata is consistent with the partition files.

### Reproducing Final PHIF OOF Predictions

The final OOF prediction workflow reads the main processed dataset directly:

`data/processed/phif_ml_dataset_v1.csv`

Run:

```powershell id="z4t8mh"
python scripts/generate_final_phif_predictions.py
```

The script performs the following operations:

1. Loads the main processed dataset.
2. Sorts observations by well and depth.
3. Generates `LOG10_RT` and the two 11-sample local variability features.
4. Identifies eligible rows with complete model features and PHIF targets.
5. Performs leave-one-well-out prediction across all nine wells.
6. Fits a separate `HistGradientBoostingRegressor` for each fold using the other wells.
7. Fits another model using all eligible observations to generate full-data fitted predictions.
8. Calculates OOF evaluation metrics by well.
9. Saves the prediction and evaluation artifacts.

The generated outputs are:

- `reports/models/final_phif_oof_predictions.csv`
- `reports/models/final_phif_full_predictions.csv`
- `reports/models/final_phif_oof_metrics_by_well.csv`

The OOF prediction output is intended for cross-well evaluation and characterization. The full-data fitted predictions are generated for descriptive analysis and must not be treated as independent performance estimates.

Verify that the OOF prediction output contains 47,746 rows across nine wells and has no missing predictions.

**Important distinction:** The final OOF workflow does not read `phif_train.csv`, `phif_validation.csv`, or `phif_test.csv`. It constructs its own leave-one-well-out training and prediction partitions directly from the main processed dataset.

The fixed well-based split and the LOO prediction workflow therefore serve different evaluation purposes and should not be treated as interchangeable.

### Reproducing OOF Characterization

To regenerate OOF characterization summaries:

```powershell id="c8v3jt"
python scripts/analyze_phif_oof_characterization.py
```

The characterization workflow produces well-level and depth-window summaries based on the final OOF prediction outputs.

### Reproducing Well-Profile Figures

To regenerate well-profile figures:

```powershell id="p6w3na"
python scripts/plot_phif_oof_profiles.py
```

### Reproducing the Cross-Well Comparison Figure

To regenerate the cross-well comparison figure:

```powershell id="f3x6rw"
python scripts/plot_phif_oof_well_comparison.py
```

These commands assume the processed dataset, the required prediction outputs, and the required Python packages are available.

Reproducing the complete workflow requires the original source data and execution of the relevant preprocessing, modeling, and characterization scripts in the documented order.

For strict reproducibility, users should also retain the source dataset version, configuration, dependency versions, and generated audit metadata used for each run.

## 11. Limitations and Next Steps

### Current Limitations

The current project has several important limitations:

- Only nine wells are available for the present PHIF workflow.
- Well-to-well variation remains substantial.
- Two wells have negative OOF R² values.
- Local variability features improve aggregate performance but do not eliminate cross-well generalization failures.
- The historical holdout confirmation is not a pristine untouched test estimate.
- PHIF target provenance limits the strength of geological claims.
- Independent geological and petrophysical validation is still required.
- The source depth units must be verified before assigning physical meaning to depth-window widths.
- Feature importance does not establish causal geological relationships.

### Next Steps

The next project phase focuses on:

1. Completing the final documentation review.
2. Verifying that README references match the actual repository files.
3. Reviewing generated model and characterization artifacts.
4. Confirming dependency consistency and reproducibility.
5. Completing repository hygiene checks.
6. Performing a final Git review before staging or committing.

The current results provide a foundation for further reservoir characterization research, but additional geological context and independent validation are necessary before operational use.

## 12. Source and Acknowledgment

The source dataset is the Volve Field Dataset distributed by Equinor.

Users should consult the official source for dataset documentation, provenance, and applicable usage terms.

[Equinor — Volve Field Data](https://www.equinor.com/energy/volve-field-data)

This project is an independent analytical workflow for research and portfolio demonstration.

Model outputs should be treated as decision-support evidence rather than definitive reservoir engineering conclusions.


## Data Attribution and Licensing

### Volve Field Dataset

This project uses data originating from the Volve Field Dataset,
provided by Equinor and the former Volve license partners:

- Equinor ASA
- ExxonMobil Exploration & Production Norway AS
- Bayerngas Norge AS

The dataset is made available under the Equinor Terms and Conditions
for Use of License to Data – Volve.

Official source:
https://www.equinor.com/energy/volve-data-sharing

License terms:
https://cdn.equinor.com/files/h61q9gi9/global/de6532f6134b9a953f6c41bac47a0c055a3712d3.pdf?equinor-hrs-terms-and-conditions-for-licence-to-data-volve.pdf=

This repository contains independent research code, documentation,
and analytical outputs developed using the Volve dataset.

Certain processed datasets and analytical outputs may constitute
Adapted Material under the Equinor Volve license. Their use and
redistribution remain subject to the applicable Volve license terms.

This project is not affiliated with, sponsored by, or endorsed by
Equinor or the former Volve license partners.

### Software License

Original project source code is licensed under the MIT License.
The MIT License does not apply to the original Volve dataset or
Adapted Material derived from it.

See [DATA_LICENSE.md](DATA_LICENSE.md) for details.




