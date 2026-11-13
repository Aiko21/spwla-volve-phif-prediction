# SPWLA Volve — Project History & Execution Flow



> **Project:** PHIF Prediction from Well Logs  

> **Dataset:** Volve Field Dataset / SPWLA Volve repository  

> **Purpose:** Maintain a chronological, reproducible record of the project from raw-data audit through ML evaluation and final deployment/publication.  

> **Status:** PHIF model selection and OOF characterization completed; documentation and repository finalization in progress  

> **Git policy:** `git add` / `git commit` will be performed only after the main analysis and modeling process is finalized and the final repository structure has been reviewed.



---



## 1. Project Objective



The project investigates whether **PHIF (Final Porosity)** can be predicted from well-log measurements available in the input LAS files.



The current modeling scope uses:



- **Target:** `PHIF`

- **Input features:** `GR`, `RT`

- **Derived feature:** `LOG10_RT`

- **Unit of generalization:** well

- **Primary evaluation principle:** unseen-well generalization



The project follows a **business/technical problem first, model second** workflow, with emphasis on:



1. Data provenance

2. Data quality

3. Target readiness

4. Feature availability

5. Leakage prevention

6. Well-based splitting

7. Baseline modeling

8. Generalization analysis

9. Error analysis

10. Reproducibility

11. GitHub-ready documentation

12. Medium-ready project storytelling



---



# 2. Project Execution Flow



```text

Volve / SPWLA Source Data

          │

          ▼

1. Repository & Dataset Inventory

          │

          ▼

2. LAS File / Header / Curve Audit

          │

          ▼

3. Input–Output Depth Alignment

          │

          ▼

4. Target Readiness & Anomaly Analysis

          │

          ▼

5. Feature Availability Audit

          │

          ▼

6. PHIF ML Dataset Construction

          │

          ▼

7. Well-Based Train / Validation / Test Split

          │

          ▼

8. Train-Only EDA

          │

          ▼

9. Feature / Distribution / Missingness Analysis

          │

          ▼

10. Baseline Models

          │

          ├── Linear Regression

          └── Random Forest

          │

          ▼

11. XGBoost Baseline

          │

          ▼

12. Generalization / Group CV

          │

          ▼

13. Depth / Feature CV Analysis

          │

          ▼

14. Error Analysis

          │

          ▼

15. Model Comparison & Final Selection

          │

          ▼

16. Final Interpretation & Limitations

          │

          ▼

17. GitHub Documentation / README

          │

          ▼

18. Final Git Review

          │

          ▼

19. git add → staged review → commit

          │

          ▼

20. Medium Article / Project Publication

```



---



# 3. Chronological History



## Phase 0 — Repository Baseline



**### Initial repository state**



The project started from an existing Git repository containing the original Volve dataset assets.



Current original Git history:



```text

c293edf (HEAD -> main, origin/main, origin/HEAD) first commit

```



Important source files already tracked by the original repository:



- `VOLVE_INVENTORY.xlsx`

- `Well_picks_Volve_v1.dat`



The raw well folders remain available locally but are excluded from the future GitHub working tree through `.gitignore`.



**### Git strategy**



A deliberate decision was made:



> Do not run `git add` or `git commit` during the active research/modeling phase.



The repository will be staged only after:



- the modeling process is complete,

- the final outputs are known,

- the project structure is finalized,

- unnecessary generated artifacts are excluded,

- README/documentation are complete.



---



# 4. Phase 1 — Dataset Inventory & Structural Audit



The Volve repository was audited to understand the available source material.



**### Inventory findings**



The repository contained:



- LAS files

- DLIS files

- PDF reports

- multiple well folders

- source inventory files

- formation-top information



A repository inventory was generated and the available file types were examined before modeling.



**### Purpose**



This phase established:



- what data exists,

- which wells are available,

- which formats can be processed,

- which files contain petrophysical inputs,

- which files contain petrophysical outputs.



---



# 5. Phase 2 — LAS Curve & Target Audit



LAS files were inspected for:



- headers

- curve names

- units

- descriptions

- depth ranges

- data quality

- target availability



Nine well input/output pairs were identified as sufficiently aligned for the current PHIF modeling workflow.



Candidate output targets included:



- `PHIF`

- `SW`

- `VSH`

- `KLOGH`



For the current ML task, `PHIF` was selected as the target.



---



# 6. Phase 3 — Target Readiness



Input/output LAS curves were aligned by rounded depth.



Depth matching used:



```python

DEPTH.round(4)

```



The resulting PHIF-ready dataset contained:



- **47,746 ML-ready rows**

- **9 wells**

- no missing `GR`

- no missing `RT`

- no missing `PHIF`



Per-well ML-ready rows:



| Well | Rows |

|---|---:|

| 15_9-F-1 | 2,872 |

| 15_9-F-1A | 2,281 |

| 15_9-F-1B | 2,105 |

| 15_9-F11 T2 | 2,017 |

| 15_9-F11A | 1,533 |

| 15_9-F11B | 13,933 |

| 15_9-F12 | 2,647 |

| 15_9-F15D | 11,936 |

| 15_9-F1C | 8,422 |



Overall PHIF statistics:



- Mean: **0.153370**

- Standard deviation: **0.076576**

- Minimum: **0.001**

- Median: **0.1703**

- Maximum: **0.3526**



---



# 7. Phase 4 — Target Quality Findings



Additional target investigations were performed.



**### PHIF**



PHIF was observed within:



```text

0.001 – 0.3526

```



No negative PHIF values or values above 1 were observed.



A repeated lower value around `0.020` was observed in several wells. This was treated as a possible floor/censoring-like behavior rather than automatically deleting the observations.



**### SW**



Approximately 30.5% of valid SW values were exactly `1.0`.



**### VSH**



VSH contained values above 1.



The anomaly was traced to the documented formula:



```text

(GR - 15) / (250 - 15)

```



in `15_9-F-1`.



Values above 250 GR naturally produced VSH values above 1 under that formula.



**### KLOGH**



KLOGH showed strong skew and repeated low values around `0.001`.



These target variables were not used as model inputs because they are not available as input features in the selected ML workflow.



---



# 8. Phase 5 — Feature Availability



An input-feature availability audit established that the common usable input curves across all nine selected input wells were:



- `GR`

- `RT`



Other candidate curves such as:



- `SW`

- `VSH`

- `KLOGH`



were not available as valid input predictors across the input LAS files.



Therefore the modeling problem was constrained to:



```text

PHIF \~ GR + RT

```



with `LOG10_RT` considered as a derived representation of RT.



---



# 9. Phase 6 — RT Missingness & Extreme Values



RT was investigated separately because it contains strong well-dependent missingness and extreme values.



Examples of missing RT intervals included long contiguous depth intervals in some wells.



However, within the PHIF-valid modeling rows:



> all PHIF-valid rows also had valid RT.



Therefore the current ML dataset did not require RT imputation.



**### RT extreme values**



The training data contained extreme RT values, including:



```text

62290.77

```



The value occurred repeatedly and formed plateau-like intervals.



The current decision was:



> Do not clip or delete RT extremes prematurely.



Instead, both:



- raw `RT`

- `LOG10_RT`



were evaluated.



---



# 10. Phase 7 — Well-Based Data Split



A fixed well-based split was created to avoid row-level leakage between the same wells.



**### TRAIN**



- `15_9-F-1A`

- `15_9-F-1B`

- `15_9-F11 T2`

- `15_9-F11A`

- `15_9-F11B`

- `15_9-F12`



**### VALIDATION**



- `15_9-F15D`



**### TEST**



- `15_9-F-1`

- `15_9-F1C`



Row counts:



| Split | Rows |

|---|---:|

| Train | 24,516 |

| Validation | 11,936 |

| Test | 11,294 |

| Total | 47,746 |



This split is intended to test the important question:



> Can a model trained on some wells generalize to previously unseen wells?



---



# 11. Phase 8 — Train-Only EDA



EDA was performed on the training set to avoid using validation/test information for model decisions.



**### GR**



GR showed a reasonably consistent inverse relationship with PHIF.



Training correlations:



- Pearson GR–PHIF: **-0.4921**

- Spearman GR–PHIF: **-0.5081**



**### RT**



RT showed a more complex relationship.



Training correlations:



- Pearson RT–PHIF: **+0.0559**

- Spearman RT–PHIF: **+0.4122**



This difference demonstrated that raw Pearson correlation was insufficient to describe the relationship between RT and PHIF.



---



# 12. Phase 9 — Baseline Modeling



Baseline configurations tested:



```text

GR

GR + RT

GR + LOG10_RT

GR + RT + LOG10_RT

```



Models:



- Dummy median baseline

- Linear Regression

- Random Forest



The strongest validation baseline was:



```text

Linear Regression

Features:

GR + RT + LOG10_RT

```



Validation:



- MAE: **0.04726**

- RMSE: **0.05662**

- R²: **0.42204**



After refitting on train + validation and evaluating on the unseen test wells:



- MAE: **0.051721**

- RMSE: **0.063289**

- R²: **0.289770**



---



# 13. Phase 10 — XGBoost Baseline



XGBoost was tested as a nonlinear baseline.



Configuration included:



- `n_estimators=300`

- `max_depth=6`

- `learning_rate=0.05`

- `subsample=0.8`

- `random_state=42`



Validation:



- MAE: **0.049779**

- RMSE: **0.063103**

- R²: **0.282148**



Final test:



- MAE: **0.055212**

- RMSE: **0.070617**

- R²: **0.115772**



The initial XGBoost baseline therefore did not outperform the linear baseline.



---



# 14. Phase 11 — Feature Distribution Shift



Feature distributions were compared between train, validation and test wells.



The test wells were not simply outside the training feature range.



For example, only a very small number of F1C GR observations fell below the training minimum, while RT and LOG10_RT remained within the training range.



This led to an important working hypothesis:



> The difficulty is more likely related to conditional/regime shift — different wells mapping similar GR/RT values to different PHIF distributions — rather than simple feature extrapolation.



This hypothesis is subject to further validation through group/leave-one-well-out evaluation.



---



# 15. Phase 12 — Error Analysis



The final linear model showed clear regression-to-the-mean behavior.



**### Test performance by well**



**#### 15_9-F-1**



- MAE: **0.033814**

- RMSE: **0.042022**

- R²: **0.144030**



Mean actual PHIF:



```text

0.081556

```



Mean predicted PHIF:



```text

0.099754

```



The model overpredicted this well.



**#### 15_9-F1C**



- MAE: **0.057827**

- RMSE: **0.069060**

- R²: **-0.051926**



Mean actual PHIF:



```text

0.177187

```



Mean predicted PHIF:



```text

0.151412

```



The model underpredicted this well.



**### Interpretation**



The model tends to compress predictions toward the global training relationship.



This is particularly problematic for:



- low-PHIF intervals

- high-PHIF intervals

- well-specific regimes



The largest errors were concentrated in F1C and included cases where actual PHIF was near the observed floor while the model predicted substantially higher values.



---



# 16. Phase 13 — Generalization Evaluation



Group/leave-one-well-out and depth-related evaluation scripts were created to investigate whether the fixed train/validation/test result is representative.



Current analysis artifacts include:



```text

reports/models/phif_group_cv_predictions.csv

reports/models/phif_group_cv_results.csv

reports/models/phif_group_cv_summary.csv



reports/models/phif_depth_feature_cv_predictions.csv

reports/models/phif_depth_feature_cv_results.csv

reports/models/phif_depth_feature_cv_summary.csv

reports/models/phif_depth_feature_cv_per_well.csv

```



> **Important:** final interpretation of these evaluations should be added here only after the actual results are reviewed. Do not infer or fabricate metrics from file names alone.



---



# 17. Historical Modeling Conclusion — Superseded by Later Experiments



The current evidence supports the following provisional conclusions:



1. `GR` is the strongest simple predictor among the currently available common input features.

2. `RT` contains useful information, but its relationship with PHIF is strongly nonlinear and well-dependent.

3. `LOG10_RT` provides a useful representation for highly skewed RT.

4. Linear Regression currently provides a stronger baseline than the tested default Random Forest and XGBoost configurations on the fixed unseen-well test.

5. Test performance is materially worse than validation performance.

6. F1C demonstrates particularly difficult generalization.

7. The main modeling challenge appears to be **well/regime heterogeneity**, not simply feature extrapolation.

8. More complex modeling should only be justified if it improves unseen-well generalization rather than merely improving in-sample or random-split performance.



---



# 18. Repository Architecture Work



The repository is being reorganized before the first new project commit.



Current target architecture:



```text

spwla_volve/

│

├── .gitignore

├── README.md

│

├── VOLVE_INVENTORY.xlsx

├── Well_picks_Volve_v1.dat

│

├── data/

│   └── processed/

│

├── scripts/

│   ├── inventory / audit scripts

│   ├── preprocessing scripts

│   ├── EDA scripts

│   └── modeling / evaluation scripts

│

└── reports/

    ├── data_quality/

    ├── figures/

    └── models/

```



**### Completed repository cleanup**



- Created `.gitignore`

- Ignored `.venv/`

- Ignored raw well directories

- Ignored raw LAS/DLIS/PDF files

- Ignored temporary `project_tree.txt`

- Moved Python scripts from repository root into `scripts/`

- Moved root-level audit CSV outputs into `reports/data_quality/`

- Renamed `Readme.md` to `README.md`

- Preserved original tracked source files:

  - `VOLVE_INVENTORY.xlsx`

  - `Well_picks_Volve_v1.dat`



**### Git policy**



No new project files have been staged or committed yet.



The original repository commit remains:



```text

c293edf first commit

```



The new project work remains uncommitted until the research pipeline is finalized.



---



# 19. Documentation Policy Going Forward



This file is a **living project history**.



Every major project phase should add an entry containing:



1. Date

2. Objective

3. Script/notebook used

4. Data involved

5. Method

6. Key findings

7. Decision made

8. Output artifacts

9. Next action



Do not rewrite earlier findings merely to make the story cleaner.



If a later experiment disproves an earlier hypothesis, record the change explicitly:



```text

Previous hypothesis

        ↓

New evidence

        ↓

Revised conclusion

        ↓

Decision

```



This preserves scientific traceability and makes the final history suitable for conversion into a Medium article.



---



# 20. Medium Article Conversion Plan



At project completion, this history can be transformed into a Medium article structure:



```text

Title

│

├── The Problem

├── Why the Volve Dataset?

├── Understanding the Data

├── Data Quality Surprises

├── Building a Leakage-Safe ML Dataset

├── Why Well-Based Splitting Matters

├── Baseline Modeling

├── Why More Complex Models Did Not Automatically Win

├── The Generalization Problem

├── Error Analysis

├── What the Model Can and Cannot Do

├── Final Model

├── Lessons Learned

└── Reproducibility / GitHub

```



The Medium article should be written from the actual experiment history rather than reconstructed from memory at the end.



---



# 21. Finalization Checklist



Before the first new project commit:



- [ ] Complete group/leave-one-well-out evaluation

- [ ] Review depth-based evaluation

- [ ] Decide final model

- [ ] Complete final error analysis

- [ ] Review data leakage risks

- [ ] Review feature availability

- [ ] Review train/validation/test design

- [ ] Decide which processed datasets belong in GitHub

- [ ] Decide which large prediction artifacts should remain ignored

- [ ] Finalize README

- [ ] Finalize project history

- [ ] Add methodology documentation

- [ ] Review repository tree

- [ ] Review `.gitignore`

- [ ] Run `git status`

- [ ] Run `git diff --cached` after staging

- [ ] Only then create the first project commit



---



## Change Log



**### 2026-10-09**



- Audited current repository structure.

- Confirmed original Git repository and existing first commit.

- Confirmed source files already tracked by the original repository.

- Created `.gitignore`.

- Excluded `.venv`, raw well directories, raw LAS/DLIS/PDF files, and temporary project tree output.

- Moved all root-level Python project scripts into `scripts/`.

- Moved root-level audit CSV outputs into `reports/data_quality/`.

- Renamed `Readme.md` to `README.md`.

- Preserved `VOLVE_INVENTORY.xlsx` and `Well_picks_Volve_v1.dat`.

- Confirmed no new files have been staged or committed.

- Established this document as the living project history for future Medium publication.



---



## Document Status



**Status:** Living document  

**Last updated:** 2026-10-09  

**Next update:** After the next completed analysis/modeling phase.







## Cross-Well Generalization Diagnostics



**### RT Incremental Value by Well**



Leave-One-Well-Out evaluation showed that RT provides additional predictive information, but its value is strongly dependent on the held-out well.



For XGBoost, the incremental effect of adding RT relative to GR-only varied substantially across wells. RT improved performance in some wells, was approximately neutral in others, and reduced performance in several wells.



This indicated that RT should not be assumed to have a universally transferable relationship with PHIF.



**### RT–PHIF Relationship Regimes**



Within-well analysis of LOG10(RT) versus PHIF revealed different relationship regimes:



- Some wells showed a strongly decreasing relationship.

- Some wells showed a positive relationship.

- Some wells showed weak or mixed relationships.

- At least one well showed a non-monotonic/U-shaped pattern.



GR remained consistently negatively associated with PHIF across all wells, while the RT–PHIF relationship changed substantially between wells.



This established GR as a relatively more stable cross-well signal and RT as a regime-dependent signal.



**### GR-Conditioned RT Analysis**



RT–PHIF relationships were further examined within descriptive GR regimes.



The analysis showed that GR conditioning could change the apparent RT–PHIF relationship in some wells, suggesting possible GR × RT interaction. However, the effect was not universal across wells.



Therefore, the observed interaction was treated as a hypothesis rather than a universal relationship.



**### Blind-Well Interaction Experiment**



A strict 9-fold Leave-One-Well-Out experiment compared:



1. `GR + LOG10_RT`

2. `GR + LOG10_RT + GR × LOG10_RT`



The simple global interaction feature produced negligible overall improvement in cross-well generalization.



The interaction feature was therefore not retained as a primary feature.



Decision:



- DEPTH: not retained

- Global standardization-derived features: not retained

- Simple GR × LOG10(RT): not retained

- GR: retained

- LOG10(RT): retained for further investigation



**### Cross-Well Failure Regimes**



A well-level diagnostic classified XGBoost generalization into:



- GOOD: 3 wells

- MODERATE: 1 well

- POOR: 5 wells



The analysis showed that poor generalization was not explained simply by PHIF target distribution because GOOD and POOR wells had similar PHIF medians.



RT behavior differed substantially between wells, reinforcing the conclusion that RT is regime-dependent.



**### Covariate vs Relationship Shift Diagnostic**



A true Leave-One-Well-Out shift diagnostic was performed.



For each held-out well, its feature distribution and feature–target relationships were compared against the other eight wells.



The result showed that input-distribution shift alone did not explain generalization failure.



GOOD wells could exhibit substantial covariate shift while still generalizing well.



Relationship shift, particularly the change in the GR–PHIF relationship, showed a stronger exploratory association with XGBoost R² than simple covariate shift.



Exploratory Spearman correlation with XGBoost R²:



- GR relationship shift: -0.5667

- LOG10(RT) relationship shift: -0.1833

- Covariate shift index: -0.1333

- Relationship shift index: -0.3000

- Total diagnostic shift: -0.1333



Because only nine wells are available, these values are treated as diagnostic evidence and hypothesis generation, not statistical proof.



**### Important Failure Regimes**



Two particularly informative failure cases were identified:



**F12**



- XGBoost R²: approximately -0.557

- Large GR distribution shift

- Relatively small RT relationship shift

- Indicates a failure regime where feature distribution, particularly GR, may be important.



**F1C**



- XGBoost R²: approximately -0.420

- Relatively small GR median shift

- Larger relationship shifts

- Consistent with a potential relationship/concept shift rather than simple feature-range extrapolation.



F15D provided an important counter-example:



- XGBoost R²: approximately +0.416

- Noticeable covariate shift

- Very small relationship shift



This suggests that distribution shift does not automatically cause failure when the underlying feature–target relationship remains relatively stable.



**### Current Generalization Hypothesis**



The current working hypothesis is:



> Cross-well PHIF prediction difficulty is driven more by changes in feature–target relationships than by input-distribution shift alone, although different wells can fail through different mechanisms.



The project therefore moves from simple feature/model selection toward understanding and modeling cross-well regime heterogeneity.



No Git commit was made at this stage.



**===================================**



## HGB Cross-Well Generalization Experiment



**### Objective**

Evaluate whether HistGradientBoostingRegressor improves blind-well PHIF generalization compared with the existing Linear Regression and XGBoost baselines.



**### Validation Design**

- Strict 9-fold Leave-One-Well-Out cross-validation.

- Each fold holds out one complete well.

- Features:

  - GR

  - LOG10_RT

- No random row-level split.

- Random state = 42.

- HGB configuration:

  - learning_rate = 0.05

  - max_iter = 300

  - max_leaf_nodes = 15

  - min_samples_leaf = 30

  - l2_regularization = 1.0



**### Aggregate Results**



| Model | Mean MAE | Mean RMSE | Mean R² |

|---|---:|---:|---:|

| Linear | 0.052988 | 0.065010 | -0.028520 |

| XGBoost | 0.050251 | 0.064764 | -0.004358 |

| HGB | 0.049533 | 0.063745 | 0.023814 |



HGB achieved the best average MAE, RMSE, and R² among the three evaluated models.



**### Per-Well Findings**



HGB achieved the best R² on 7 of 9 held-out wells.



Good HGB generalization:

- F15D: R² = 0.467780

- F11B: R² = 0.403748

- F-1B: R² = 0.296515

- F-1: R² = 0.228356



Poor HGB generalization:

- F11A: R² = -0.039710

- F11 T2: R² = -0.093843

- F-1A: R² = -0.120926

- F1C: R² = -0.386184

- F12: R² = -0.541412



**### Comparison Against XGBoost**



HGB improved R² over XGBoost in 6 of 9 wells.



Mean R² improvement:

- HGB vs XGBoost: +0.028172



Mean MAE improvement:

- HGB vs XGBoost: +0.000718



**### Comparison Against Linear Regression**



HGB improved R² over Linear Regression in 7 of 9 wells.



Mean R² improvement:

- HGB vs Linear: +0.052334



Mean MAE improvement:

- HGB vs Linear: +0.003455



However, HGB degraded relative to Linear on F11A and F1C.



**### Interpretation**



HGB is currently the strongest generalization baseline for blind-well PHIF prediction.



However, the persistence of five poor-generalization wells indicates that increased model capacity alone does not solve cross-well heterogeneity.



Important failure cases:



- F12 remains the most difficult well despite substantial improvement over Linear Regression.

- F1C remains a strong failure case despite HGB improving over XGBoost.

- F11A demonstrates that a simpler linear model can outperform nonlinear tree-based models under some well-specific regimes.



These results strengthen the hypothesis that cross-well PHIF prediction is limited not only by model capacity but also by well-specific changes in feature-target relationships and geological/log-response regimes.



**### Decision**



- HGB is retained as the current champion generalization baseline.

- No hyperparameter optimization is performed yet.

- Further model tuning is deferred until the causes of well-specific generalization failure are investigated.

- The next analysis will focus on HGB failure regimes and compare GOOD versus POOR wells using covariate shift, feature-target relationship shift, and RT usefulness diagnostics.



**### Current Working Hypothesis**



Cross-well PHIF prediction difficulty is driven more by changes in feature-target relationships than by input-distribution shift alone, although different wells may fail through different mechanisms.



This remains an exploratory hypothesis because only nine wells are available for cross-well evaluation.



**===============================**







## HGB Failure Regime Analysis



**### Objective**



Investigate why the HGB model generalizes well to some wells but fails on others under strict Leave-One-Well-Out cross-validation.



**### Validation Integrity**



- 9 unique wells analyzed.

- 9 diagnostic rows.

- No duplicate wells remained after merge validation.

- All diagnostic analyses use blind-well evaluation results.

- Exploratory correlations are based on only 9 wells and are therefore hypothesis-generating rather than statistically conclusive.



**### HGB Generalization Regimes**



GOOD wells:

- F15D: HGB R² = 0.467780

- F11B: HGB R² = 0.403748

- F-1B: HGB R² = 0.296515

- F-1: HGB R² = 0.228356



POOR wells:

- F11A: HGB R² = -0.039710

- F11 T2: HGB R² = -0.093843

- F-1A: HGB R² = -0.120926

- F1C: HGB R² = -0.386184

- F12: HGB R² = -0.541412



**### Covariate Shift vs Relationship Shift**



Mean diagnostic comparison:



| Regime | HGB R² | Covariate Shift | Relationship Shift |

|---|---:|---:|---:|

| GOOD | 0.349100 | 23.447871 | 0.464013 |

| POOR | -0.236415 | 19.103645 | 0.690462 |



POOR wells show higher mean relationship shift but lower mean covariate shift than GOOD wells.



This indicates that input-distribution shift alone does not explain cross-well model failure.



**### Exploratory Spearman Analysis**



| Diagnostic | Spearman correlation with HGB R² |

|---|---:|

| GR relationship shift | -0.566667 |

| Relationship shift index | -0.300000 |

| Covariate shift index | -0.133333 |

| Total diagnostic shift | -0.133333 |

| LOG10_RT relationship shift | -0.183333 |

| GR median shift | +0.100000 |

| LOG10_RT median shift | +0.383333 |



The strongest exploratory association was between GR–PHIF relationship shift and HGB generalization performance.



Because only nine wells are available, these correlations are treated as exploratory evidence rather than statistical proof.



**### Distinct Failure Mechanisms**



F12:

- HGB R² = -0.541412

- Covariate shift index = 34.775935

- Relationship shift index = 0.128221

- GR median shift = 32.013097



F12 is therefore a candidate distribution/covariate-shift-dominated failure case.



F1C:

- HGB R² = -0.386184

- Covariate shift index = 8.248301

- Relationship shift index = 0.473907

- GR median shift = 0.235100



F1C is a candidate relationship/concept-shift-dominated failure case.



These examples indicate that poor generalization may occur through different mechanisms across wells.



**### Important Counterexample**



F-1 has:

- Covariate shift index = 30.889157

- Relationship shift index = 1.276265

- HGB R² = 0.228356



Therefore, high diagnostic shift does not automatically imply model failure.



Relationship shift appears informative but is not sufficient by itself to determine generalization success.



**### RT Transferability**



The incremental RT contribution is highly well-dependent.



Examples:

- F12: ΔR² = +0.644924

- F11 T2: ΔR² = +0.130943

- F-1A: ΔR² = +0.148476

- F1C: ΔR² = -0.324934

- F11A: ΔR² = -0.238482



This demonstrates that a feature can provide predictive value in one modeling context while remaining poorly transferable across wells.



**### Current Hypothesis**



Cross-well PHIF generalization is influenced more by changes in feature-target relationships than by covariate distribution shift alone, although different wells may fail through different mechanisms.



**### Modeling Decision**



- HGB remains the current champion blind-well generalization model.

- No further hyperparameter tuning is performed at this stage.

- No adaptive target-well calibration is introduced yet.

- The next phase focuses on explainability and geoscience interpretation of the champion HGB model.



**=======================**







## Explainability — Held-Out Permutation Importance for HGB



**### Objective**



Evaluate which input logs contribute most to blind-well PHIF prediction and determine whether feature importance is consistent across wells.



The analysis used the champion HGB configuration identified from strict 9-fold Leave-One-Well-Out cross-validation:



- Model: HistGradientBoostingRegressor

- Features:

  - GR

  - LOG10_RT

- Target:

  - PHIF

- Permutation importance computed separately on each held-out well

- Scoring metric:

  - negative MAE

- Repeats:

  - 10

- random_state:

  - 42



**### Aggregated held-out importance**



| Feature | Mean Importance | Median Importance | Positive Wells | Rank |

|---|---:|---:|---:|---:|

| LOG10_RT | 0.011057 | 0.013128 | 9/9 | 1 |

| GR | 0.010152 | 0.010027 | 9/9 | 2 |



LOG10_RT had slightly higher average held-out permutation importance than GR, but the difference was small (0.000905). Therefore, both features should be considered important rather than treating RT as overwhelmingly dominant.



Both features had positive permutation importance in all 9 held-out wells, indicating that both provide measurable predictive information across the dataset.



**### Well-dependent feature importance**



LOG10_RT was the dominant feature in:



- 15_9-F-1

- 15_9-F-1A

- 15_9-F11 T2

- 15_9-F12



GR was the dominant feature in:



- 15_9-F-1B

- 15_9-F11A

- 15_9-F11B

- 15_9-F15D

- 15_9-F1C



Therefore, feature dominance is not stationary across wells.



**### Key observations**



The magnitude of feature importance varied substantially between wells.



For LOG10_RT, importance ranged from:



- 0.000516 in 15_9-F1C

- to 0.020797 in 15_9-F11 T2



For GR, importance ranged from:



- 0.003546 in 15_9-F-1A

- to 0.020588 in 15_9-F15D



This demonstrates that the predictive contribution of the same log can change substantially between wells.



**### Connection with generalization**



The explainability results reinforce previous cross-well diagnostics.



For 15_9-F1C:



- HGB R² = -0.386

- LOG10_RT importance = 0.000516

- Previous RT incremental ΔR² ≈ -0.325

- RT–PHIF relationship was weak



This provides converging evidence that RT provides limited transferable predictive value in F1C.



For 15_9-F12:



- HGB R² = -0.541

- LOG10_RT importance = 0.014405

- RT incremental ΔR² ≈ +0.645



This demonstrates that a feature can be highly predictive/useful to the model while the overall learned relationship still fails to generalize to a target well.



Therefore:



> Feature importance and cross-well generalization are different concepts.



**### Interpretation**



The analysis supports the hypothesis that blind-well PHIF prediction is regime-dependent.



The model does not rely on a single universally dominant log. Instead, the relative contribution of GR and LOG10_RT changes between wells.



This provides additional evidence that cross-well performance is influenced by changes in feature–target relationships rather than only by differences in input distributions.



**### Methodological caution**



Permutation importance measures model-dependent predictive contribution, not geological causality.



Because GR and RT may contain correlated or overlapping information, permutation importance should not be interpreted as proof that one log is physically more important than another.



The conclusion is limited to the evaluated HGB model, feature set, dataset, and held-out permutation procedure.



**### Outputs**



- reports/models/phif_hgb_permutation_importance.csv

- reports/models/phif_hgb_permutation_importance_by_well.csv





















# ---------------------------------

# 27 Okt 2026





---



# 22. Phase 14 — Local Feature Engineering and Final PHIF Model Candidate



**Date:** 2026-10-27



## Objective



Investigate whether local vertical variability in well-log measurements improves PHIF prediction across wells, then consolidate the prediction outputs for reservoir characterization.



## Feature engineering



The following feature families were evaluated:



- Base: `GR`, `LOG10_RT`

- Local variability: `GR_LOCAL_STD_11`, `LOGRT_LOCAL_STD_11`

- Local change: gradient features

- Combined local features: local means, variability, and gradients



Feature selection was conducted using seven development wells, with F-1 and F1C reserved for a historical holdout confirmation.



The selected candidate was:



- Model: `HistGradientBoostingRegressor`

- Features: `GR`, `LOG10_RT`, `GR_LOCAL_STD_11`, `LOGRT_LOCAL_STD_11`

- Rolling window: 11 samples

- Parameters:

  - `learning_rate=0.05`

  - `max_iter=300`

  - `max_leaf_nodes=15`

  - `min_samples_leaf=30`

  - `l2_regularization=1.0`

  - `random_state=42`



## Development-only feature selection



The selected local variability configuration achieved:



- Mean MAE: 0.047241

- Mean RMSE: 0.062123

- Mean R²: 0.129152

- Positive R²: 5 of 7 development wells



The BASE configuration achieved mean MAE 0.050767 and mean R² 0.028356.



The MAE difference between `LOCAL_VARIABILITY / window 11` and `ALL_LOCAL / window 21` was very small. The simpler local variability configuration was retained as the candidate using MAE as the primary selection metric.



## Historical holdout confirmation



| Model | MAE | RMSE | R² |

|---|---:|---:|---:|

| BASE HGB | 0.054147 | 0.069308 | 0.148268 |

| LOCAL VARIABILITY HGB, window 11 | 0.050076 | 0.065491 | 0.239487 |



The local variability model improved aggregate holdout MAE by approximately 7.52% and RMSE by approximately 5.51%.



Both F-1 and F1C improved on MAE, RMSE, and R² relative to the BASE HGB model.



**Methodological qualification:** F-1 and F1C had already appeared in earlier leave-one-well-out experiments. These results are therefore described as historical holdout confirmation, not as a pristine untouched test estimate.



## Decision



The HGB model with local variability features is the current locked candidate for the PHIF characterization phase.



Further model, feature, and window hunting is paused. Any future model changes must be justified by a clearly defined new research question.



---



# 23. Phase 15 — Final OOF Prediction Dataset



**Date:** 2026-10-27



## Method



Generated leave-one-well-out predictions using the locked candidate configuration.



For each fold, the held-out well was excluded from training. Local rolling variability features were calculated independently within each well.



## Validation



- Input dataset: `data/processed/phif_ml_dataset_v1.csv`

- Input rows: 47,746

- Wells: 9

- OOF rows: 47,746

- OOF missing predictions: 0

- Full-data fitted prediction rows: 47,746

- Full-data missing predictions: 0



## OOF metrics by well



| Well | MAE | RMSE | R² | Mean residual |

|---|---:|---:|---:|---:|

| 15_9-F-1 | 0.027883 | 0.034151 | 0.434662 | +0.018808 |

| 15_9-F-1A | 0.042536 | 0.057616 | 0.086288 | +0.024571 |

| 15_9-F-1B | 0.035278 | 0.050370 | 0.323818 | -0.004601 |

| 15_9-F11 T2 | 0.041270 | 0.056178 | 0.158092 | +0.029830 |

| 15_9-F11A | 0.051941 | 0.065937 | 0.105435 | -0.025965 |

| 15_9-F11B | 0.043912 | 0.059311 | 0.426181 | +0.009096 |

| 15_9-F12 | 0.070789 | 0.088380 | -0.505645 | -0.061988 |

| 15_9-F15D | 0.040131 | 0.051171 | 0.527944 | +0.002617 |

| 15_9-F1C | 0.059649 | 0.074609 | -0.227784 | -0.033556 |



Residual convention: `PREDICTED_PHIF - PHIF`.



## Main findings



1. Performance varies materially between wells.

2. F15D has the strongest R² among the nine wells.

3. F12 has the largest MAE and strongest negative mean bias.

4. F1C also has negative R² and systematic underprediction.

5. Predicted vertical variability is generally lower than actual PHIF variability, consistent with smoothing in many wells.



Negative R² indicates that the predictions perform worse than the held-out well's mean-target reference for that fold. It does not mean every individual prediction is unusable.



## Output artifacts



- `reports/models/final_phif_oof_predictions.csv`

- `reports/models/final_phif_full_predictions.csv`

- `reports/models/final_phif_oof_metrics_by_well.csv`



**Important:** full-data fitted predictions are not independent estimates of generalization performance. OOF predictions are the primary source for the current cross-well characterization.



---



# 24. Phase 16 — OOF Reservoir Characterization



**Date:** 2026-10-27



## Method



Summarized OOF PHIF predictions and errors by well and by fixed-width depth windows of 10 depth units. Generated well-log profiles and a cross-well comparison figure.



## Output artifacts



- `reports/models/phif_oof_characterization_by_well.csv`

- `reports/models/phif_oof_characterization_by_depth_window.csv`

- `reports/figures/phif_oof_profiles/`

- `reports/figures/phif_oof_well_comparison.png`



The depth-window table contains 500 windows across nine wells.



## Interpretation



- Prediction bias is well-dependent.

- F12 and F1C exhibit substantial underprediction.

- F-1, F-1A, and F11 T2 exhibit positive mean bias.

- F15D has low mean bias and relatively strong OOF performance, although local prediction errors remain.

- Predicted variability is lower than actual variability in eight of nine wells when comparing the mean within-window standard deviations. F15D is the exception.



These observations indicate that aggregate performance alone is insufficient for interval interpretation.



## Geological and operational limitations



PHIF is the target curve supplied by the petrophysical dataset. This analysis evaluates the ability to reproduce that target, not independent laboratory porosity.



High predicted PHIF is not proof of high permeability, hydrocarbon presence, or commercial productivity. Reservoir-quality interpretations require additional geological context and independent evidence.



The 10-depth-unit aggregation is an exploratory summary scale, not a validated geological boundary or reservoir thickness. Depth units must be confirmed before describing these windows in metres.



## Current project decision



The project has moved from model development into PHIF characterization and interpretation.



No additional threshold optimization or model hunting is planned at this stage. Remaining work should focus on:



1. Final interpretation and limitations.

2. README and reproducibility documentation.

3. Repository and artifact review.

4. Final Git review before staging or committing.

5. Preparation of the publication narrative from the chronological history.





# 25. Phase 17 — Dataset Rebuild Reproducibility Verification



**Status:** Completed  

**Objective:** Verify that the PHIF machine learning dataset can be rebuilt reproducibly from the source LAS files.



**### Activities**

1. Reviewed and corrected the repository root path in `build_phif_ml_dataset.py`.

2. Verified the availability of the required LAS input and output files across the nine wells.

3. Rebuilt the processed PHIF machine learning dataset.

4. Checked depth alignment, required curves, missing values, and ML-ready records.

5. Compared the regenerated dataset and audit output against previously recorded SHA256 hashes.



**### Results**

- Final dataset: 47,746 rows across 9 wells.

- Required features and target: GR, RT, PHIF.

- No missing values in the final ML-ready dataset.

- Regenerated dataset SHA256 matched the recorded baseline:

  `2CFC824616717931391972D56D4266F9B835EBA36B507656549D22C44B99FC5A`



The dataset rebuild was verified as byte-for-byte reproducible against the recorded baseline.



---



# 26. Phase 18 — Well-Based Split Reproducibility Verification



**Status:** Completed  

**Objective:** Verify that the train, validation, and test datasets can be regenerated consistently using the established well-based split configuration.



**### Split Configuration**



| Split | Wells | Rows |

|---|---|---:|

| Train | 6 wells | 24,516 |

| Validation | 1 well | 11,936 |

| Test | 2 wells | 11,294 |

| **Total** | **9 wells** | **47,746** |



**### Activities**

1. Reviewed and corrected the repository root path in `create_well_split.py`.

2. Regenerated the train, validation, and test datasets.

3. Verified the well assignments and split metadata.

4. Checked that the split row counts sum to the complete ML dataset.

5. Compared all four regenerated split artifacts against their recorded SHA256 baselines.



**### Results**



All four artifacts matched their recorded SHA256 hashes:



- `phif_train.csv`

- `phif_validation.csv`

- `phif_test.csv`

- `phif_well_split_metadata.csv`



The well-based split was verified as byte-for-byte reproducible against the recorded baseline.



**### Conclusion**



Dataset construction and well-based partitioning have passed the reproducibility verification. The existing model results and prediction artifacts remain unchanged by this verification activity.



# 27. Phase 19 — Final Model Artifact Validation



**Status:** Completed  

**Date:** 2026-11-02



**### Objective**



Validate the consistency, completeness, and reproducibility of the final PHIF model selection and OOF characterization artifacts before repository finalization.



**### Activities Performed**



1. Inspected final model development selection results.

2. Verified the selected LOCAL_VARIABILITY feature set with an 11-sample window.

3. Validated historical holdout confirmation metrics.

4. Checked OOF metrics across all wells.

5. Verified OOF characterization summaries by well and depth window.



**### Validation Results**



**Final model selection:**



- Feature set: LOCAL_VARIABILITY

- Window: 11 samples

- Mean MAE: 0.047241

- Mean RMSE: 0.062123

- Mean R²: 0.129152

- Positive R²: 5 of 7 development wells



**Historical holdout confirmation:**



- Baseline HGB MAE: 0.054147

- Local variability HGB MAE: 0.050076

- Baseline HGB RMSE: 0.069308

- Local variability HGB RMSE: 0.065491

- Baseline HGB R²: 0.148268

- Local variability HGB R²: 0.239487



This result is reported as historical holdout confirmation, not as a pristine untouched test estimate.



**OOF validation:**



- Number of wells: 9

- Total OOF samples: 47,746

- Wells with negative R²: 2 (15_9-F12 and 15_9-F1C)



**OOF characterization:**



- Number of depth windows: 500

- Number of wells represented: 9



**### Key Findings**



The selected local variability feature configuration and the final OOF artifacts are consistent with the previously documented project results.



Cross-well generalization remains uneven. Negative OOF R² values on 15_9-F12 and 15_9-F1C indicate that model reliability must be assessed at the individual-well level.



**### Outputs Validated**



- `reports/models/final_phif_development_selection.csv`

- `reports/models/final_phif_holdout_confirmation.csv`

- `reports/models/final_phif_oof_metrics_by_well.csv`

- `reports/models/phif_oof_characterization_by_well.csv`

- `reports/models/phif_oof_characterization_by_depth_window.csv`



**### Conclusion**



Final model selection and OOF characterization artifacts passed the current consistency checks. These outputs are suitable for continued documentation and repository audit, subject to final review of the complete project artifacts.



No Git staging or commit was performed during this phase.


---

# 28. Phase 20 — Historical Model Comparison Consistency Audit and Correction

**Status:** Completed  
**Date:** 2026-11-03

## Objective

Ensure that the historical comparison of Linear Regression, XGBoost, and HistGradientBoosting (HGB) uses one consistent evaluation source, feature configuration, and well-based validation design.

## Initial Finding

The original `scripts/final_phif_model_comparison.py` combined results from two different evaluation files:

- `reports/models/phif_group_cv_results.csv` supplied Linear Regression and XGBoost metrics.
- `reports/models/phif_hgb_cv_results.csv` supplied HGB metrics.

These files were produced using different feature configurations:

| Evaluation source | Models used in the original comparison | Features |
|---|---|---|
| `phif_group_cv_results.csv` | Linear, XGBoost | GR, RT, LOG10_RT |
| `phif_hgb_cv_results.csv` | HGB | GR, LOG10_RT |

Although both evaluations used well-based cross-validation, combining their metrics did not provide a fully controlled, like-for-like model comparison because the feature sets differed.

## Root Cause

The original comparison script selected Linear and XGBoost results from `phif_group_cv_results.csv`, then selected only HGB rows from `phif_hgb_cv_results.csv`.

The second file already contained all three models — `LINEAR`, `XGBOOST`, and `HGB` — evaluated over the same nine Leave-One-Well-Out folds with the same two input features:

- `GR`
- `LOG10_RT`

Therefore, a consistent historical comparison could be produced entirely from `phif_hgb_cv_results.csv`.

## Corrective Actions

1. Inspected the original `final_phif_model_comparison.py`.
2. Confirmed the different feature configurations in the two historical evaluation sources.
3. Updated the comparison script to read only `reports/models/phif_hgb_cv_results.csv`.
4. Added validation for required input columns, expected model names, missing or invalid values, duplicate model–held-out-well combinations, consistent held-out well membership, and exactly nine evaluated wells.
5. Regenerated `reports/models/final_phif_model_comparison.csv`.
6. Compared the regenerated model summary against `reports/models/phif_hgb_cv_summary.csv` using a numerical tolerance of `1e-10`.

## Validation Results

| Validation item | Result |
|---|---|
| Evaluation source | `phif_hgb_cv_results.csv` |
| Models | Linear, XGBoost, HGB |
| Held-out wells | 9 |
| Total model evaluations | 27 |
| Duplicate model–well combinations | None |
| Well membership consistent across models | Yes |
| Summary comparison against source summary | All metrics matched |

All three models passed the numerical comparison for mean MAE, mean RMSE, and mean R².

## Corrected Historical Model Comparison

| Model | Mean MAE | Mean RMSE | Mean R² | Positive R² wells |
|---|---:|---:|---:|---:|
| HGB | 0.049533 | 0.063745 | 0.023814 | 4/9 |
| XGBoost | 0.050251 | 0.064764 | -0.004358 | 4/9 |
| Linear | 0.052988 | 0.065010 | -0.028520 | 4/9 |

HGB achieved the lowest mean MAE and mean RMSE and the highest mean R² among the three historical baseline models.

At the individual-well level, HGB achieved the best result on seven of nine wells for each of MAE, RMSE, and R². Linear Regression performed best on the other two wells for each metric.

## Interpretation

The corrected comparison supports HGB as the strongest of the three historical baseline models under the two-feature configuration (`GR` and `LOG10_RT`).

However, five of the nine held-out wells had negative HGB R². Cross-well generalization therefore remains uneven and must be assessed at the well level.

The historical two-feature comparison must be kept distinct from the selected final local-variability HGB configuration, which uses four features:

- `GR`
- `LOG10_RT`
- `GR_LOCAL_STD_11`
- `LOGRT_LOCAL_STD_11`

The historical comparison is a like-for-like comparison among three baseline algorithms. It is not a direct comparison against the final four-feature model because the feature configuration and modeling workflow differ.

## Output Artifact

- `reports/models/final_phif_model_comparison.csv`

## Conclusion

The historical model comparison was corrected and regenerated from a single consistent LOO evaluation source. Its summary metrics were verified against `phif_hgb_cv_summary.csv`, with all three models passing the numerical checks.

No Git staging or commit was performed during this phase.

---

## Change Log — Latest Update

### 2026-11-03

- Audited the complete historical model comparison workflow.
- Identified that the previous comparison combined Linear and XGBoost results from a three-feature evaluation with HGB results from a two-feature evaluation.
- Confirmed that `phif_hgb_cv_results.csv` contains all three models across the same nine LOO folds using `GR` and `LOG10_RT`.
- Updated `scripts/final_phif_model_comparison.py` to use the single consistent evaluation source.
- Added input and evaluation-integrity validation.
- Regenerated `reports/models/final_phif_model_comparison.csv`.
- Verified all three model summaries against `phif_hgb_cv_summary.csv`; mean MAE, mean RMSE, and mean R² matched within `1e-10`.
- Confirmed 27 model evaluations across nine wells, with no duplicate model–well combinations.
- Preserved the distinction between the historical two-feature baseline comparison and the final four-feature local-variability HGB model.
- No Git staging or commit was performed.

## Document Status

**Status:** Living document — research and repository finalization in progress  
**Last updated:** 2026-11-03  
**Next update:** After README reconciliation and the remaining repository/artifact audit.

---

# 29. Phase 21 — Python Script Path Stabilization

**Status:** Completed  
**Date:** 2026-11-06

## Objective

Stabilize project path resolution across the Python scripts so that scripts can locate project resources independently of the current working directory.

The objective was to improve repository portability and execution consistency without changing the existing data processing, analytical, or modeling logic.

## Activities Performed

1. Audited all Python scripts under the `scripts/` directory.
2. Reviewed project-root path definitions and relative path references.
3. Corrected parent-directory resolution in scripts where the existing path definition did not consistently resolve to the project root.
4. Anchored remaining working-directory-relative paths to the project root.
5. Performed an Abstract Syntax Tree (AST) parsing audit.
6. Conducted a working-directory independence check from an external temporary directory.

## Validation Results

| Validation item | Result |
|---|---:|
| Python scripts audited | 63 |
| Scripts passing AST parsing | 63 |
| Scripts with syntax errors | 0 |
| Scripts with corrected `.parent` resolution | 28 |
| Scripts already using `.parent.parent` appropriately | 9 |
| Additional scripts with root-anchored paths | 4 |
| Working-directory independence checks passed | 11 |
| Working-directory independence checks failed | 0 |

The working-directory independence checks inspected path resolution without executing the analytical or modeling pipelines.

## Scope and Preservation

The stabilization was limited to filesystem path resolution.

No intentional changes were made to:

- Dataset contents
- Feature engineering logic
- Model configurations
- Evaluation methodology
- Prediction values
- Existing analytical conclusions
- Experimental output artifacts

## Conclusion

The Python scripts passed the syntax audit, and the selected path-resolution checks confirmed that project resources could be resolved independently of the current working directory.

This phase improves the maintainability and portability of the repository without changing the established experimental results.

---

# 30. Phase 22 — Baseline Artifact Integrity Verification

**Status:** Completed  
**Date:** 2026-11-06

## Objective

Verify that the important experimental artifacts remained unchanged during repository stabilization.

The verification uses a previously generated SHA256 baseline manifest as the reference for artifact integrity.

## Activities Performed

1. Reviewed the existing baseline manifest:

   `reports/data_quality/baseline_manifest_pre_stabilization_2026-11-06.csv`

2. Identified the 30 selected experimental artifacts recorded in the manifest.
3. Recalculated SHA256 hashes for the corresponding existing files.
4. Compared the calculated hashes against the recorded baseline values.
5. Checked for missing artifacts and content changes.

## Validation Results

| Validation item | Result |
|---|---:|
| Baseline artifacts recorded | 30 |
| Artifacts matched | 30 |
| Missing artifacts | 0 |
| Changed artifacts | 0 |
| Integrity verification | PASS |

## Interpretation

All 30 selected artifacts matched their recorded SHA256 baseline values.

This confirms that the selected experimental artifacts remained byte-for-byte unchanged during the repository stabilization activities covered by this verification.

The manifest provides a reference point for future maintenance and integrity checks.

## Conclusion

The baseline artifact integrity verification passed.

The existing experimental outputs were preserved, and no changes to the selected baseline artifacts were detected.

---

# 31. Phase 23 — README and Dependency Audit

**Status:** Completed with follow-up documentation items  
**Date:** 2026-11-06 to 2026-11-09

## Objective

Review the main project documentation and Python dependency definitions to improve repository clarity, reproducibility, and readiness for publication.

## README Review

The README was reviewed as the primary entry point for users, reviewers, and potential recruiters.

The review covered:

1. Project overview and problem definition.
2. Dataset description and scope.
3. Methodology and evaluation strategy.
4. Feature engineering and final model configuration.
5. OOF prediction results.
6. PHIF characterization and explainability.
7. Repository architecture.
8. Key output artifacts.
9. Environment and reproducibility instructions.
10. Limitations and future work.

The README was expanded to document the project workflow, final modeling approach, evaluation results, and interpretation limitations.

## Artifact Reference Validation

The artifact references listed in the README were checked against the current repository files.

| Validation item | Result |
|---|---:|
| README artifact references checked | 25 |
| Existing artifacts | 25 |
| Missing artifacts | 0 |

The README also documents the baseline integrity manifest and its role in preserving experimental outputs.

## Dependency Audit

The project dependency definitions were reviewed against the active Python environment.

Environment:

- Python: 3.12.10
- NumPy: 2.5.3
- Pandas: 3.0.6
- Scikit-learn: 1.9.1
- SciPy: 1.18.1
- Matplotlib: 3.11.2
- lasio: 0.32
- XGBoost: 3.4.1

The dependency file uses pinned versions to improve environment reproducibility.

The installed environment was checked using:

`pip check`

Result:

`No broken requirements found.`

## Documentation Validation

The following checks were performed:

- Reviewed the README content and section structure.
- Confirmed that the documented artifact references exist.
- Reviewed the Git diff for whitespace errors.
- Confirmed that `git diff --check` completed without reported errors.

## Follow-up Documentation Items

The following items remain subject to final review:

1. The repository tree illustration in the README requires final alignment with the actual directory structure.
2. The case-sensitive filename representation of `README.md` in the filesystem and `Readme.md` in the Git index requires deliberate resolution before the first project commit.
3. The final README reproducibility instructions must remain consistent with the scripts' actual execution behavior and artifact overwrite risks.

These items are documentation and repository-finalization concerns. They do not require changes to the experimental outputs.

## Conclusion

The README artifact references and dependency consistency checks passed.

The main documentation is substantially complete, while the remaining repository presentation and Git filename consistency items are retained for final review.

No Git staging or commit was performed.

---

# 32. Phase 24 — Dataset Builder and Split Script Static Review

**Status:** Completed  
**Date:** 2026-11-09

## Objective

Review the source code of the dataset construction and well-based splitting scripts to assess their path resolution, processing logic, output behavior, and execution safety.

The review was conducted without rerunning the scripts because both scripts can overwrite existing experimental artifacts.

## Scripts Reviewed

1. `scripts/build_phif_ml_dataset.py`
2. `scripts/create_well_split.py`

## Dataset Builder Review

The script `build_phif_ml_dataset.py` was reviewed for:

- Project-root resolution.
- Well-directory and LAS input/output file discovery.
- Required curve validation.
- Numeric conversion.
- Depth alignment using rounded depth keys.
- One-to-one merge validation.
- ML readiness checks.
- Dataset sorting and output generation.

The script uses:

`Path(__file__).resolve().parents[1]`

to resolve the project root.

The reviewed output artifacts are:

- `data/processed/phif_ml_dataset_v1.csv`
- `reports/data_quality/phif_ml_dataset_audit_v1.csv`

The script contains a main execution guard.

## Well Split Script Review

The script `create_well_split.py` was reviewed for:

- Project-root resolution.
- Dataset input path.
- Fixed well assignments.
- Unknown and missing well validation.
- Split overlap validation.
- Row-count consistency checks.
- Split metadata generation.

The script uses the same project-root resolution approach.

Its expected outputs are:

- `data/processed/phif_train.csv`
- `data/processed/phif_validation.csv`
- `data/processed/phif_test.csv`
- `data/processed/phif_well_split_metadata.csv`

## Execution Safety Findings

Both scripts write output files directly using CSV export operations.

Therefore, rerunning them may overwrite existing artifacts.

An additional difference was identified:

- `build_phif_ml_dataset.py` uses a main execution guard.
- `create_well_split.py` executes its main processing logic at module level when loaded.

For this reason, importing the split script as a module can trigger data processing and output generation.

## Validation Scope

This phase was a static source-code review.

The scripts were not executed during this review.

No existing dataset, split file, prediction artifact, or model output was intentionally modified.

The earlier dataset and split reproducibility verification results remain documented in Phases 17 and 18.

## Conclusion

Both scripts use project-root-based path resolution.

Their processing logic and output destinations were reviewed, and the potential overwrite behavior was identified.

No additional code refactoring was considered necessary during this stabilization phase.

---

# 33. Phase 25 — Existing Well-Based Split Consistency Verification

**Status:** Completed  
**Date:** 2026-11-09

## Objective

Verify that the existing train, validation, and test CSV files are consistent with the documented fixed well-based split configuration.

This verification was performed using read-only inspection of the existing CSV files and split metadata.

## Activities Performed

1. Loaded the existing train CSV.
2. Loaded the existing validation CSV.
3. Loaded the existing test CSV.
4. Identified the unique well names in each partition.
5. Counted the number of rows and wells in each partition.
6. Loaded the split metadata CSV.
7. Compared the actual well assignments against the documented split configuration.

No dataset generation or split-generation script was executed.

## Validation Results

| Partition | Rows | Wells |
|---|---:|---:|
| Train | 24,516 | 6 |
| Validation | 11,936 | 1 |
| Test | 11,294 | 2 |
| **Total** | **47,746** | **9** |

## Well Assignment Verification

**Train:**

- `15_9-F-1A`
- `15_9-F-1B`
- `15_9-F11 T2`
- `15_9-F11A`
- `15_9-F11B`
- `15_9-F12`

**Validation:**

- `15_9-F15D`

**Test:**

- `15_9-F-1`
- `15_9-F1C`

## Split Metadata Verification

The metadata file contained nine records, corresponding to the nine selected wells.

The recorded split assignments matched the actual well membership found in the three existing partition files.

## Validation Findings

| Validation item | Result |
|---|---|
| Train row count | Consistent |
| Validation row count | Consistent |
| Test row count | Consistent |
| Total row count | 47,746 |
| Train well membership | Consistent |
| Validation well membership | Consistent |
| Test well membership | Consistent |
| Metadata records | 9 |
| Overall consistency | PASS |

## Methodological Qualification

This phase verifies the consistency of the existing partition files and their metadata.

It does not represent a new execution of the split-generation process.

The earlier byte-for-byte reproducibility verification, including SHA256 comparison of the four split artifacts, is documented separately in Phase 18.

The fixed well-based split remains designed to evaluate generalization to wells excluded from model training.

## Conclusion

The existing train, validation, and test datasets are consistent with the documented well-based split configuration.

The total of 47,746 rows across nine wells is preserved.

No existing split files or model artifacts were modified during this verification.

No Git staging or commit was performed.

---

---

## Change Log — Repository Stabilization Updates

### 2026-11-06

- Completed the AST audit of 63 Python scripts.
- Stabilized project-root path resolution across the relevant scripts.
- Completed 11 working-directory independence checks with no failures.
- Verified 30 selected experimental artifacts against the pre-stabilization SHA256 baseline.
- Confirmed that all 30 baseline artifacts matched, with no missing or changed files.
- Reviewed the README structure and documented project methodology, model results, and limitations.
- Verified 25 README artifact references against existing repository files.
- Reviewed pinned Python dependencies.
- Confirmed that `pip check` reported no broken requirements.

### 2026-11-09

- Reviewed `build_phif_ml_dataset.py` and `create_well_split.py` through static source-code inspection.
- Identified direct output-overwrite behavior in both scripts.
- Documented the difference in execution behavior between the dataset builder and split-generation script.
- Verified existing train, validation, and test CSV row counts.
- Confirmed the nine well assignments against the split metadata.
- Confirmed a total of 47,746 rows across the three partitions.
- Confirmed that the existing split files were not regenerated during this read-only verification.
- Continued repository and documentation stabilization.
- No Git staging or commit was performed.

### 2026-11-12

- Reviewed the README repository structure and confirmed that the documented project layout reflects the current PHIF workflow.
- Added overwrite warnings for `build_phif_ml_dataset.py` and `create_well_split.py` based on their previously verified direct-write behavior.
- Compared the updated README against `README.md.pre_tree_fix.bak`; confirmed that the only differences were the two overwrite-warning edits.
- Verified README UTF-8 decoding and preserved CRLF line endings.
- Confirmed that `git diff --check -- Readme.md` reported no whitespace errors.
- Identified the remaining filename-case discrepancy between `Readme.md` in the Git index and `README.md` in the working directory.
- No dataset regeneration, split regeneration, model execution, Git staging, or commit was performed.

## Document Status

**Status:** Living document — repository stabilization and final documentation review in progress

**Last updated:** 2026-11-12

**Completed:** Main PHIF modeling workflow, final OOF characterization, dataset and split reproducibility verification, final model artifact validation, historical model comparison correction, and initial repository stabilization.

**Remaining work:**

1. Complete README repository-tree and filename consistency review.
2. Complete final project documentation reconciliation.
3. Perform a final read-only repository and artifact audit.
4. Review `.gitignore` and Git tracking behavior.
5. Review the complete Git diff before staging.
6. Obtain explicit approval before the first new project commit.
7. Prepare the Medium publication narrative from the verified project history.

**Git policy:** No `git add` or `git commit` until the main project and final repository review are complete and explicitly approved.
