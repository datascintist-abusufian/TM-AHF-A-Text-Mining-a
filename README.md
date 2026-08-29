# TM-AHF revised analysis

This package is an end-to-end Python template for rerunning the TM-AHF manuscript analysis with stronger validation.

## 1. Put your CSV in a local data folder

Example:

```text
project/
├── tm_ahf_revision_analysis.py
├── requirements.txt
└── data/
    └── tm_ahf.csv
```

## 2. Edit the `Config` block

At minimum map the real column names for:

- patient ID
- hospital/centre ID
- AHF target or BNP/NT-proBNP fields
- age
- pulse
- systolic BP
- diastolic BP
- weight
- height
- chief complaint
- present medical history
- past medical history

Do not run the script until these match the real dataset.

## 3. Target definition

If the dataset already contains the final BNP/NT-proBNP-defined binary label:

```python
derive_target_from_biomarkers = False
target_col = "your_existing_label"
```

If you want to regenerate the target from biomarkers:

```python
derive_target_from_biomarkers = True
```

The supplied rule is:

- BNP >= 300 ng/L
- NT-proBNP >= 450 ng/L if age < 50
- NT-proBNP >= 900 ng/L if age 50-75
- NT-proBNP >= 1800 ng/L if age > 75

Confirm these exact thresholds and age boundaries against the original protocol before using them.

## 4. Install and run

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
python tm_ahf_revision_analysis.py
```

## 5. Main outputs

The script creates:

- baseline characteristics table
- exact train/test patient manifest
- 10-model combined-data benchmark
- LightGBM combined/text/structured comparison
- 95% bootstrap confidence intervals
- DeLong AUROC comparisons
- ROC curves
- precision-recall curves
- Brier scores
- calibration intercept and slope
- calibration curves
- decision-curve analysis
- SHAP beeswarm/global feature table
- two SHAP local waterfall explanations
- bidirectional hospital A -> B and B -> A external validation
- saved fitted LightGBM pipelines

## Important methodological difference from the submitted manuscript

The revised code deliberately performs TF-IDF fitting, KNN imputation, feature selection and hyperparameter tuning only within the training data/CV folds. The held-out test set and external hospital are never used to fit preprocessing.

The code uses chi-square selection of 38 text features after a training-fitted 5,000-feature TF-IDF representation. This is more reproducible than the manuscript's vague statement that a "gradient descent algorithm" selected the final 44 features. If you still have the exact original feature-selection procedure, compare it before replacing the Methods description.

## Chinese word segmentation

The script assumes the text stored in the EHR is already tokenizable by TF-IDF. If your original workflow explicitly used Jieba segmentation, add a tokenizer function to `TfidfVectorizer`. Do this only after checking how the source text was cleaned in the original analysis.

## External validation

The code expects exactly two values in the hospital ID column. It trains/tunes using one centre only and evaluates on the untouched other centre, then reverses the direction. This is substantially stronger than a random 80/20 split alone.
