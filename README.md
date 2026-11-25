# Explaining Instant EMI-Finance Decisions

External Practical Examination (Set B) for **Responsible and Safe AI (AIML4001)**
Charotar University of Science and Technology, CSPIT - Seventh Semester B. Tech (AIML)

**Name:** Siddhraj Anilkumar Thakor
**Enrollment No:** 23AIML070

## About

A finance company (NBFC) gives instant EMI loans at electronics showrooms and a machine learning model decides who gets approved. Customers who are rejected want to know why. This project builds an explainability pipeline on the German Credit dataset that shows which features drive the model overall and gives a clear reason for one rejected applicant.

## Files

| File | Description |
|---|---|
| `RSA_SetB_23AIML070.ipynb` | Colab notebook with the full code, text cells, outputs and plots |
| `23AIML070_SET_B_RSA.docx` | Practical report with aim, algorithm, program, output, conclusion and answers to the four questions |

## What the notebook does

1. Loads the German Credit data (1000 applicants, 20 attributes), encodes the categorical columns with OrdinalEncoder and splits it 80:20 (stratified, `random_state = 42`).
2. Trains a Random Forest (200 trees, max depth 8) and reports test accuracy and ROC-AUC.
3. Uses SHAP (TreeExplainer) to get global feature importance and draws the summary plot.
4. Picks one rejected applicant from the test set and draws the SHAP waterfall plot with the top 3 reasons.
5. Explains the same applicant with LIME and compares it with SHAP.
6. Writes a plain-language rejection message and checks whether sensitive attributes (age, personal_status, foreign_worker) appear among the reasons.

## How to run

1. Open `23AIML070_External_Practical_Exam.ipynb` in [Google Colab](https://colab.research.google.com/drive/1XtP7k4iVxc3xE-17W5VH9I1aFk_-5wjI?usp=sharing).
2. Click **Runtime > Run all**.

No dataset needs to be uploaded. The notebook downloads the German Credit file from GitHub, converts the coded values to readable words and saves it as `german_credit.csv`. It also installs `shap` and `lime` in the first cell. An internet connection is needed.

Libraries used: Pandas, NumPy, Scikit-learn, SHAP, LIME, Matplotlib.

## Results

| Item | Value |
|---|---|
| Test accuracy | 0.77 |
| ROC-AUC | 0.83 |
| Top 5 features | checking_status, duration, credit_amount, savings_status, purpose |
| Selected applicant | test index 570, P(bad) = 0.57 (rejected) |
| SHAP and LIME top-3 agreement | 2 / 3 |
| Sensitive attributes in reasons | age, personal_status |

Rejection message generated:

> Your EMI request was not approved mainly because your current account balance is negative and the requested duration (24 months) is long.

Since personal_status and age came up among the reasons, such cases should be reviewed for fairness before the model is used for real decisions.

## Plots saved by the notebook

Running the notebook creates these images in the Colab working folder:

- `shap_summary.png`
- `shap_waterfall.png`
- `lime_explanation.png`

## Note

Values may change slightly with different versions of scikit-learn, SHAP or LIME. If that happens, use your own observed values in the report.
