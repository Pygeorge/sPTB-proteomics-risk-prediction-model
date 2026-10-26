# sPTB-proteomics-risk-prediction-model
Code and supporting files for the development and evaluation of a cervicovaginal-fluid (CVF) proteomics-based model for spontaneous preterm birth (sPTB) risk prediction.

The repository contains the analysis scripts, the processed proteomic matrix used by the model, and the fitted logistic-regression model for reproducibility and further methodological evaluation.

## Repository contents

| File or directory                                   | Description                                                  |
| --------------------------------------------------- | ------------------------------------------------------------ |
| `data/proteome.txt`                                 | Processed sample-level clinical and proteomic data. The first columns contain sample metadata; subsequent columns contain protein measurements. |
| `model/LR_final.pkl`                                | Serialized final logistic-regression model.                  |
| `#model_construction_and_evaluation.py`             | Main Python workflow for model construction, prediction, ROC/precision–recall analysis, subgroup analysis, calibration-related evaluation, and fixed-threshold performance. |
| `#model_construction_and_evaluation_GWS_and_GWD.py` | Evaluation for gestational-week sampling and delivery definitions. |
| `#CL_model_construction_and_evaluation.py`          | Workflow for evaluating the model with cervical-length measurements. |
| `jackknifing.py`                                    | Jackknife and differential-expression utilities.             |
| `EN.R`                                              | Elastic-net model selection and leave-one-out evaluation in R. |
| `elbow_point.py`                                    | Elbow-point analysis for selecting the number of proteins. |
| `UMAP.py`                                           | UMAP visualization of MCH1 proteomic profiles.                    |
| `composite_score_function.py`                       | Composite ranking function for candidate protein prioritization. |
| `LICENSE`                                           | Repository license.                                          |

Using the fitted model

The fitted model can be loaded with `joblib`:

```python
import joblib

model = joblib.load("model/LR_final.pkl")
```

The fitted model can be represented as the following logistic-regression equation:

```text
p(sPTB) = 1 / {1 + exp[-(-0.95505271
         + 0.05338880 × AMBP
         + 0.04637507 × FN1
         + 0.03702857 × LUM
         + 0.05277170 × TIMP1
         + 0.03004814 × B2M)]}
```

> **Research use only:** This model was developed for research and methodological evaluation. It has not been prospectively validated for clinical use and should not be used for diagnosis, treatment decisions, or individual patient risk assessment.

## Data availability

Primary proteomics data are available through the iProX repository under accession `IPX0008492002` 



## Contact

For questions about the code or model, please contact the corresponding author listed in the associated manuscript.
