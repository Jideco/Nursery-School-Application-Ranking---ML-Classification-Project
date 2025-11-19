# Nursery School Application Ranking - ML Classification Project

## Problem Description
**Context:** In the 1980s, Ljubljana, Slovenia faced excessive enrollment for nursery schools, requiring an objective system to rank applications and provide transparent explanations for rejections. Traditional manual evaluation was time-consuming and lacked consistency.

**Problem:** Can we build a machine learning model to automatically rank nursery school applications based on family characteristics, providing fair and explainable decisions?

**Solution:** This project develops a multi-class classification model that predicts application priority levels based on 8 categorical family attributes, offering:
- **Objective Decision Making:** Consistent evaluation criteria across all applications
- **Transparency:** Clear explanations for acceptance/rejection decisions  
- **Efficiency:** Automated processing of large application volumes
- **Fairness:** Bias detection and mitigation in the decision process

**Business/Social Value:**
- **🏫 Educational Institutions:** Streamline admission processes and optimize resource allocation
- **👨👩👧👦 Families:** Receive transparent, objective feedback on application status
- **📊 Policy Makers:** Identify systemic patterns and improve social support programs
- **⚖️ Social Equity:** Ensure fair access to early childhood education resources

## Dataset
**Source:** UCI Machine Learning Repository - Nursery Database

**Description:**
- **Samples:** 12,960 nursery school applications
- **Features:** 8 categorical attributes representing family characteristics
- **Target:** Application class (5 categories: not_recom, recommend, very_recom, priority, spec_prior)
- **Origin:** Derived from a hierarchical decision model used in Ljubljana, Slovenia during 1980s nursery school enrollment crises

**Features:**
- `parents` - Parents' occupation status
  - *Categories:* usual, pretentious, great_pret
- `has_nurs` - Child's nursery status
  - *Categories:* proper, less_proper, improper, critical, very_crit
- `form` - Family structure form
  - *Categories:* complete, completed, incomplete, foster
- `children` - Number of children in family
  - *Categories:* 1, 2, 3, more
- `housing` - Housing conditions
  - *Categories:* convenient, less_conv, critical
- `finance` - Family financial standing
  - *Categories:* convenient, inconv
- `social` - Social conditions
  - *Categories:* non-prob, slightly_prob, problematic
- `health` - Health conditions
  - *Categories:* recommended, priority, not_recom

**Target Variable:**
- `class` - Final application ranking
  - *Categories:* not_recom, recommend, very_recom, priority, spec_prior

**Dataset included in repository:** `nursery.csv`

## Project Structure
```
nursery-application-ranking/
│
├── data/
│   └── nursery.csv                # dataset
├── notebook.ipynb                 # EDA, feature work, experiments
├── train.py                       # training script
├── predict.py                      # FastAPI app (prediction endpoint)
├── decision_department.py         # example client / tests
├── model.bin                      # trained XGBoost model
├── pyproject.toml                 # dependencies
├── Dockerfile                     # container image
├── README.md                      # this file
└── fly.toml                       # fly.io deploy config
```

---

## Key findings from EDA
**Data Overview**
- No missing values ✓ 

**Most Important Features**
*(Based on feature importance and mutual information analysis)*
- `health` - Health conditions may be a key factor (0.6638)
- `has_nurs` - Child's current nursery situation likely critical (0.1363)
- `social` - Social conditions affect family stability (0.0149)
- `finance` - Financial standing impacts priority (0.0026)

## Model Performance
**Algorithm:** XGBoost classifier (best performer)

**Performance Metrics:**
- **Accuracy:** [0.9984567901234568]
- **Confusion Matrix:** [<img width="683" height="547" alt="image" src="https://github.com/user-attachments/assets/aeb50f2f-70d4-42e4-94e0-581f18eec792" />
]

**Model Comparison:**
- Random Forest: [0.9807]
  <img width="683" height="547" alt="image" src="https://github.com/user-attachments/assets/650d0b7b-d26f-4381-af4e-4f37c7ad788f" />

- Gradient Boosting: [0.9984]
  <img width="683" height="547" alt="image" src="https://github.com/user-attachments/assets/e412a408-33ef-4801-b076-9aa7f76af29c" />


---

## Quickstart, local

### requirements

* Python 3.12 recommended

### run locally

1. create env, install deps

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

2. start API

```bash
uvicorn predict:app --reload --port 9696
```

3. open docs: `http://localhost:9696/docs`

---

## Docker

1. build

```bash
docker build -t predict-decision .
```

2. run

```bash
docker run -it -p 9696:9696 predict-decision
```

3. verify

```bash
curl http://localhost:9696/health
```

---

## Deployed

Example (fly.io): `https://silent-thunder-9151.fly.dev/docs/`

---

## API

### Predict

`POST /predict`
Content-Type: `application/json`
Request example

```json
{
  "parents": "usual",
  "has_nurs": "proper",
  "form": "complete",
  "children": "2",
  "housing": "convenient",
  "finance": "convenient",
  "social": "non-prob",
  "health": "recommended"
}
```
<img width="911" height="721" alt="image" src="https://github.com/user-attachments/assets/e6b03f4b-f03b-40ec-ba4f-8e04442e77cd" />
---

## Reproducibility

* use a fixed seed for splits and training, for example `random_state=42`
* always `stratify=y` on train/test split when using the `class` label
* training command (example)

```bash
python train.py --data data/nursery.csv --seed 42
```

Training will:

* encode categories consistently
* train the XGBoost classifier with tuned params
* save `model.bin` and encoder metadata

---

## Limitations & next steps

**Current limits**

* dataset mirrors 1980s local rules, may not generalize across regions or eras.
* heavy reliance on `health` could bake policy choices into the model.

---

## Troubleshooting

* `ModuleNotFoundError`: activate env, install deps.
* Port conflicts: free the port or use a different host port with Docker.
* Missing model in container: ensure `model.bin` is copied in `Dockerfile`, rebuild image.


---

## Tech stack

* Python 3.12
* XGBoost, scikit-learn
* pandas, numpy
* FastAPI, uvicorn
* matplotlib, seaborn
* Docker, Fly.io

---

## Author

Mujeeb Olajide Opabode
GitHub: `@jideco`
Email: `jideopabode@gmail.com`

---

## License

Educational and research use. Use responsibly.

---
