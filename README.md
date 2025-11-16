# Nursery-School-Application-Ranking---ML-Classification-Project


# 👶 Nursery School Application Ranking - ML Classification Project

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
- **Target:** Application class (3 categories: recommended, priority, not_recom)
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
  - *Categories:* recommended, priority, not_recom

**Dataset included in repository:** `nursery.csv`

## Project Structure
nursery-application-ranking/
│
├── data/
│   └── nursery.csv                   # Dataset
│
├── notebook.ipynb                    # Complete EDA and modeling
├── train.py                          # Training script
├── predict.py                        # Flask API service
├── test_api.py                       # API testing script
│
├── nursery_classifier.pkl            # Trained classification model
├── feature_encoder.pkl               # Feature encoder for categorical variables
│
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker configuration
├── README.md                         # This file
│
└── app.py                           # (BONUS) Streamlit web app

## Key Findings from EDA
*(You'll fill this in after your analysis, but here are expected sections)*

**Data Overview**
- No missing values ✓
- Class distribution: [You'll analyze this]
- Feature correlations and patterns

**Most Important Features**
*(Based on domain knowledge and your analysis)*
- `has_nurs` - Child's current nursery situation likely critical
- `finance` - Financial standing impacts priority
- `health` - Health conditions may be a key factor
- `social` - Social conditions affect family stability

**Class Distribution**
- Significant class imbalance expected (most applications likely "not_recom")
- Requires careful handling with techniques like class weighting or resampling

## Model Performance
*(You'll fill this with your results)*

**Best Model:** [Your chosen model after comparison]

**Performance Metrics:**
- **Accuracy:** [Value]%
- **Precision (weighted):** [Value]
- **Recall (weighted):** [Value] 
- **F1-Score (weighted):** [Value]
- **Confusion Matrix:** [Visual description]

**Model Comparison:**
- Decision Tree: [Results]
- Random Forest: [Results] 
- Gradient Boosting: [Results]
- [Other models you test]

## Installation & Setup
**Prerequisites**
- Python 3.11+
- Docker (for containerized deployment)

**Option 1: Local Installation**
```bash
# Clone the repository
git clone https://github.com/your-username/nursery-application-ranking.git
cd nursery-application-ranking

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (optional - model already included)
python train.py

# Start the Flask API
python predict.py
