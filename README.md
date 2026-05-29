# 🔥 AI-Era Employee Burnout Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)
![CodeCarbon](https://img.shields.io/badge/CodeCarbon-CO₂%20Tracked-2ECC71?style=for-the-badge&logo=leaflet&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**End-to-End Machine Learning Pipeline for Predicting Employee Burnout in the Age of AI**

*8,500 samples · 5 classifiers · GridSearchCV · SMOTETomek · Carbon Tracked*

</div>

---

## 🎯 Problem Statement

> As AI tools rapidly integrate into the workplace, employees face increasing cognitive load, skill anxiety, and work-life imbalance. This project builds a **multi-class classification system** to identify burnout risk levels — enabling HR departments to intervene **before** burnout occurs.


---

## 🗂️ Dataset

| Feature | Detail |
|---|---|
| 📊 **Samples** | 8,500 employee records |
| 🔢 **Features** | 22 variables (18 numeric, 2 categorical, 1 target) |
| 🏭 **Industries** | Finance, Technology, Healthcare, Education, Creative, Legal |
| 👔 **Job Roles** | 30 unique positions |
| ❌ **Missing Values** | None |
| 🎯 **Target** | `Burnout_Label` → No Risk / At Risk / Burned Out |

---

## 🔄 ML Pipeline

---

## 🤖 Models

| # | Algorithm | Type | class_weight |
|---|---|---|---|
| 1 | **Logistic Regression** | Linear | `balanced` |
| 2 | **K-Nearest Neighbors** | Instance-based | `distance` weights |
| 3 | **Decision Tree** | Tree-based | `balanced` |
| 4 | **Random Forest** | Ensemble | `balanced` |
| 5 | **XGBoost** | Gradient Boosting | `sample_weight` |

---

## ⚙️ Feature Engineering

Two domain-driven features were derived:

```python
# 1️⃣ Burnout Risk Index
#    High stress × long hours / poor sleep → composite burnout signal
Burnout_Risk_Index = (Stress_Level × Work_Hours_Per_Week) / (Sleep_Hours_Per_Night + 1)

# 2️⃣ Cognitive Overload Score
#    Mental pressure per unit of focused work time
Cognitive_Overload_Score = (Cognitive_Load_Score + Interruptions_Per_Day + Meetings_Per_Week) \
                           / (Deep_Work_Hours + 1)
```

Both features showed **high ANOVA F-scores**, confirming strong class discriminability.

---

## ⚖️ Class Imbalance Strategy

---

## 📊 Evaluation Metrics

> ⚠️ **Accuracy is misleading here** — a model predicting "No Risk" for everyone achieves 77.7% accuracy. We use **Macro F1-Score** as the primary metric.

| Metric | Why Used |
|---|---|
| **Macro F1-Score** | Primary — treats all classes equally regardless of size |
| **Macro ROC-AUC** | Ranking quality across all classes |
| **Precision / Recall** | Class-level performance breakdown |
| **Confusion Matrix** | Visual error analysis per class |

---

## 🌱 Carbon Footprint Tracking

Each model's environmental cost was measured using **CodeCarbon**:

```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(log_level="error", save_to_file=False)
tracker.start()
model.fit(X_train, y_train)
emissions = tracker.stop()  # kg CO₂ equivalent
```

| Tracked Metric | Description |
|---|---|
| ⏱️ Training Time | Wall-clock seconds |
| 🧠 RAM Usage | Peak memory via `psutil` |
| 💾 Model Size | Serialized `.pkl` size (MB) |
| 🌿 CO₂ Emissions | Grams of CO₂ equivalent |

---

## 🗃️ Project Structure
---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-burnout-prediction.git
cd ai-burnout-prediction

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook burnout_classification.ipynb
```

---

## 📦 Requirements
---

## 🏗️ Key Technical Decisions

<details>
<summary><b>Why Target Encoding over One-Hot?</b></summary>

`Job_Role` has 30 unique categories. One-Hot Encoding would create 30 new columns (dimensionality explosion). Target Encoding maps each category to its statistical relationship with the target — no extra columns, meaningful signal.

</details>

<details>
<summary><b>Why Macro F1 over Accuracy?</b></summary>

With 194× class imbalance, a naive model predicting "No Risk" always achieves 77.7% accuracy. Macro F1 weights all classes equally — a "Burned Out" miss costs the same as a "No Risk" miss.

</details>

<details>
<summary><b>Why SMOTETomek over plain SMOTE?</b></summary>

With only 34 "Burned Out" examples, plain SMOTE interpolates between very similar points, creating noisy synthetic samples near the decision boundary. Tomek Links removes these boundary noise points, yielding cleaner class separation.

</details>

---

## 📌 Results Summary

| Model | CV F1 (mean ± std) | Test F1 | Overfitting Gap |
|---|---|---|---|
| Logistic Regression | — | — | — |
| KNN | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

> 💡 Run the notebook to populate this table with your actual results.

---

<div align="center">

**Made with ❤️ for Machine Learning Course**

*If this project helped you, consider giving it a ⭐*

</div>
