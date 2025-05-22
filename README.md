# Phantom Inventory Classifier

Detect phantom inventory in retail store systems using machine learning models (LightGBM, XGBoost, Random Forest, LSTM) trained on real-world stock, sales, and forecasting data.

---

## 🔍 What is Phantom Inventory?

**Phantom inventory** refers to stock that appears in the system but is not actually available in the store, and it's often due to scan errors, shrinkage, or replenishment issues. It leads to:

- Missed sales opportunities
- Wasted replenishment efforts
- False stockout signals

This repository provides a machine learning solution to detect phantom inventory with high precision.

> ⚠️ **Note:** The dataset used in this project is protected under NDA and cannot be shared or distributed under any circumstances.

---

## ⚙️ Technology Stack

- Python 3.11.8
- Scikit-learn (Pipelines, Imputers, OneHotEncoder)
- LightGBM / XGBoost / RandomForest / LSTM
- Pandas / NumPy
- Matplotlib / Seaborn (for visualizations)
- Jupyter Notebooks for training
- `joblib` for model serialization

---

## 🚀 Project Structure

```text
phantom-inventory-classifier/
│
├─ model/                     # Final trained models
│   ├─ lgbm/                  # LightGBM version
│   ├─ lstm/                  # LSTM version
│   ├─ rf/                    # Random Forest version
│   └─ xgb/                   # XGBoost version
│
├─ notebooks/                 # Training and analysis notebooks
│   ├─ 01_lgbm_training.ipynb
│   ├─ 02_xgb_training.ipynb
│   ├─ 03_rf_training.ipynb
│   └─ 04_lstm_training.ipynb
│
├─ requirements.txt           # All dependencies
├─ environment.yml            # Same environment setup as mine
├─ .gitignore
└─ README.md                  # This file
```

---

## 🌟 Modeling Process

I experimented with four modeling approaches:

- **LightGBM (LGBM)**: Best performance. Achieved 0.91 precision and 0.76 recall at threshold 0.81.
- **XGBoost**: Also high precision but slightly lower recall.
- **Random Forest**: Solid baseline model.
- **LSTM**: Deep learning sequence model for capturing temporal patterns.

Each model was trained on the same feature set, which includes engineered indicators like:

- `DailyBOH`, `sales_gap`, `forecast_gap`, `inventory_discrepancy`
- Rolling stock ratios
- Day-of-week indicators

I optimized for **Precision ≥ 0.90**, as client stakeholders required high trust in phantom alerts.

---

## 📊 Results Snapshot

| Model    | Precision | Recall | F1   | AUC   |
|----------|-----------|--------|------|-------|
| ⚡️ **LGBM**     | **0.91**  | **0.77**   | **0.83** | **0.986** |
| 📈 XGBoost  | 0.91      | 0.59   | 0.72 | 0.934 |
| 🌳 RF       | 0.91      | 0.53   | 0.67 | 0.876 |
| 🧠 LSTM     | 0.91      | 0.48   | 0.63 | 0.879 |

---

## 🧪 How to Use

1. Clone the repo:
    ```bash
    git clone https://github.com/hungchenhsu/phantom-inventory-classifier.git
    cd phantom-inventory-classifier
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run inference (within notebook):

    You can run predictions directly using the final `.pkl` pipeline and the saved threshold:

    ```python
    import joblib, json
    import pandas as pd

    # Load model and threshold
    pipeline = joblib.load("model/lgbm/pipeline.pkl")
    tau = json.load(open("model/lgbm/threshold.json"))["precision_floor_threshold"]

    # Load new data
    df = pd.read_csv("data/new_sku_data.csv")

    # Predict phantom flags
    proba = pipeline.predict_proba(df)[:, 1]
    df["phantom_alert"] = (proba >= tau).astype(int)
    df.to_csv("phantom_predictions.csv", index=False)
    ```


Each model folder contains:

- `pipeline.pkl`: Full preprocessing + model pipeline
- `threshold.json`: Tuned threshold for binary decision (precision ≥ 0.90)
- `model_card.pdf`: Summary of training setup and results

---

## 📚 Table of Contents

- [What is Phantom Inventory?](#-what-is-phantom-inventory)
- [Technology Stack](#️-technology-stack)
- [Project Structure](#-project-structure)
- [Modeling Process](#-modeling-process)
- [Results Snapshot](#-results-snapshot)
- [How to Use](#-how-to-use)
- [Table of Contents](#-table-of-contents)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Citation

If you find this repository helpful in your research, teaching, or professional work,
please consider citing or linking back to the repository:

Hung-Chen Hsu. Phantom Inventory Classifier: Multi-Model Detection of Retail Stock Discrepancies. GitHub, 2025.
Repository: https://github.com/hungchenhsu/phantom-inventory-classifier

This helps acknowledge the original work and supports open sharing in the machine learning and retail analytics community 🙌

---

Created with 💻 and 🎯 by Hung-Chen Hsu
