# 🧠 Breast Cancer Priority Prediction Model

This project builds a machine learning model to **predict issue priority** (High, Medium, Low) for breast cancer cases using a structured dataset from Kaggle. The aim is to support **resource allocation and triage decisions** in healthcare using predictive analytics.

## 📁 Dataset

- **Source:** [Kaggle Competition Dataset: IUSS 23-24 Automatic Diagnosis Breast Cancer](https://www.kaggle.com/competitions/iuss-23-24-automatic-diagnosis-breast-cancer)
- **Features:** Tumor characteristics (e.g. radius, texture, concavity)
- **Target:** Custom-generated `priority` label based on diagnosis and tumor severity

## 🧪 Model Workflow

1. **Data Preprocessing**
   - Loaded the dataset using Kaggle API
   - Dropped ID and non-useful columns
   - Generated a `priority` label:
     - `High`: Malignant & radius > mean
     - `Medium`: Malignant & radius ≤ mean
     - `Low`: Benign

2. **Training**
   - Split data into training and test sets (80/20)
   - Trained a **Random Forest Classifier**
   - Evaluated with **Accuracy**, **F1-score**, and **Confusion Matrix**

3. **Results**
   - Achieved high classification performance
   - `radius_mean`, `concavity_mean`, and `area_mean` were top predictors

4. **Model Export**
   - Saved trained model using `joblib` (`random_forest_priority_model.pkl`)

## 📊 Visualization

- Confusion Matrix for evaluating predictions
- Feature Importance bar chart to explain model reasoning

## 📦 Dependencies

Install required packages:

```bash
pip install pandas scikit-learn matplotlib seaborn joblib
```
#NOTE THAT THE SCREENSHOTS FOR THE MODEL ARE WITHIN THE PDF FILE



##HOW TO USE
import joblib
import pandas as pd

# Load saved model
model = joblib.load('random_forest_priority_model.pkl')

# Predict new data
new_data = pd.read_csv("new_input.csv")
predictions = model.predict(new_data)
