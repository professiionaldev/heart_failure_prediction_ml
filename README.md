# 🫀 Heart Failure Prediction using Supervised Machine Learning

This project explores multiple supervised machine learning models to predict heart failure events using clinical data. It includes data preprocessing, feature scaling, model training, accuracy comparison, and visualization.

---

## 📂 Dataset

**Source**: `heart_failure_clinical_records_dataset.csv`  
**Samples**: 299 patients  
**Features**: 13 clinical variables  
**Target**: `DEATH_EVENT` (0 = survived, 1 = death occurred)

---

## 🔍 Objectives

- Predict the risk of heart failure using various classifiers
- Evaluate and compare performance of each model
- Visualize model accuracy with a clean bar chart
- Understand how scaling affects model performance

---

## ⚙️ Workflow Overview

1. **Data Loading & Exploration**
   - Verified data shape, types, and checked for missing values
   - Split features into continuous & categorical variables

2. **Feature Selection**
   - Used only continuous features:
     - Age, Creatinine Phosphokinase, Ejection Fraction, Platelets, Serum Creatinine, Serum Sodium, Time

3. **Target Variable**
   - `DEATH_EVENT` (binary classification)

4. **Train-Test Split**
   - 70% Training | 30% Testing

5. **Data Scaling**
   - Applied `StandardScaler` to normalize feature ranges

6. **Model Training**
   - ✅ Logistic Regression  
   - ✅ Support Vector Machine (SVC)  
   - ✅ K-Nearest Neighbors (with K optimization)  
   - ✅ Decision Tree (entropy, depth=2)  
   - ✅ Naive Bayes  
   - ✅ Random Forest

7. **Evaluation**
   - Used accuracy score (rounded to 4 decimal places & converted to %)  
   - Plotted results using `seaborn` with clean labels and color palettes

---

## 📊 Accuracy Results

| Model               | Accuracy (%) |
|--------------------|--------------|
| Logistic Regression| 87.78        |
| SVC                | 84.44        |
| KNN (k=6)          | 84.44        |
| Decision Tree      | **88.89**    |
| Naive Bayes        | 82.22        |
| Random Forest      | 86.67        |

📈 Visualized using a Seaborn bar chart with model-wise comparison and annotations.

---

## 🧠 Key Insights

- Decision Tree (with simple tuning) gave the best accuracy.
- Scaling data significantly helps models like SVM & KNN.
- Visualization makes it easy to compare multiple classifiers.
- Simpler models like Logistic Regression performed surprisingly well.

---

## 📌 Technologies Used

- Python 3.x  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook

---

## 🚀 Future Work

- Add confusion matrix and ROC-AUC scores  
- Use cross-validation for robust model comparison  
- Try ensemble methods like XGBoost, LightGBM  
- Build a front-end to input patient data and return predictions (Web App)

---

## 💬 Author

Created by Kuldeep Singh — aspiring data scientist & ML engineer.

---

> ⭐ *Feel free to fork, improve, or use this notebook as a base for your own ML projects!*

