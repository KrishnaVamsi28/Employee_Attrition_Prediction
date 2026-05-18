# Employee Attrition Prediction (HR Analytics)

## 📌 Project Overview

This project focuses on predicting employee attrition using Machine Learning techniques. The objective is to identify employees who are likely to leave the organization based on HR-related factors such as salary, overtime, job satisfaction, work-life balance, and experience.

The project uses Logistic Regression and Random Forest algorithms along with Genetic Algorithm–based feature selection to improve prediction performance and identify important attrition drivers.

---

# 📊 Dataset Information

* Dataset Size: 1470 rows × 35 columns
* Domain: HR Analytics
* Target Variable: Attrition (Yes/No)

## Important Features

* Age
* MonthlyIncome
* OverTime
* JobSatisfaction
* WorkLifeBalance
* TotalWorkingYears
* YearsAtCompany
* JobRole
* Department

---

# ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Genetic Algorithm (sklearn-genetic-opt)

---

# 🔄 Project Workflow

## 1. Data Preprocessing

* Converted categorical data into numerical format
* Applied One-Hot Encoding
* Removed unnecessary features
* Checked for missing values

## 2. Exploratory Data Analysis (EDA)

Performed visualization and analysis to identify patterns related to attrition.

### Key Findings

* Employees working overtime are more likely to leave
* Lower salary is associated with higher attrition
* Early-career employees show higher attrition

## 3. Model Building

Built the following machine learning models:

* Logistic Regression
* Random Forest

## 4. Feature Optimization

Used Genetic Algorithm for feature selection to identify the most impactful features and optimize model performance.

## 5. Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* Confusion Matrix
* ROC-AUC Score

---

# 📈 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 87%      |
| Random Forest       | 87.4%    |

## ROC-AUC Score

* ROC-AUC ≈ 0.75

## Important Features Identified

* MonthlyIncome
* OverTime
* Age
* TotalWorkingYears
* DistanceFromHome
* YearsAtCompany

---

# 🎯 Business Insights

The project helps HR teams:

* Identify employees at high risk of attrition
* Improve employee retention strategies
* Reduce hiring and training costs
* Take proactive decisions using data insights

---

# 🚀 How to Run the Project

## Clone Repository

```bash
git clone https://github.com/KrishnaVamsi28/Employee_Attrition_Prediction.git
```

## Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn sklearn-genetic-opt
```

## Run Project

```bash
python scripts/attrition_analysis.py
```

---

# 📌 Future Improvements

* Improve recall for attrition prediction
* Handle class imbalance using SMOTE
* Build interactive Power BI dashboard
* Deploy model using Flask or Streamlit

---

# 👨‍💻 Author

Krishna Vamsi
