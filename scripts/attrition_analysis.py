import pandas as pd

# load dataset
df = pd.read_csv("dataset/employee.csv")

# show first 5 rows
print("First 5 rows of dataset:\n")
print(df.head())

# show dataset shape
print("\nDataset Shape (rows, columns):")
print(df.shape)

# show column names
print("\nColumns in dataset:")
print(df.columns)

# dataset info
print("\nDataset Information:\n")
print(df.info())

# statistical summary
print("\nStatistical Summary:\n")
print(df.describe())

# missing values
print("\nMissing Values in each column:\n")
print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Attrition distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.show()

# Attrition vs Overtime
plt.figure(figsize=(6,4))
sns.countplot(x='OverTime', hue='Attrition', data=df)

plt.title("Attrition vs Overtime")
plt.xlabel("OverTime")
plt.ylabel("Number of Employees")

plt.show()

# Correlation heatmap

plt.figure(figsize=(14,10))

# select only numerical columns
numeric_df = df.select_dtypes(include=['int64'])

sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')

plt.title("Feature Correlation Heatmap")

plt.show()

# -------------------------------
# Data Preprocessing
# -------------------------------

# convert Attrition to numeric
df['Attrition'] = df['Attrition'].map({'Yes':1, 'No':0})

# convert Overtime
df['OverTime'] = df['OverTime'].map({'Yes':1, 'No':0})

# convert Gender
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

# drop unnecessary columns
df = df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis=1)

print("\nDataset after preprocessing:")
print(df.head())

# -------------------------------
# Encode categorical variables
# -------------------------------

# one-hot encoding for categorical columns
df = pd.get_dummies(df, drop_first=True)

print("\nDataset after encoding categorical variables:")
print(df.head())

print("\nNew dataset shape after encoding:")
print(df.shape)

from sklearn.model_selection import train_test_split

# separate features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# -------------------------------
# Logistic Regression Model
# -------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# create model
log_model = LogisticRegression(max_iter=1000)

# train model
log_model.fit(X_train, y_train)

# predictions
y_pred = log_model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Random Forest Model
# -------------------------------

from sklearn.ensemble import RandomForestClassifier

# create model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# train model
rf_model.fit(X_train, y_train)

# predictions
y_pred_rf = rf_model.predict(X_test)

# accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:", accuracy_rf)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# -------------------------------
# Feature Importance
# -------------------------------

import pandas as pd

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

# top 10 important features
top_features = feature_importances.sort_values(ascending=False).head(10)

print("\nTop 10 Important Features:")
print(top_features)

# -------------------------------
# Genetic Algorithm Feature Selection
# -------------------------------

from sklearn_genetic import GAFeatureSelectionCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

ga_selector = GAFeatureSelectionCV(
    estimator=rf,
    cv=5,
    scoring="accuracy",
    population_size=20,
    generations=10,
    n_jobs=-1,
    verbose=True
)

ga_selector.fit(X_train, y_train)

# selected features
selected_features = X_train.columns[ga_selector.support_]

print("\nSelected Features using Genetic Algorithm:")
print(selected_features)

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# probability predictions
y_prob = rf_model.predict_proba(X_test)[:,1]

# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# calculate AUC score
auc_score = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", auc_score)

# plot ROC curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0,1],[0,1],'r--')

plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()
plt.show()