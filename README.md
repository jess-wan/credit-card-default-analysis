**Credit Card Default Analysis Report**

**group work**

---

### **Introduction**
This project analyzes credit card default data to uncover trends and risk factors associated with credit defaults. By exploring demographic attributes, financial behaviors, and past payment history, we identify the key drivers influencing default rates. Using data visualization, we transform raw data into meaningful insights to enhance credit risk assessment.

### **What Our Project Entails**
- **Default Rate Trends** – Understand how default rates vary across different customer segments.
- **Feature Relationships** – Identify which factors have the most significant impact on credit defaults.
- **Scatter Plots** – Explore connections between credit limit, income, and default risk.
- **Box Plot Analysis** – Compare default rates across different income and age groups.
- **Pair Plots and Heatmaps** – Analyze interactions between multiple financial and demographic features.

### **Data Preprocessing**
We performed several preprocessing steps to clean and prepare the dataset for analysis:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
credit = pd.read_csv("credit_card_default.csv")

# Checking for missing values
print(credit.isnull().sum())

# Handling missing values (if any)
credit.fillna(credit.median(), inplace=True)

# Splitting data into features and target variable
y = credit['default']
X = credit.drop(columns=['default'])

# Standardizing numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

📉### **Data Visualization & Insights**
We performed several visual analyses to gain insights into credit card defaults:

- **Default Rate Distribution**: A histogram to show how defaults are distributed across the dataset.

```python
plt.figure(figsize=(8,5))
sns.histplot(credit['default'], bins=2, kde=True)
plt.title("Distribution of Default Rates")
plt.show()
```

- **Feature Correlation Heatmap**: Identifies how different financial and demographic factors correlate with default risk.

```python
plt.figure(figsize=(10,6))
sns.heatmap(credit.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()
```

- **Scatter Plots**:
  - **Credit Limit vs. Default Rate**: Customers with lower credit limits may have higher default rates.
  - **Age vs. Default Rate**: Younger individuals might have higher default risks compared to older demographics.
  - **Payment History vs. Default Rate**: Late payments correlate strongly with increased default probability.

```python
plt.figure(figsize=(8,5))
sns.scatterplot(x=credit['credit_limit'], y=credit['default'])
plt.title("Credit Limit vs Default")
plt.show()
```

- **Box Plot Analysis**: Compares default rates based on factors like income level, employment status, and payment history.

```python
plt.figure(figsize=(8,5))
sns.boxplot(x=credit['income'], y=credit['default'])
plt.title("Income vs Default")
plt.show()
```

### **About the Dataset**
The dataset includes:
- **Demographic Information** (age, gender, marital status, etc.)
- **Financial Behavior** (credit limit, bill amounts, previous payments, etc.)
- **Default Status** (whether the customer defaulted on their payments)

### **Tools We Used**
- **Python** – The foundation of the analysis.
- **Pandas** – For data organization and processing.
- **Seaborn & Matplotlib** – To create clear and informative visualizations.
- **Scikit-Learn** – For data preprocessing and machine learning modeling.
- **Jupyter Notebook** – For interactive data exploration and analysis.

### **Getting Started**
To run an analysis on our project, follow these steps:
1. Load the dataset into a Pandas DataFrame.
2. Perform exploratory data analysis (EDA) to identify key trends and correlations.
3. Clean and preprocess the data by handling missing values and outliers.
4. Train machine learning models to predict credit default risk.
5. Evaluate model performance using accuracy, precision, recall, and F1-score.

### **Next Steps**
- Engineer new features to improve predictive modeling.
- Experiment with machine learning algorithms to enhance default risk prediction.
- Perform additional statistical tests to validate insights.

---

**LET’S GET STARTED**

This report provides a structured approach to analyzing credit card default risks. By leveraging data insights, we aim to improve risk management strategies and credit scoring models.

