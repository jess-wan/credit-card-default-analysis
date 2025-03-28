🔶Credit Card Default Analysis Report🔶GROUP WORK


![image](https://github.com/user-attachments/assets/eb2e7e5b-957c-4c86-a33d-8f1afdfb1098)



---

INTRODUCTION

This project analyzes credit card default data to uncover trends and risk factors associated with credit defaults. By exploring demographic attributes, financial behaviors, and past payment history, we identify the key drivers influencing default rates. Using data visualization, we transform raw data into meaningful insights to enhance credit risk assessment.

**What Our Project Entails**
- **Default Rate Trends** – Understand how default rates vary across different customer segments.
- **Feature Relationships** – Identify which factors have the most significant impact on credit defaults.
- **Scatter Plots** – Explore connections between credit limit, income, and default risk.
- **Box Plot Analysis** – Compare default rates across different income and age groups.
- **Pair Plots and Heatmaps** – Analyze interactions between multiple financial and demographic features.

🔄️Data Preprocessing🔄️

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

📉Data Visualization & Insights.

We performed several visual analyses to gain insights into credit card defaults:

- **Default Rate Distribution**:
-  A histogram to show how defaults are distributed across the dataset.

```python
plt.figure(figsize=(8,5))
sns.histplot(credit['default'], bins=2, kde=True)
plt.title("Distribution of Default Rates")
plt.show()
```
![image](https://github.com/user-attachments/assets/b25bbaa7-3047-493a-8bba-d5dcaa1730e7)


- **Feature Correlation Heatmap**: Identifies how different financial and demographic factors correlate with default risk.

```python
plt.figure(figsize=(10,6))
sns.heatmap(credit.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

```
![image](https://github.com/user-attachments/assets/e3ed6fa0-a6c8-49d1-9f12-a1f40744c6a2)
Bill Statement Amounts Are Highly Correlated:

    The bill statement amounts from different months (e.g., bill_statement_sep, bill_statement_aug, etc.) exhibit strong positive correlations (close to 1.0).

    This suggests that customers with high outstanding balances in one month tend to have high balances in subsequent months.

Previous Payments Also Show Correlations:

    The previous payment amounts for different months (previous_payment_sep, previous_payment_aug, etc.) are strongly correlated.

    This indicates that customers who make higher payments in one month tend to continue making higher payments in following months.

Weak Correlation Between Credit Limit and Age:

    The correlation between limit_bal (credit limit) and age is weak (around 0.14), suggesting that credit limits are not significantly influenced by a customer's age.

    This implies that other factors, such as income or credit history, may play a more crucial role in determining credit limits.

Low Correlation Between Bill Statements and Previous Payments:

    The correlation between bill statements and previous payments is relatively weak.

    This suggests that higher outstanding balances do not always lead to proportionally higher payments, which could indicate financial distress in some customers.

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
EXAMPLE OF A SCATTER PLOT
    s Are Higher for Middle-Aged Individuals (30-50 Years Old):

        Most customers in this age range have higher credit limits compared to younger (20-30) and older (60+) individuals.

        This suggests that banks may assign higher credit limits to individuals in their peak earning years.

    Default Cases (Green Dots) Are Spread Across Age Groups but More Frequent at Lower Credit Limits:

        The green dots (defaulted accounts) are more concentrated in the lower range of credit limits.

        This indicates that customers with lower credit limits tend to default more often.

    Fewer Defaults at Higher Credit Limits:

        There are very few default cases among individuals with credit limits exceeding 600,000.

        This suggests that high-credit-limit customers may have better financial stability or more stringent approval criteria.

    Young Customers (20-30 Years) Show More Defaults:

        There is a higher concentration of default cases in the younger population.

        This could indicate that younger individuals may have riskier financial behavior, lower financial experience, or unstable income sources.

![image](https://github.com/user-attachments/assets/882ffbfe-f906-4776-88b3-443c855d0f9f)

- **Box Plot Analysis**:
-  Compares default rates based on factors like income level, employment status, and payment history.

```python
plt.figure(figsize=(8,5))
sns.boxplot(x=credit['income'], y=credit['default'])
plt.title("Income vs Default")
plt.show()
```
 BOX Plot Analysis Interpretation

Key Observations

    Interquartile Range (IQR) – The Box

        The box represents the middle 50% of the data (from Q1 to Q3).

        The median (central line inside the box) suggests that most individuals have a credit limit below 200,000.

        A large portion of the dataset has credit limits concentrated between 0 and ~300,000.

    Whiskers – Data Spread

        The whiskers extend to the minimum and maximum values within 1.5 times the IQR.

        The lower whisker suggests that some individuals have very low credit limits close to 0.

        The upper whisker shows the highest non-outlier credit limit is just above 400,000.

    Outliers (Dots Beyond the Whiskers)

        Many data points beyond the upper whisker represent outliers (customers with significantly higher credit limits).

        A few individuals have credit limits as high as 1,000,000, but they are rare.

        This suggests that most people have relatively low credit limits, while a few have exceptionally high ones.

Implications for Credit Default Analysis

    Higher Credit Limits Have Fewer Individuals

        Since outliers represent a small number of individuals, it could indicate that banks issue high credit limits only to financially stable customers.

        If we compare this to default rates, we would likely see lower default rates among high-credit-limit individuals.

    Defaults May Be Concentrated in the Lower Credit Limits

        Since the majority of customers fall in the lower range of credit limits, it is likely that most defaults happen in this range.

        This could mean riskier borrowers tend to have lower credit limits, either because they are new to credit or have lower financial stability.

    Need for Further Analysis

        A box plot of credit limits grouped by default status could confirm if lower credit limit holders default more often.

        Analyzing the credit limit vs. income or debt ratio could help explain why some individuals receive higher limits.

![image](https://github.com/user-attachments/assets/ce05774b-010c-49f8-a0e4-585a12f74765)

### **About the Dataset**
The dataset includes:
- **Demographic Information** (age, gender, marital status, etc.)
- **Financial Behavior** (credit limit, bill amounts, previous payments, etc.)
- **Default Status** (whether the customer defaulted on their payments)

✅Tools We Used
- **Python** – The foundation of the analysis.
- **Pandas** – For data organization and processing.
- **Seaborn & Matplotlib** – To create clear and informative visualizations.
- **Scikit-Learn** – For data preprocessing and machine learning modeling.
- **Jupyter Notebook** – For interactive data exploration and analysis.

💥Getting Started
To run an analysis on our project, follow these steps:
1. Load the dataset into a Pandas DataFrame.
2. Perform exploratory data analysis (EDA) to identify key trends and correlations.
3. Clean and preprocess the data by handling missing values and outliers.
4. Train machine learning models to predict credit default risk.
5. Evaluate model performance using accuracy, precision, recall, and F1-score.

Next Steps
- Engineer new features to improve predictive modeling.
- Experiment with machine learning algorithms to enhance default risk prediction.
- Perform additional statistical tests to validate insights.

---

💫CONCLUSION

This report provides a structured approach to analyzing credit card default risks. By leveraging data insights, we aim to improve risk management strategies and credit scoring models.

