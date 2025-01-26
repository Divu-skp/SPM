import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('Tesla.csv')

# Initial Exploration
print("Dataframe Head:\n", df.head())
print("Dataframe Shape:", df.shape)
df.info()

# Plot Tesla Closing Prices
plt.figure(figsize=(10, 4))
plt.plot(df['Close'], color='blue', linewidth=1.5)
plt.title('Tesla Close Price', fontsize=14, fontweight='bold')
plt.ylabel('Price in Dollars', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Drop 'Adj Close' column (duplicate of 'Close')
if 'Adj Close' in df.columns:
    df = df.drop(['Adj Close'], axis=1)

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Plot Distributions
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(10, 6))
for i, column in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.histplot(df[column], kde=True, color='skyblue')
    plt.title(f'Distribution of {column}', fontsize=10)
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot Boxplots
plt.subplots(figsize=(10, 6))
for i, column in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.boxplot(x=df[column], color='lightgreen')
    plt.title(f'Boxplot of {column}', fontsize=10)
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Feature Engineering
# Extract Day, Month, and Year from 'Date'
date_split = df['Date'].str.split('/', expand=True)
df['Day'] = date_split[1].astype('int')
df['Month'] = date_split[0].astype('int')
df['Year'] = date_split[2].astype('int')
df['is_quarter_end'] = np.where(df['Month'] % 3 == 0, 1, 0)
df = df.drop(['Date'], axis=1)

# Group Data by Year
yearly_avg = df.groupby('Year').mean()
plt.subplots(figsize=(10, 6))
for i, column in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 2, i + 1)
    yearly_avg[column].plot.bar(color='teal')
    plt.title(f'Yearly Avg of {column}', fontsize=10)
plt.tight_layout()
plt.show()

# Add new features
df['Open_Close_Diff'] = df['Open'] - df['Close']
df['Low_High_Diff'] = df['Low'] - df['High']
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Display Target Distribution
plt.pie(df['Target'].value_counts(), 
        labels=['Down', 'Up'], autopct='%1.1f%%', colors=['salmon', 'skyblue'])
plt.title('Target Distribution', fontsize=12)
plt.show()

# Plot Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sb.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()

# Data Preparation
feature_set = df[['Open_Close_Diff', 'Low_High_Diff', 'is_quarter_end']]
target = df['Target']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_set)

# Train-Test Split
X_train, X_valid, Y_train, Y_valid = train_test_split(
    scaled_features, target, test_size=0.1, random_state=2022
)
print("Train Shape:", X_train.shape, "Validation Shape:", X_valid.shape)

# Model Training and Evaluation
models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True),
    XGBClassifier()
]

for model in models:
    model.fit(X_train, Y_train)
    print(f"{model.__class__.__name__}:")
    print("Training ROC AUC:", metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1]))
    print("Validation ROC AUC:", metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:, 1]))
    print()

# Confusion Matrix for Logistic Regression
ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid, cmap='Blues')
plt.title('Confusion Matrix: Logistic Regression', fontsize=12)
plt.tight_layout()
plt.show()
