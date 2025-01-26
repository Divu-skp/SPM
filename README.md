# Stock Price Prediction using Machine Learning

This project involves building a stock price prediction model using machine learning techniques. The dataset used in this project is based on Tesla's stock prices. The primary goal is to predict whether the stock's closing price will increase or decrease the following day. This repository includes exploratory data analysis, feature engineering, and model training.

## Features
- **Exploratory Data Analysis (EDA):**
  - Visualization of stock price trends
  - Distribution and outlier analysis using histograms and boxplots
  - Feature correlations using a heatmap

- **Feature Engineering:**
  - Extraction of new features such as `Open_Close_Diff`, `Low_High_Diff`, and `is_quarter_end`
  - Creation of a target variable indicating stock price movement

- **Machine Learning Models:**
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - XGBoost Classifier

- **Evaluation Metrics:**
  - ROC AUC Score
  - Confusion Matrix

## Dataset
The dataset used in this project includes the following columns:
- **Date:** The trading date
- **Open:** Opening price of the stock
- **High:** Highest price during the day
- **Low:** Lowest price during the day
- **Close:** Closing price of the stock
- **Adj Close:** Adjusted closing price
- **Volume:** Number of shares traded

The dataset should be saved as `Tesla.csv` in the root directory.

## Requirements
The following Python libraries are required:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

To install the required libraries, use:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Project Workflow

### 1. Data Loading
```python
# Load the dataset
df = pd.read_csv('Tesla.csv')
```

### 2. Data Exploration
- Display the first few rows of the dataset.
- Check the dataset's shape and data types.

### 3. Visualization
- Plot the closing price trend.
- Create distribution plots for features like `Open`, `High`, `Low`, `Close`, and `Volume`.
- Use boxplots to identify outliers.

### 4. Feature Engineering
- Extract `Day`, `Month`, and `Year` from the `Date` column.
- Create new features:
  - `Open_Close_Diff`: Difference between the opening and closing prices.
  - `Low_High_Diff`: Difference between the lowest and highest prices.
  - `is_quarter_end`: Boolean indicating if the date is a quarter-end.

- Define the target variable:
  ```python
  df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
  ```

### 5. Correlation Heatmap
Visualize correlations between features using a heatmap.
```python
sb.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

### 6. Data Preparation
- Normalize the features using `StandardScaler`.
- Split the data into training and validation sets:
  ```python
  X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)
  ```

### 7. Model Training and Evaluation
Train and evaluate the following models:
- **Logistic Regression**
- **SVC (Support Vector Classifier)**
- **XGBoost Classifier**

Evaluate the models using:
- Training and validation ROC AUC scores
- Confusion Matrix for Logistic Regression

### 8. Results and Visualization
- Display model performance metrics.
- Plot confusion matrices for better insights into classification results.

## File Structure
```
├── Tesla.csv           # Dataset
├── README.md           # Project documentation
├── stock_prediction.py # Python script containing the full code
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Divu-skp/SPM.git
   ```
2. Navigate to the project directory:
   ```bash
   cd stock-price-prediction
   ```
3. Run the Python script:
   ```bash
   python stock_prediction.py
   ```

## Results
The results include:
- Visualizations showing trends and distributions in the data
- Model evaluation metrics (ROC AUC scores)
- Confusion matrix analysis

## References
This project is based on the methodology described in the [GeeksforGeeks article](https://www.geeksforgeeks.org/stock-price-prediction-using-machine-learning-in-python/).

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---
Feel free to contribute to this project by submitting issues or pull requests!
