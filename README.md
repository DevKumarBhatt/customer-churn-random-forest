# Customer Churn Prediction Using Random Forest

## 📌 Project Overview

This project predicts whether a customer is likely to **churn (leave the service)** using a **Random Forest Classification** model.

The project covers the complete machine learning workflow, including data preprocessing, visualization, train-test splitting, model training, prediction, and model evaluation.

## 🎯 Objective

The main objective of this project is to build a machine learning model that can identify customers who are likely to churn based on their available customer information.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Jupyter Notebook

## 🤖 Machine Learning Model

**Random Forest Classifier**

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions. It is suitable for classification problems such as customer churn prediction.

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=10,
    random_state=42
)
```

- `n_estimators=10` → Uses 10 decision trees.
- `random_state=42` → Makes the results reproducible.

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Data Visualization
   ↓
Feature & Target Selection
   ↓
Train-Test Split
   ↓
Random Forest Model
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Customer Churn Prediction
```

## 📊 Data Visualization

The project includes a churn distribution visualization using Seaborn to understand the number of customers who stayed and customers who churned.

```python
sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.show()
```

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using an 80:20 ratio.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

`stratify=y` helps maintain a similar distribution of the target classes in both datasets.

## 📈 Model Evaluation

The Random Forest model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

### Accuracy

Accuracy measures the proportion of correct predictions made by the model.

### Classification Report

The classification report provides:

- Precision
- Recall
- F1-score
- Support

### Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions for each class.

```python
cm = confusion_matrix(y_test, y_pred)
```

### ROC-AUC

ROC-AUC is calculated using the predicted churn probabilities:

```python
y_probability = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(
    y_test,
    y_probability
)
```

## 🔮 New Customer Prediction

The trained model can also be used to predict churn for a new customer.

```python
prediction = model.predict(new_customer)

print("Prediction:", prediction)
```

The model can also provide the probability of the predicted class using:

```python
model.predict_proba(new_customer)
```

## 📁 Project Structure

```text
M6_Random_Forest/
│
├── customer_churn_random_forest.csv
├── M6_Random_Forest.ipynb
├── customer_churn_final_results.txt
├── README.md
└── .gitignore
```

> If the CSV contains sensitive or restricted data, it should not be uploaded to a public GitHub repository.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/DevKumarBhatt/M6_Random_Forest.git
```

### 2. Open the notebook

Open:

```text
M6_Random_Forest.ipynb
```

using Jupyter Notebook or Google Colab.

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Load the dataset

Make sure the dataset path is correctly configured before running the notebook.

### 5. Run the notebook

Run the cells in order from data loading through model evaluation.

## 📌 Key Learning Outcomes

Through this project, I practiced:

- Loading and exploring a dataset
- Data visualization using Matplotlib and Seaborn
- Feature and target selection
- Train-test splitting
- Random Forest classification
- Model prediction
- Classification metrics
- Confusion matrix visualization
- Probability-based predictions
- ROC-AUC evaluation
- Saving machine learning results

## 👨‍💻 Author

**Dev Kumar Bhatt**

- GitHub: [DevKumarBhatt](https://github.com/DevKumarBhatt)
- LinkedIn: [Dev Kumar Bhatt](https://linkedin.com/in/dev-kumar-bhatt-b74540347)

## 📄 License

This project is available for educational and learning purposes.
