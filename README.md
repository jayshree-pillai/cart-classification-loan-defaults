# 💳 Loan Default Prediction with Profit-Driven Model Optimization

This project builds a profit-optimized loan default prediction pipeline using Logistic Regression, CART, Random Forest, and XGBoost. Models are evaluated not only on accuracy and AUC but also on **business profitability**, selecting thresholds that maximize net return per applicant.

---

## 🎯 Objective

Predict whether a loan applicant will default and determine the optimal decision cutoff that **maximizes profit**, rather than just model accuracy. This project simulates how real lenders can deploy interpretable and ensemble models aligned with business goals.

---

## 📦 Dataset

- `loans.csv` (real-world anonymized loan data)
- **Target:** `default` (1 = default, 0 = no default)
- **Features include:** financial variables, date/time fields, behavioral indicators (details anonymized for privacy)

---

## 🧠 Models Compared

- Logistic Regression (Logit)
- CART (`DecisionTreeClassifier`)
- Random Forest
- XGBoost

---

## 💰 Profit-Based Evaluation Strategy

- Profit matrix:
  - +$1000 for every correctly predicted non-default
  - –$5000 for every missed default
- Cutoffs (0.1–0.9) analyzed to find:
  - **Max profit per applicant**
  - **Optimal threshold**
- Custom evaluation functions:
  - `profit()`
  - `profit_table()`
  - `best_profit_and_cutoff()`

---

## 📈 Key Outcomes

- Tree-based models (Random Forest, XGBoost) consistently outperformed baselines
- Optimal cutoffs delivered >2x profitability vs naive thresholding
- Visualizations for tree structure and confusion matrix aid interpretability
- Logistic model included for business transparency benchmarking
- Profit-centric decision making
- Cost-sensitive cutoff tuning
- Trade-off between interpretability and performance

---

## 🛠️ Stack

- Python, Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib

---

## 📁 Files

- `CART-Classification.ipynb`: Full classification pipeline, evaluation, and tuning
- `loans.csv`: Input dataset (expected in same folder)

---
