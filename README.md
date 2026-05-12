# Financial Risk Modeling: HELOC Default & Loss Severity Prediction

## 📌 The Business Problem
In consumer lending, accurately predicting credit default and estimating potential financial loss is vital for capital allocation and risk management. This project develops predictive models for a Home Equity Line of Credit (HMEQ) portfolio to predict two distinct outcomes:
1. **Default Probability (`TARGET_BAD_FLAG`):** Binary classification of loan default risk.
2. **Loss Severity (`TARGET_LOSS_AMT`):** Regression analysis to estimate the financial impact of a defaulted loan.

## 📊 Data Architecture & Sourcing
* **Data Source:** Proprietary Home Equity Loan (HMEQ) dataset containing 5,960 individual loan records.
* **Target Distribution:** The portfolio exhibits a ~19.95% default rate (1,189 defaults). Loss severity modeling was trained strictly on a subset of 941 defaulted accounts.
* **Predictor Variables (12 Features):**
  * *Categorical:* `REASON` (e.g., Debt Consolidation, Home Improvement), `JOB` (e.g., Self-Employed, Sales, Office).
  * *Numeric:* `LOAN` (Credit line amount), `MORTDUE` (Outstanding balance), `VALUE` (Property value), `DEBTINC` (Debt-to-income ratio), `DELINQ` (Recent delinquencies), `CLAGE` (Credit line age), `NINQ` (Recent inquiries), `CLNO` (Number of credit lines), `DEROG` (Derogatory marks), and `YOJ` (Years on job).

## 🛠️ Analytical Methodology
* **Feature & Correlation Analysis:** Identified that default probability is driven by behavioral leverage variables (e.g., Debt-to-Income, Delinquencies), whereas loss severity is predominantly driven by exposure size. Correlation analysis revealed that the `LOAN` amount has an 83.7% correlation with ultimate loss outcomes.
* **Algorithm Benchmarking:** Trained and evaluated multiple architectures utilizing an 80/20 train/test split:
  * *Traditional & Ensemble Trees:* Decision Trees, Random Forest, Gradient Boosting.
  * *Deep Learning:* TensorFlow Neural Networks testing various configurations, including 1 to 2 hidden layers, dropout regularization (20% rate to prevent overfitting), and multiple activation functions (ReLU, Tanh, Sigmoid).
* **Evaluation Metrics:** Classification models were evaluated on ROC/AUC and Accuracy, while loss regression models were evaluated on Root Mean Squared Error (RMSE).

## 🚀 Key Strategic Insights & Model Selection
* **The Winning Model:** Gradient Boosting outperformed Neural Networks and Random Forests, achieving the lowest test error (RMSE of ~$2,620) and high classification accuracy (>90%). 
* **Business Recommendation:** Gradient Boosting effectively captured non-linear financial risk relationships with limited data while maintaining better model explainability than the deep learning architectures—a critical regulatory requirement in financial services.
