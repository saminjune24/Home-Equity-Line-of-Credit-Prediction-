#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: saminbahizad
"""


import math
import pandas as pd
import numpy as np
from operator import itemgetter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn import tree
from sklearn.tree import _tree

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


sns.set()

DATA_FILE = "hmeq_eda_outputs/HMEQ_CLEANED_Output.csv"

TARGET_FLAG = "TARGET_BAD_FLAG"      # 1 = default, 0 = paid
TARGET_LOSS = "TARGET_LOSS_AMT"      # loss amount (defaults only)


df = pd.read_csv(DATA_FILE)

# Predictor matrix and targets
X = df.drop([TARGET_FLAG, TARGET_LOSS], axis=1)
Y = df[[TARGET_FLAG, TARGET_LOSS]].copy()

# Train / Test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=1
)

print("\nFLAG DATA")
print("TRAINING =", X_train.shape)
print("TEST     =", X_test.shape)


train_default_mask = (
    (Y_train[TARGET_FLAG] == 1) &
    (~Y_train[TARGET_LOSS].isna())
)

X_loss_train = X_train[train_default_mask]
Y_loss_train = Y_train.loc[train_default_mask, TARGET_LOSS]

test_default_mask = (
    (Y_test[TARGET_FLAG] == 1) &
    (~Y_test[TARGET_LOSS].isna())
)

X_loss_test = X_test[test_default_mask]
Y_loss_test = Y_test.loc[test_default_mask, TARGET_LOSS]

print("\nAMOUNT DATA (Defaults w/ Loss Only)")
print("TRAINING =", X_loss_train.shape)
print("TEST     =", X_loss_test.shape)


def get_tree_variables(model, variable_names):
    tree_struct = model.tree_
    used_vars = set()
    for i in tree_struct.feature:
        if i != _tree.TREE_UNDEFINED:
            used_vars.add(i)
    return [variable_names[i] for i in used_vars]


def get_ensemble_variables(model, variable_names):
    importances = model.feature_importances_
    avg_imp = np.mean(importances)

    output = []
    for i, val in enumerate(importances):
        if val > avg_imp:
            scaled = int(val / np.max(importances) * 100)
            output.append((variable_names[i], scaled))

    return sorted(output, key=itemgetter(1), reverse=True)


VARIABLE_NAMES = list(X.columns)


# Decision Tree Model

print("DECISION TREE")
print("=============")

# Default Prediction
dt_flag_model = tree.DecisionTreeClassifier(max_depth=3, random_state=1)
dt_flag_model.fit(X_train, Y_train[TARGET_FLAG])

train_pred = dt_flag_model.predict(X_train)
test_pred = dt_flag_model.predict(X_test)

print("\nProbability of DEFAULT")
print("Accuracy Train:", metrics.accuracy_score(Y_train[TARGET_FLAG], train_pred))
print("Accuracy Test :", metrics.accuracy_score(Y_test[TARGET_FLAG], test_pred))

# ROC Curve
train_probs = dt_flag_model.predict_proba(X_train)[:, 1]
fpr_train, tpr_train, _ = metrics.roc_curve(Y_train[TARGET_FLAG], train_probs)
auc_train = metrics.auc(fpr_train, tpr_train)

test_probs = dt_flag_model.predict_proba(X_test)[:, 1]
fpr_test, tpr_test, _ = metrics.roc_curve(Y_test[TARGET_FLAG], test_probs)
auc_test = metrics.auc(fpr_test, tpr_test)

fpr_tree, tpr_tree, auc_tree = fpr_test, tpr_test, auc_test

plt.title("Decision Tree ROC Curve")
plt.plot(fpr_train, tpr_train, label="AUC TRAIN = %0.2f" % auc_train)
plt.plot(fpr_test, tpr_test, label="AUC TEST  = %0.2f" % auc_test)
plt.plot([0, 1], [0, 1], "r--")
plt.legend(loc="lower right")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

tree.export_graphviz(
    dt_flag_model,
    out_file="hmeq_tree_flag.dot",
    feature_names=VARIABLE_NAMES,
    class_names=["Good", "Bad"],
    filled=True,
    rounded=True,
    impurity=False
)

print("\nVariables in Decision Tree Defaulted Loan:")
for v in get_tree_variables(dt_flag_model, VARIABLE_NAMES):
    print(v)


# Loss Prediction
dt_loss_model = tree.DecisionTreeRegressor(max_depth=4, random_state=1)
dt_loss_model.fit(X_loss_train, Y_loss_train)

train_loss_pred = dt_loss_model.predict(X_loss_train)
test_loss_pred = dt_loss_model.predict(X_loss_test)

rmse_train = math.sqrt(metrics.mean_squared_error(Y_loss_train, train_loss_pred))
rmse_test = math.sqrt(metrics.mean_squared_error(Y_loss_test, test_loss_pred))

print("\nLoss Amount | DEFAULT")
print("RMSE Train:", rmse_train)
print("RMSE Test :", rmse_test)

RMSE_TREE = rmse_test

tree.export_graphviz(
    dt_loss_model,
    out_file="hmeq_tree_amt.dot",
    feature_names=VARIABLE_NAMES,
    filled=True,
    rounded=True,
    impurity=False,
    precision=0
)

print("\nVariables in Decision Tree Loss Amount:")
for v in get_tree_variables(dt_loss_model, VARIABLE_NAMES):
    print(v)



# Random Forest Model
# =================================================
print("\n=============")
print("RANDOM FOREST")
print("=============")

rf_flag_model = RandomForestClassifier(n_estimators=100, random_state=1)
rf_flag_model.fit(X_train, Y_train[TARGET_FLAG])

print("\nProbability of Defaulted Loans")
print("Accuracy Train:", metrics.accuracy_score(Y_train[TARGET_FLAG], rf_flag_model.predict(X_train)))
print("Accuracy Test :", metrics.accuracy_score(Y_test[TARGET_FLAG], rf_flag_model.predict(X_test)))

rf_probs = rf_flag_model.predict_proba(X_test)[:, 1]
fpr_RF, tpr_RF, _ = metrics.roc_curve(Y_test[TARGET_FLAG], rf_probs)
auc_RF = metrics.auc(fpr_RF, tpr_RF)

print("\nVariables in Random Forest DEFAULT Model:")
for v in get_ensemble_variables(rf_flag_model, VARIABLE_NAMES):
    print(v)


rf_loss_model = RandomForestRegressor(n_estimators=200, random_state=1)
rf_loss_model.fit(X_loss_train, Y_loss_train)

test_loss_pred = rf_loss_model.predict(X_loss_test)
rmse_rf = math.sqrt(metrics.mean_squared_error(Y_loss_test, test_loss_pred))

print("\nLoss Amount | DEFAULT")
print("RMSE Test :", rmse_rf)

RMSE_RF = rmse_rf

print("\nVariables in Random Forest Loss Amount:")
for v in get_ensemble_variables(rf_loss_model, VARIABLE_NAMES):
    print(v)


# =================================================
# GRADIENT BOOSTING
# =================================================
print("\n=============")
print("GRADIENT BOOSTING")
print("=============")

gb_flag_model = GradientBoostingClassifier(random_state=1)
gb_flag_model.fit(X_train, Y_train[TARGET_FLAG])

print("\nProbability of DEFAULT")
print("Accuracy Train:", metrics.accuracy_score(Y_train[TARGET_FLAG], gb_flag_model.predict(X_train)))
print("Accuracy Test :", metrics.accuracy_score(Y_test[TARGET_FLAG], gb_flag_model.predict(X_test)))

gb_probs = gb_flag_model.predict_proba(X_test)[:, 1]
fpr_GB, tpr_GB, _ = metrics.roc_curve(Y_test[TARGET_FLAG], gb_probs)
auc_GB = metrics.auc(fpr_GB, tpr_GB)

print("\nVariables in Gradient Boosting Defaulted Loans:")
for v in get_ensemble_variables(gb_flag_model, VARIABLE_NAMES):
    print(v)


gb_loss_model = GradientBoostingRegressor(random_state=1)
gb_loss_model.fit(X_loss_train, Y_loss_train)

test_loss_pred = gb_loss_model.predict(X_loss_test)
rmse_gb = math.sqrt(metrics.mean_squared_error(Y_loss_test, test_loss_pred))

print("\nLoss Amount | DEFAULT")
print("RMSE Test :", rmse_gb)

RMSE_GB = rmse_gb

print("\nVariables in Gradient Boosting (LOSS):")
for v in get_ensemble_variables(gb_loss_model, VARIABLE_NAMES):
    print(v)


# =================================================
# ROC COMPARISON (TEST SET)
# =================================================
plt.title("ROC Curve Comparison (Test Set)")
plt.plot(fpr_tree, tpr_tree, label="TREE AUC = %0.2f" % auc_tree)
plt.plot(fpr_RF, tpr_RF, label="RF AUC = %0.2f" % auc_RF)
plt.plot(fpr_GB, tpr_GB, label="GB AUC = %0.2f" % auc_GB)
plt.plot([0, 1], [0, 1], "r--")
plt.legend(loc="lower right")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


print("\n==============================")
print("RMSE SUMMARY (Loss | Default)")
print("==============================")
print("TREE:", RMSE_TREE)
print("RF  :", RMSE_RF)
print("GB  :", RMSE_GB)
