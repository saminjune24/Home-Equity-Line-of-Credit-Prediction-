#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday Feb 19 23:48:56 2026

@author: saminbahizad

HMEQ - Comparing Tree, Regression and  TensorFlow Models

This script:
1. Builds baseline models:
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Logistic / Linear Regression (all variables)

2. Builds TensorFlow models for:
   - Default probability
   - Loss amount given default

TensorFlow requirements covered:
- at least 3 activation functions: relu, tanh, sigmoid
- 1 and 2 hidden layers
- dropout layer
- variable selection technique

Outputs:
- ROC charts for each baseline and TF default model
- Combined ROC chart: best TF default model vs baselines
- Combined RMSE chart: best TF loss model vs baselines
"""

import math
import pandas as pd
import numpy as np
from operator import itemgetter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn import tree
from sklearn.tree import _tree

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

sns.set()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

"""
Input Data
"""

INFILE = "hmeq_eda_outputs/HMEQ_CLEANED_Output.csv"

TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"

df = pd.read_csv( INFILE )

X = df.drop( [ TARGET_F, TARGET_A ], axis=1 ).copy()
Y = df[ [ TARGET_F, TARGET_A ] ].copy()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=1
)

# Regression subset: only defaults with observed loss
F = ~Y_train[ TARGET_A ].isna()
W_train = X_train[F].copy()
Z_train = Y_train[F].copy()

F = ~Y_test[ TARGET_A ].isna()
W_test = X_test[F].copy()
Z_test = Y_test[F].copy()

feature_cols = list( X.columns.values )

print( "\nFLAG DATA" )
print( "TRAINING =", X_train.shape )
print( "TEST     =", X_test.shape )

print( "\nAMOUNT DATA (Defaults w/ Loss Only)" )
print( "TRAINING =", W_train.shape )
print( "TEST     =", W_test.shape )

"""
MODEL ACCURACY METRICS
"""

def getProbAccuracyScores( NAME, MODEL, X_in, Y_in ) :
    pred      = MODEL.predict( X_in )
    probs     = MODEL.predict_proba( X_in )
    acc_score = metrics.accuracy_score( Y_in, pred )
    p1        = probs[ :, 1 ]
    fpr, tpr, threshold = metrics.roc_curve( Y_in, p1 )
    auc = metrics.auc( fpr, tpr )
    return [ NAME, acc_score, fpr, tpr, auc ]

def save_ROC_Curve( TITLE, LIST, FILENAME ) :
    fig = plt.figure( figsize=( 6, 4 ) )
    plt.title( TITLE )
    for theResults in LIST :
        NAME     = theResults[0]
        fpr      = theResults[2]
        tpr      = theResults[3]
        auc      = theResults[4]
        theLabel = "AUC " + NAME + ' %0.2f' % auc
        plt.plot( fpr, tpr, label=theLabel )
    plt.legend( loc='lower right' )
    plt.plot( [0, 1], [0, 1], 'r--' )
    plt.xlim( [0, 1] )
    plt.ylim( [0, 1] )
    plt.ylabel( 'True Positive Rate' )
    plt.xlabel( 'False Positive Rate' )
    plt.tight_layout()
    plt.savefig( FILENAME, dpi=120 )
    plt.close()

def print_Accuracy( TITLE, LIST ) :
    print( TITLE )
    print( "======" )
    for theResults in LIST :
        NAME = theResults[0]
        ACC  = theResults[1]
        print( NAME, " = ", ACC )
    print( "------\n\n" )

def getAmtAccuracyScores( NAME, MODEL, X_in, Y_in ) :
    pred = MODEL.predict( X_in )
    pred = np.array( pred ).reshape( -1 )
    MEAN = float( np.mean( Y_in ) )
    RMSE = math.sqrt( metrics.mean_squared_error( Y_in, pred ) )
    return [ NAME, RMSE, MEAN ]

def save_RMSE_Chart( TITLE, LIST, FILENAME ) :
    fig = plt.figure( figsize=( 6, 4 ) )
    plt.title( TITLE )
    labels = [ x[0] for x in LIST ]
    values = [ x[1] for x in LIST ]
    plt.bar( labels, values )
    plt.xticks( rotation=35, ha="right" )
    plt.ylabel( "RMSE ($)" )
    plt.xlabel( "Model" )
    plt.tight_layout()
    plt.savefig( FILENAME, dpi=120 )
    plt.close()


"""
VARIABLE SELECTION HELPERS
"""

def getTreeVars( TREE, varNames ) :
    tree_    = TREE.tree_
    nameSet  = set()
    for i in tree_.feature :
        if i != _tree.TREE_UNDEFINED :
            nameSet.add( i )
    nameList       = list( nameSet )
    parameter_list = []
    for i in nameList :
        parameter_list.append( varNames[i] )
    return parameter_list

def getEnsembleTreeVars( ENSTREE, varNames ) :
    importance = ENSTREE.feature_importances_
    index      = np.argsort( importance )
    theList    = []
    for i in index :
        imp_val = importance[i]
        if imp_val > np.average( ENSTREE.feature_importances_ ) :
            v = int( imp_val / np.max( ENSTREE.feature_importances_ ) * 100 )
            theList.append( ( varNames[i], v ) )
    theList = sorted( theList, key=itemgetter(1), reverse=True )
    return theList

"""
BASELINE MODELS
"""

"""
DECISION TREE
"""

WHO = "TREE"

CLM = tree.DecisionTreeClassifier( max_depth=4, random_state=1 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM  = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

save_ROC_Curve(
    WHO + " - Probability of Loan Default",
    [ TRAIN_CLM, TEST_CLM ],
    "hmeq_eda_outputs/roc_TREE.png"
)
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

vars_tree_flag = getTreeVars( CLM, feature_cols )

AMT = tree.DecisionTreeRegressor( max_depth=4, random_state=1 )
AMT = AMT.fit( W_train, Z_train[ TARGET_A ] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train, Z_train[ TARGET_A ] )
TEST_AMT  = getAmtAccuracyScores( WHO, AMT, W_test, Z_test[ TARGET_A ] )

print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

vars_tree_amt = getTreeVars( AMT, feature_cols )

TREE_CLM = TEST_CLM.copy()
TREE_AMT = TEST_AMT.copy()


"""
RANDOM FOREST
"""

WHO = "RF"

CLM = RandomForestClassifier( n_estimators=25, random_state=1 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM  = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

save_ROC_Curve(
    WHO + " - Probability of Loan Default",
    [ TRAIN_CLM, TEST_CLM ],
    "hmeq_eda_outputs/roc_RF.png"
)
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

vars_RF_flag = getEnsembleTreeVars( CLM, feature_cols )

AMT = RandomForestRegressor( n_estimators=100, random_state=1 )
AMT = AMT.fit( W_train, Z_train[ TARGET_A ] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train, Z_train[ TARGET_A ] )
TEST_AMT  = getAmtAccuracyScores( WHO, AMT, W_test, Z_test[ TARGET_A ] )

print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

vars_RF_amt = getEnsembleTreeVars( AMT, feature_cols )

RF_CLM = TEST_CLM.copy()
RF_AMT = TEST_AMT.copy()


"""
GRADIENT BOOSTING
"""

WHO = "GB"

CLM = GradientBoostingClassifier( random_state=1 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM  = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

save_ROC_Curve(
    WHO + " - Probability of Loan Default",
    [ TRAIN_CLM, TEST_CLM ],
    "hmeq_eda_outputs/roc_GB.png"
)
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

vars_GB_flag = getEnsembleTreeVars( CLM, feature_cols )

AMT = GradientBoostingRegressor( random_state=1 )
AMT = AMT.fit( W_train, Z_train[ TARGET_A ] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train, Z_train[ TARGET_A ] )
TEST_AMT  = getAmtAccuracyScores( WHO, AMT, W_test, Z_test[ TARGET_A ] )

print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

vars_GB_amt = getEnsembleTreeVars( AMT, feature_cols )

GB_CLM = TEST_CLM.copy()
GB_AMT = TEST_AMT.copy()


"""
LOGISTIC / LINEAR REGRESSION (ALL VARIABLES)
"""

WHO = "REG_ALL"

CLM = LogisticRegression( solver='newton-cg', max_iter=1000 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM  = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

save_ROC_Curve(
    WHO + " - Probability of Loan Default",
    [ TRAIN_CLM, TEST_CLM ],
    "hmeq_eda_outputs/roc_REG_ALL.png"
)
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

AMT = LinearRegression()
AMT = AMT.fit( W_train, Z_train[ TARGET_A ] )

TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train, Z_train[ TARGET_A ] )
TEST_AMT  = getAmtAccuracyScores( WHO, AMT, W_test, Z_test[ TARGET_A ] )

print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

REG_ALL_CLM = TEST_CLM.copy()
REG_ALL_AMT = TEST_AMT.copy()

"""
TENSORFLOW
"""

print( "\nTensorFlow version:", tf.__version__ )

tf.random.set_seed( 1 )
np.random.seed( 1 )


def get_TF_ProbAccuracyScores( NAME, MODEL, X, Y ) :
    probs = MODEL.predict( X, verbose=False )
    probs = probs.reshape( -1 )
    pred  = ( probs >= 0.5 ).astype( int )
    acc_score = metrics.accuracy_score( Y, pred )
    fpr, tpr, threshold = metrics.roc_curve( Y, probs )
    auc = metrics.auc( fpr, tpr )
    return [ NAME, acc_score, fpr, tpr, auc ]


def get_TF_AmtAccuracyScores( NAME, MODEL, X, Y_actual ) :
    # predictions are in log1p scale — convert back to dollars with expm1
    pred_log = MODEL.predict( X, verbose=False ).reshape( -1 )
    pred = np.expm1( pred_log )
    pred = np.maximum( pred, 0 )
    RMSE = math.sqrt( metrics.mean_squared_error( Y_actual, pred ) )
    MEAN = float( np.mean( Y_actual ) )
    return [ NAME, RMSE, MEAN ]


"""
SCALE DATA
"""

theScaler = MinMaxScaler()
theScaler.fit( X_train )

U_train = theScaler.transform( X_train )
U_test  = theScaler.transform( X_test )

U_train = pd.DataFrame( U_train )
U_test  = pd.DataFrame( U_test )

U_train.columns = list( X_train.columns.values )
U_test.columns  = list( X_train.columns.values )

V_train = theScaler.transform( W_train )
V_test  = theScaler.transform( W_test )

V_train = pd.DataFrame( V_train )
V_test  = pd.DataFrame( V_test )

V_train.columns = list( W_train.columns.values )
V_test.columns  = list( W_train.columns.values )


# Variable selection: GB-selected vars for flag model; all vars for loss model
TF_flag_vars = [ v[0] for v in vars_GB_flag ]
TF_amt_vars  = list( W_train.columns )

U_train = U_train[ TF_flag_vars ]
U_test  = U_test[ TF_flag_vars ]

V_train = V_train[ TF_amt_vars ]
V_test  = V_test[ TF_amt_vars ]

# Convert to numpy arrays
U_train_np = U_train.values.astype( np.float32 )
U_test_np  = U_test.values.astype( np.float32 )

V_train_np = V_train.values.astype( np.float32 )
V_test_np  = V_test.values.astype( np.float32 )

y_flag_train_np = Y_train[ TARGET_F ].values.astype( np.float32 )
y_flag_test_np  = Y_test[ TARGET_F ].values.astype( np.float32 )

# Loss target: log1p transform reduces skew; predictions inverted with expm1
y_amt_train_raw = Z_train[ TARGET_A ].values.astype( np.float32 )
y_amt_test_raw  = Z_test[ TARGET_A ].values.astype( np.float32 )

y_amt_train_np = np.log1p( y_amt_train_raw )
y_amt_test_np  = np.log1p( y_amt_test_raw )


"""
EARLY STOPPING CALLBACKS
"""

earlyStop_flag = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    mode="min"
)

earlyStop_amt = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    mode="min"
)


"""
TENSORFLOW MODEL LOOP
- 3 activation functions: relu, tanh, sigmoid
- 1 and 2 hidden layers
- dropout layer variant
- variable selection applied above
"""

TF_MODEL_SPECS = [
    ( "TF_relu_1HL",        tf.keras.activations.relu,    1, None  ),
    ( "TF_tanh_2HL",        tf.keras.activations.tanh,    2, None  ),
    ( "TF_sigmoid_1HL",     tf.keras.activations.sigmoid, 1, None  ),
    ( "TF_relu_2HL_Drop20", tf.keras.activations.relu,    2, 0.20  ),
]

TF_CLM_TEST = []
TF_AMT_TEST = []

for WHO, theActivation, theHiddenLayers, theDropout in TF_MODEL_SPECS :

    print( "Training", WHO, "..." )

    """
    DEFAULT PROBABILITY MODEL
    """

    F_theShapeSize = U_train_np.shape[1]
    F_theUnits     = F_theShapeSize * 2

    F_LAYER_01     = tf.keras.layers.Dense( units=F_theUnits, activation=theActivation, input_dim=F_theShapeSize )
    F_LAYER_DROP   = tf.keras.layers.Dropout( theDropout if theDropout else 0 )
    F_LAYER_02     = tf.keras.layers.Dense( units=F_theUnits, activation=theActivation )
    F_LAYER_OUTPUT = tf.keras.layers.Dense( units=1, activation=tf.keras.activations.sigmoid )

    CLM = tf.keras.Sequential()
    CLM.add( F_LAYER_01 )
    if theDropout is not None :
        CLM.add( F_LAYER_DROP )
    if theHiddenLayers == 2 :
        CLM.add( F_LAYER_02 )
    CLM.add( F_LAYER_OUTPUT )
    CLM.compile( loss="binary_crossentropy", optimizer="adam" )
    CLM.fit( U_train_np, y_flag_train_np,
             validation_split=0.20,
             epochs=40,
             batch_size=512,
             verbose=False,
             callbacks=[ earlyStop_flag ] )

    TRAIN_CLM = get_TF_ProbAccuracyScores( WHO + "_Train", CLM, U_train_np, y_flag_train_np )
    TEST_CLM  = get_TF_ProbAccuracyScores( WHO, CLM, U_test_np, y_flag_test_np )

    save_ROC_Curve(
        WHO + " - Probability of Loan Default",
        [ TRAIN_CLM, TEST_CLM ],
        "hmeq_eda_outputs/roc_" + WHO + ".png"
    )
    print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

    TF_CLM_TEST.append( TEST_CLM )

    """
    LOSS AMOUNT MODEL
    - target: log1p( TARGET_LOSS_AMT ) — reduces skew; inverted with expm1 at evaluation
    - all features from loss subset used (no variable selection filter)
    - Huber loss (delta=0.5) — robust to large loss outliers
    - wider first layer (64 units), second layer (32 units)
    """

    print( "Training", WHO + "_LOSS", "..." )

    A_theShapeSize = V_train_np.shape[1]
    A_theOptimizer = tf.keras.optimizers.Adam( learning_rate=0.001 )
    A_theLoss      = tf.keras.losses.Huber( delta=0.5 )

    A_LAYER_01     = tf.keras.layers.Dense( units=64, activation=theActivation, input_dim=A_theShapeSize )
    A_LAYER_DROP   = tf.keras.layers.Dropout( theDropout if theDropout else 0 )
    A_LAYER_02     = tf.keras.layers.Dense( units=32, activation=theActivation )
    A_LAYER_OUTPUT = tf.keras.layers.Dense( units=1,  activation=tf.keras.activations.linear )

    AMT = tf.keras.Sequential()
    AMT.add( A_LAYER_01 )
    if theDropout is not None :
        AMT.add( A_LAYER_DROP )
    if theHiddenLayers == 2 :
        AMT.add( A_LAYER_02 )
    AMT.add( A_LAYER_OUTPUT )
    AMT.compile( loss=A_theLoss, optimizer=A_theOptimizer )
    AMT.fit( V_train_np, y_amt_train_np,
             validation_split=0.20,
             epochs=400,
             batch_size=128,
             verbose=False,
             callbacks=[ earlyStop_amt ] )

    TRAIN_AMT = get_TF_AmtAccuracyScores( WHO + "_LOSS_Train", AMT, V_train_np, y_amt_train_raw )
    TEST_AMT  = get_TF_AmtAccuracyScores( WHO + "_LOSS",       AMT, V_test_np,  y_amt_test_raw  )

    print_Accuracy( WHO + "_LOSS RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

    TF_AMT_TEST.append( TEST_AMT )


"""
BEST TF MODELS
"""

BEST_TF_CLM = sorted( TF_CLM_TEST, key=lambda x: x[4], reverse=True )[ 0 ]
BEST_TF_AMT = sorted( TF_AMT_TEST, key=lambda x: x[1] )[ 0 ]

print( "\nBEST TF DEFAULT MODEL:", BEST_TF_CLM[0], "AUC =",  round( BEST_TF_CLM[4], 4 ) )
print( "BEST TF LOSS MODEL:",    BEST_TF_AMT[0], "RMSE =", round( BEST_TF_AMT[1], 2 ) )


"""
ALL MODELS - ROC CURVE (TEST)
Tree , Regression and Best TF Default Model
"""

ALL_CLM = [ TREE_CLM, RF_CLM, GB_CLM, REG_ALL_CLM, BEST_TF_CLM ]
ALL_CLM = sorted( ALL_CLM, key=lambda x: x[4], reverse=True )

save_ROC_Curve(
    "ALL MODELS - ROC (Test) - Tree + Regression + TensorFlow",
    ALL_CLM,
    "hmeq_eda_outputs/roc_ALL_MODELS_WITH_TF.png"
)


"""
ALL MODELS - LOSS AMOUNT RMSE (TEST)
Tree, Regression and Best TF Loss Model
"""

ALL_AMT = [ TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, BEST_TF_AMT ]
ALL_AMT = sorted( ALL_AMT, key=lambda x: x[1] )

print_Accuracy(
    "ALL MODELS - LOSS AMOUNT RMSE (Test) - Tree , Regression andTensorFlow",
    ALL_AMT
)

save_RMSE_Chart(
    "ALL MODELS - Loss Amount RMSE (Test) - Tree , Regression and TensorFlow",
    ALL_AMT,
    "hmeq_eda_outputs/rmse_ALL_MODELS_WITH_TF.png"
)

print( "\nDone. All charts saved." )
