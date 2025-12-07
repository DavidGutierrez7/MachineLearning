'''
    Code for the classification models exluding neural networks
'''

import mlflow
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import KBinsDiscretizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.utils import class_weight

K_FEATURES = [16, 17, 18, 19, 20]
CV = [5,10,15]

def main() -> None:
    BEST_AUC_H1N1 = 0
    BEST_CV_H1N1 = 0
    BEST_K_H1N1 = 0
    MODEL_H1N1 = ''

    BEST_AUC_SEASONAL = 0
    BEST_CV_SEASONAL = 0
    BEST_K_SEASONAL = 0
    MODEL_SEASONAL = ''
    # Cargar el CSV en un DataFrame
    df_vaccine = pd.read_csv('../datasets_procesados/training_set_processed.csv')
    df_labels = pd.read_csv('../datasets_originales/training_set_labels.csv')

    # Extracting the features and the targets
    features = df_vaccine.columns[:-2]
    X_train = df_vaccine[features]
    y_h1n1_train = df_labels['h1n1_vaccine']
    y_seasonal_train = df_labels['seasonal_vaccine']

    # Perform the split for initial evaluation of the models
    X_train_split, X_val_split, y_h1n1_train_split, y_h1n1_val_split = train_test_split(X_train, y_h1n1_train, test_size=0.2, random_state=42)
    _, _, y_seasonal_train_split, y_seasonal_val_split = train_test_split(X_train, y_seasonal_train, test_size=0.2, random_state=42)

    class_weights = class_weight.compute_class_weight('balanced', 
                                                classes=np.unique(y_h1n1_train_split), y=y_h1n1_train_split)
    for cv in CV:
        print(f'Cross-validation: {cv}')
        for k in K_FEATURES:
            print(f'K features: {k}')
            # Extract the most important features for each target
            k_features_h1n1 = extract_important_features(X_train, y_h1n1_train,features,k)
            k_features_seasonal = extract_important_features(X_train, y_seasonal_train,features,k)

            # Model's tryouts for h1n1_vaccine
            #roc_auc = tryout_knn_h1n1(X_train_split, y_h1n1_train_split, X_val_split, y_h1n1_val_split, k_features_h1n1,cv)
            roc_auc = tryout_logistic_regression_h1n1(X_train_split, y_h1n1_train_split, X_val_split, y_h1n1_val_split, k_features_h1n1,cv,class_weights)
            #roc_auc = tryout_svc_h1n1(X_train_split, y_h1n1_train_split, X_val_split, y_h1n1_val_split, k_features_h1n1,cv,class_weights)
            if roc_auc > BEST_AUC_H1N1:
                BEST_AUC_H1N1 = roc_auc
                BEST_CV_H1N1 = cv
                BEST_K_H1N1 = len(k_features_h1n1)
                MODEL_H1N1 = 'logistic_regression'
            # Model's tryouts for seasonal_vaccine
            #roc_auc = tryout_knn_seasonal(X_train_split, y_seasonal_train_split, X_val_split, y_seasonal_val_split, k_features_seasonal,cv)
            roc_auc = tryout_logistic_regression_seasonal(X_train_split, y_seasonal_train_split, X_val_split, y_seasonal_val_split, k_features_seasonal,cv)
            #roc_auc = tryout_svc_seasonal(X_train_split, y_seasonal_train_split, X_val_split, y_seasonal_val_split, k_features_seasonal,cv)
            if roc_auc > BEST_AUC_SEASONAL:
                BEST_AUC_SEASONAL = roc_auc
                BEST_CV_SEASONAL = cv
                BEST_K_SEASONAL = len(k_features_seasonal)
                MODEL_SEASONAL = 'logistic_regression'

    print(f'Best AUC H1N1: {BEST_AUC_H1N1}')
    print(f'Best CV H1N1: {BEST_CV_H1N1}')
    print(f'Best K H1N1: {BEST_K_H1N1}')
    print(f'Model H1N1: {MODEL_H1N1}')
    print('-------------------------------------------------')
    print(f'Best AUC Seasonal: {BEST_AUC_SEASONAL}')
    print(f'Best CV Seasonal: {BEST_CV_SEASONAL}')
    print(f'Best K Seasonal: {BEST_K_SEASONAL}')
    print(f'Model Seasonal: {MODEL_SEASONAL}')
    
def extract_important_features(X,y,features,n_features):
    # Select the best features with mutual information for h1n1_vaccine
    mutual_info = mutual_info_classif(X, y,random_state=42)
    print('Mutual information')
    mutual_info_df = pd.DataFrame({'Attributes': features,
                'Mutual information': mutual_info}).sort_values('Mutual information', ascending=False).head(n_features)

    k_features = mutual_info_df['Attributes']
    return k_features


def tryout_knn_h1n1(X_train,y_train,X_val,y_val,features_h1n1,cv):

    X_train = X_train[features_h1n1]
    X_val = X_val[features_h1n1]

    model = KNeighborsClassifier()

    # Set up the parameter grid
    param_grid = {'n_neighbors': list(range(1, 31)),  # Number of neighbors to use
                'weights': ['uniform', 'distance']  # Weight function used in prediction
                }

    # Set up the GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='roc_auc',
                            n_jobs=1, 
                            verbose=1)

    # Train the model
    grid_search.fit(X_train, y_train)

    # Predict on the test set
    y_pred = grid_search.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_score_ = f1_score(y_val, y_pred,average='binary',zero_division=0)
    precision = precision_score(y_val, y_pred,average='binary',zero_division=0)
    fpr, tpr, _ = metrics.roc_curve(y_val, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    print('h1n1_vaccine')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1_score_}')
    print(f'Precision: {precision}')
    print(f'ROC AUC: {roc_auc}')

    best_knn = grid_search.best_estimator_

    # Imprimimos la matriz de confusion
    cm = metrics.confusion_matrix(y_val, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
    disp.plot(cmap=plt.cm.Blues)
    
    return roc_auc

def tryout_knn_seasonal(X_train,y_train,X_val,y_val,features_seasonal,cv):

    X_train = X_train[features_seasonal]
    X_val = X_val[features_seasonal]

    model = KNeighborsClassifier()

    # Set up the parameter grid
    param_grid = {'n_neighbors': list(range(1, 31)),  # Number of neighbors to use
                'weights': ['uniform', 'distance'],  # Weight function used in prediction
                }

    # Set up the GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='roc_auc',
                            n_jobs=1, 
                            verbose=1)

    # Train the model
    grid_search.fit(X_train, y_train)

    # Predict on the test set
    y_pred = grid_search.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_score_ = f1_score(y_val, y_pred,average='binary',zero_division=0)
    precision = precision_score(y_val, y_pred,average='binary',zero_division=0)
    fpr, tpr, _ = metrics.roc_curve(y_val, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    print('Seasonal Vaccine')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1_score_}')
    print(f'Precision: {precision}')
    print(f'ROC AUC: {roc_auc}')

    best_knn = grid_search.best_estimator_

    # Imprimimos la matriz de confusion
    cm = metrics.confusion_matrix(y_val, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
    disp.plot(cmap=plt.cm.Blues)

    return roc_auc

def tryout_logistic_regression_h1n1(X_train,y_train,X_val,y_val,features_h1n1,cv,class_weights):

    X_train = X_train[features_h1n1]
    X_val = X_val[features_h1n1]

    model = LogisticRegression(random_state=42,max_iter=5000,class_weight={0:class_weights[0],1:class_weights[1]})

    # Set up the parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
        'penalty': ['l2'],  # Norm used in the penalization
        'solver': ['lbfgs', 'newton-cg', 'sag','saga'],  # Algorithm to use in the optimization problem
        'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # Tolerance for stopping criteria
    } 

    # Set up the GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='roc_auc',
                            n_jobs=-1, 
                            verbose=1)

    # Train the model
    grid_search.fit(X_train, y_train)

    # Predict on the test set
    y_pred = grid_search.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_score_ = f1_score(y_val, y_pred,average='macro',zero_division=0)
    precision = precision_score(y_val, y_pred,average='macro',zero_division=0)
    fpr, tpr, _ = metrics.roc_curve(y_val, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    print('h1n1_vaccine')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1_score_}')
    print(f'Precision: {precision}')
    print(f'ROC AUC: {roc_auc}')

    return roc_auc

def tryout_logistic_regression_seasonal(X_train,y_train,X_val,y_val,features_seasonal,cv):

    X_train = X_train[features_seasonal]
    X_val = X_val[features_seasonal]

    model = LogisticRegression(random_state=42,max_iter=5000)

    # Set up the parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
        'penalty': ['l2'],  # Norm used in the penalization
        'solver': ['lbfgs', 'newton-cg', 'sag','saga'],  # Algorithm to use in the optimization problem
        'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # Tolerance for stopping criteria
    } 

    # Set up the GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='roc_auc',
                            n_jobs=-1, 
                            verbose=1)

    # Train the model
    grid_search.fit(X_train, y_train)

    # Predict on the test set
    y_pred = grid_search.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_score_ = f1_score(y_val, y_pred,average='macro',zero_division=0)
    precision = precision_score(y_val, y_pred,average='macro',zero_division=0)
    fpr, tpr, _ = metrics.roc_curve(y_val, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    print('seasonal_vaccine')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1_score_}')
    print(f'Precision: {precision}')
    print(f'ROC AUC: {roc_auc}')

    return roc_auc

def tryout_svc_h1n1(X_train,y_train,X_val,y_val,features_h1n1,cv,class_weights):

    model = SVC(random_state=42,max_iter=5000,class_weight={0:class_weights[0],1:class_weights[1]})

    # Set up the parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type to be used in the algorithm
        'gamma': ['scale', 'auto'],  # Kernel coefficient
        'degree': [2, 3, 4]  # Degree of the polynomial kernel function
    } 

    # Set up the GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='roc_auc',
                            n_jobs=-1, 
                            verbose=1)

    # Train the model
    grid_search.fit(X_train[features_h1n1], y_train)

    # Predict on the test set
    y_pred = grid_search.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_score_ = f1_score(y_val, y_pred,average='macro',zero_division=0)
    precision = precision_score(y_val, y_pred,average='macro',zero_division=0)
    fpr, tpr, _ = metrics.roc_curve(y_val, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    print(f'H1N1 VACCINE')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1_score_}')
    print(f'Precision: {precision}')
    print(f'ROC AUC: {roc_auc}')

    # Imprimimos la matriz de confusion
    cm = metrics.confusion_matrix(y_val, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
    disp.plot(cmap=plt.cm.Blues)

    return roc_auc

def tryout_svc_seasonal(X_train,y_train,X_val,y_val,features_seasonal,cv):
    model = SVC(random_state=42,max_iter=5000)

    # Set up the parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Inverse of regularization strength
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type to be used in the algorithm
        'gamma': ['scale', 'auto'],  # Kernel coefficient
        'degree': [2, 3, 4]  # Degree of the polynomial kernel function
    } 

    # Set up the GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='roc_auc',
                            n_jobs=-1, 
                            verbose=1)

    # Train the model
    grid_search.fit(X_train[features_seasonal], y_train)

    # Predict on the test set
    y_pred = grid_search.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_score_ = f1_score(y_val, y_pred,average='macro',zero_division=0)
    precision = precision_score(y_val, y_pred,average='macro',zero_division=0)
    fpr, tpr, _ = metrics.roc_curve(y_val, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    print(f'SEASONAL VACCINE')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1_score_}')
    print(f'Precision: {precision}')
    print(f'ROC AUC: {roc_auc}')

    # Imprimimos la matriz de confusion
    cm = metrics.confusion_matrix(y_val, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
    disp.plot(cmap=plt.cm.Blues)

    return roc_auc

if __name__ == '__main__':
    main()