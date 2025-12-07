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
from sklearn import metrics
from ucimlrepo import fetch_ucirepo

N_BINS = 7
CV = 4
matrix_dir = "matrix_plots/"

def main():
    # Train different models of clasification with the same dataset
    X, y_enc, df_wine_aux = load_data(N_BINS)
    X_train, X_test, y_train, y_test, X_train_, X_test_, y_train_, y_test_ = split_data(X,y_enc,df_wine_aux)
    decision_tree_model(X_train_, X_test_, y_train_, y_test_,N_BINS)
    gaussian_naive_bayes_model(X_train, X_test, y_train, y_test,N_BINS)
    multinomial_naive_bayes_model(X_train_, X_test_, y_train_, y_test_,N_BINS)
    complement_naive_bayes_model(X_train_, X_test_, y_train_, y_test_,N_BINS)
    knn_model(X_train, X_test, y_train, y_test, N_BINS)
    random_forest_model(X_train_, X_test_, y_train_, y_test_,N_BINS)
    print(f"[SYSTEM]: MODELS WITH {N_BINS} BINS TRAINED\n")

# Classifiation algorithms
# Decision Tree
def decision_tree_model(X_train_, X_test_, y_train_, y_test_,n_bins):
    # Define the model hyperparameters
    tree_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    # Set up the parameter grid
    param_grid = {
        'max_depth': list(range(2, 31)),  # Max depth from 2 to 30
        'min_samples_split': [2, 10, 20, 30, 40],  # Minimum number of samples required to split a node
    }

    # Set up the GridSearchCV with cross-validation and mean squared error as the scoring metric
    grid_search = GridSearchCV(estimator=tree_model,
                            param_grid=param_grid,
                            cv=CV,
                            n_jobs=1, 
                            verbose=1)

    # Train the model
    grid_search.fit(X_train_, y_train_)

    # Predict on the test set
    y_pred = grid_search.predict(X_test_)

    # Calculate metrics
    accuracy = accuracy_score(y_test_, y_pred)
    f1_score_ = f1_score(y_test_, y_pred, average='weighted',zero_division=1)
    precision = precision_score(y_test_, y_pred, average='weighted',zero_division=1)
    roc_acu, roc_curve_path = roc_curve(grid_search,X_test_,y_test_,"DT")

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    
    # Create a new MLflow Experiment
    mlflow.set_experiment("ML-task1-classification")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(grid_search.best_params_)

        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_score_)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("n_bins", n_bins)
        for i in range(len(roc_acu)):
            mlflow.log_metric(f"roc_auc_{i+3}", roc_acu[i])

        mlflow.log_artifact(roc_curve_path)

        # Confusion matrix
        cm = metrics.confusion_matrix(y_test_, y_pred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(f"{matrix_dir}confusion_matrix_DT.png")
        plt.close()
        mlflow.log_artifact(f"{matrix_dir}confusion_matrix_DT.png")
        
        # Classification report
        report = classification_report(y_test_, y_pred, zero_division=1)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(report)
            report_path = f.name  # Guarda la ruta del archivo temporal
        mlflow.log_artifact(report_path, "classification_report")
        
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", f"Basic DT model for wine quality data with {n_bins} bins")

        # Infer the model signature
        signature = infer_signature(X_train_, grid_search.predict(X_train_))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=grid_search,
            artifact_path="wine_quality_model", 
            signature=signature,
            input_example=X_train_,
            registered_model_name="decision_tree_model",
        )

# Gaussian Naive Bayes
def gaussian_naive_bayes_model(X_train, X_test, y_train, y_test, n_bins):
    # Define the model hyperparameters
    gnb = GaussianNB()
    # Set up the parameter grid
    param_grid = {
        'var_smoothing': [1e-13,1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1, 10]  # Smoothing parameter
    }

    # Set up the GridSearchCV with cross-validation and mean squared error as the scoring metric
    grid_search = GridSearchCV(estimator=gnb,
                            param_grid=param_grid,
                            cv=CV,
                            n_jobs=1,
                            verbose=1)
    
    # Train the model
    grid_search.fit(X_train, y_train)
    # Predict on the test set
    y_pred = grid_search.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_score_ = f1_score(y_test, y_pred, average='weighted',zero_division=1)
    precision = precision_score(y_test, y_pred, average='weighted',zero_division=1)
    roc_acu, roc_curve_path = roc_curve(grid_search,X_test,y_test,"GNB")

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("ML-task1-classification")
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(grid_search.best_params_)
        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_score_)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("n_bins", n_bins)
        for i in range(len(roc_acu)):
            mlflow.log_metric(f"roc_auc_{i+3}", roc_acu[i])

        mlflow.log_artifact(roc_curve_path)
        
        # Confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(f"{matrix_dir}confusion_matrix_GNB.png")
        plt.close()
        mlflow.log_artifact(f"{matrix_dir}confusion_matrix_GNB.png")

        # Classification report
        report = classification_report(y_test, y_pred, zero_division=1)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(report)
            report_path = f.name  # Guarda la ruta del archivo temporal
        mlflow.log_artifact(report_path, "classification_report")

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", f"Basic GNB model for wine quality data with {n_bins} bins")
        # Infer the model signature
        signature = infer_signature(X_train, grid_search.predict(X_train))
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=grid_search,
            artifact_path="wine_quality_model", 
            signature=signature,
            input_example=X_train,
            registered_model_name="gaussian_naive_bayes_model",
        )


# Multinomial Naive Bayes
def multinomial_naive_bayes_model(X_train_, X_test_, y_train_, y_test_,n_bins):
    # Define the model hyperparameters
    # Search for the best hyperparameters
    aux = MultinomialNB()
    param_grid = {'alpha': [0.001,0.01,0.1,1.0,2.0,5.0,10.0]}  # Testing alpha values from 0.1 to 1.0
    # Set up GridSearchCV for Multinomial Naive Bayes
    grid_search_multinomial = GridSearchCV(estimator=aux, param_grid=param_grid,
                                        cv=CV, n_jobs=1,verbose=1)

    grid_search_multinomial.fit(X_train_, y_train_)

    # Predict on the test set
    y_pred = grid_search_multinomial.predict(X_test_)
    # Calculate metrics
    accuracy = accuracy_score(y_test_, y_pred)
    f1_score_ = f1_score(y_test_, y_pred, average='weighted',zero_division=1)
    precision = precision_score(y_test_, y_pred, average='weighted',zero_division=1)
    roc_acu, roc_curve_path = roc_curve(grid_search_multinomial,X_test_,y_test_,"MNB")

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("ML-task1-classification")
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(grid_search_multinomial.best_params_)
        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_score_)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("n_bins", n_bins)
        for i in range(len(roc_acu)):
            mlflow.log_metric(f"roc_auc_{i+3}", roc_acu[i])

        mlflow.log_artifact(roc_curve_path)

        # Confusion matrix
        cm = metrics.confusion_matrix(y_test_, y_pred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_multinomial.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(f"{matrix_dir}confusion_matrix_MNB.png")
        plt.close()
        mlflow.log_artifact(f"{matrix_dir}confusion_matrix_MNB.png")

        # Classification report
        report = classification_report(y_test_, y_pred, zero_division=1)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(report)
            report_path = f.name  # Guarda la ruta del archivo temporal
        mlflow.log_artifact(report_path, "classification_report")

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", f"Basic MNB model for wine quality data with {n_bins} bins")
        # Infer the model signature
        signature = infer_signature(X_train_, grid_search_multinomial.predict(X_train_))
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=grid_search_multinomial,
            artifact_path="wine_quality_model", 
            signature=signature,
            input_example=X_train_,
            registered_model_name="multinomial_naive_bayes_model",
        )


# Complement Naive Bayes
def complement_naive_bayes_model(X_train_, X_test_, y_train_, y_test_, n_bins):
    # Define the model hyperparameters
    # Search for the best hyperparameters
    aux = ComplementNB()
    param_grid = {'alpha': [0.1,1.0,2.0,5.0,10.0]}  # Testing alpha values from 0.1 to 1.0

    # Set up GridSearchCV for Multinomial Naive Bayes
    grid_search_complement = GridSearchCV(estimator=aux, param_grid=param_grid,
                                        cv=CV, n_jobs=1,verbose=1)

    grid_search_complement.fit(X_train_, y_train_)

    # Predict on the test set
    y_pred = grid_search_complement.predict(X_test_)
    # Calculate metrics
    accuracy = accuracy_score(y_test_, y_pred)
    f1_score_ = f1_score(y_test_, y_pred, average='weighted',zero_division=1)
    precision = precision_score(y_test_, y_pred, average='weighted',zero_division=1)    
    roc_acu, roc_curve_path = roc_curve(grid_search_complement,X_test_,y_test_,"CNB")

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("ML-task1-classification")
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(grid_search_complement.best_params_)
        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_score_)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("n_bins", n_bins)

        for i in range(len(roc_acu)):
            mlflow.log_metric(f"roc_auc_{i+3}", roc_acu[i])

        mlflow.log_artifact(roc_curve_path)

        # Confusion matrix
        cm = metrics.confusion_matrix(y_test_, y_pred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_complement.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(f"{matrix_dir}confusion_matrix_CNB.png")
        plt.close()
        mlflow.log_artifact(f"{matrix_dir}confusion_matrix_CNB.png")

        # Classification report
        report = classification_report(y_test_, y_pred, zero_division=1)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(report)
            report_path = f.name  # Guarda la ruta del archivo temporal
        mlflow.log_artifact(report_path, "classification_report")

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", f"Basic CNB model for wine quality data with {n_bins} bins")
        # Infer the model signature
        signature = infer_signature(X_train_, grid_search_complement.predict(X_train_))
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=grid_search_complement,
            artifact_path="wine_quality_model", 
            signature=signature,
            input_example=X_train_,
            registered_model_name="complement_naive_bayes_model",
        )

# KNN
def knn_model(X_train, X_test, y_train, y_test,n_bins):
    # Define the model hyperparameters
    # Search for the best hyperparameters
    aux = KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(1, 31)),  # Number of neighbors to use
                'weights': ['uniform', 'distance']  # Weight function used in prediction
                }

    # Set up GridSearchCV for Multinomial Naive Bayes
    grid_knn = GridSearchCV(estimator=aux, param_grid=param_grid,
                                        cv=CV, n_jobs=1,verbose=1)

    grid_knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = grid_knn.predict(X_test)
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_score_ = f1_score(y_test, y_pred, average='weighted',zero_division=1)
    precision = precision_score(y_test, y_pred, average='weighted',zero_division=1)    
    roc_acu, roc_curve_path = roc_curve(grid_knn,X_test,y_test,"KNN")

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("ML-task1-classification")
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(grid_knn.best_params_)
        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_score_)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("n_bins", n_bins)

        for i in range(len(roc_acu)):
            mlflow.log_metric(f"roc_auc_{i+3}", roc_acu[i])

        mlflow.log_artifact(roc_curve_path)

        # Confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_knn.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(f"{matrix_dir}confusion_matrix_KNN.png")
        plt.close()
        mlflow.log_artifact(f"{matrix_dir}confusion_matrix_KNN.png")
        
        # Classification report
        report = classification_report(y_test, y_pred, zero_division=1)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(report)
            report_path = f.name  # Guarda la ruta del archivo temporal
        mlflow.log_artifact(report_path, "classification_report")

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", f"Basic KNN model for wine quality data with {n_bins} bins")
        # Infer the model signature
        signature = infer_signature(X_train, grid_knn.predict(X_train))
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=grid_knn,
            artifact_path="wine_quality_model", 
            signature=signature,
            input_example=X_train,
            registered_model_name="knn_model",
        )

# Random Forest
def random_forest_model(X_train_, X_test_, y_train_, y_test_,n_bins):
    # Define the model hyperparameters
    # Search for the best hyperparameters
    aux = RandomForestClassifier(random_state=42,class_weight='balanced_subsample')
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Set up GridSearchCV for Multinomial Naive Bayes
    grid_rf = GridSearchCV(estimator=aux, param_grid=param_grid,
                                        cv=CV, n_jobs=1,verbose=1)

    grid_rf.fit(X_train_, y_train_)

    # Predict on the test set
    y_pred = grid_rf.predict(X_test_)
    # Calculate metrics
    accuracy = accuracy_score(y_test_, y_pred)
    f1_score_ = f1_score(y_test_, y_pred, average='weighted',zero_division=1)
    precision = precision_score(y_test_, y_pred, average='weighted',zero_division=1)    
    roc_acu, roc_curve_path = roc_curve(grid_rf,X_test_,y_test_,"RF")

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("ML-task1-classification")
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(grid_rf.best_params_)
        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_score_)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("n_bins", n_bins)

        for i in range(len(roc_acu)):
            mlflow.log_metric(f"roc_auc_{i+3}", roc_acu[i])

        mlflow.log_artifact(roc_curve_path)

        # Confusion matrix
        cm = metrics.confusion_matrix(y_test_, y_pred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_rf.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(f"{matrix_dir}confusion_matrix_RF.png")
        plt.close()
        mlflow.log_artifact(f"{matrix_dir}confusion_matrix_RF.png")
        
        # Classification report
        report = classification_report(y_test_, y_pred, zero_division=1)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(report)
            report_path = f.name  # Guarda la ruta del archivo temporal
        mlflow.log_artifact(report_path, "classification_report")

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", f"Basic RF model for wine quality data with {n_bins} bins")
        # Infer the model signature
        signature = infer_signature(X_train_, grid_rf.predict(X_train_))
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=grid_rf,
            artifact_path="wine_quality_model", 
            signature=signature,
            input_example=X_train_,
            registered_model_name="random_forest_model",
        )

########################### AUXILIARY FUNCTIONS ###########################
def roc_curve(model,X_test,y_test,model_name):
    # Predict probabilities instead of class labels
    # As roc curve principally is used for binary classification, we need to binarize the output
    
    # Binarize the output labels for multi-class
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    # Predict probabilities
    y_prob = model.predict_proba(X_test)  # Get probabilities for all classes

    # Compute ROC AUC for each class
    roc_auc = {}
    for i in range(n_classes):
        roc_auc[i] = metrics.roc_auc_score(y_test_bin[:, i], y_prob[:, i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        fpr, tpr, _ = metrics.roc_curve(y_test_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # Plotting the diagonal line for a random classifier
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    # Add labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc='lower right')

    # Save the plot
    roc_curve_path = f'roc_plots/roc_curve_{model_name}.png'
    plt.savefig(roc_curve_path)
    plt.close()

    return roc_auc, roc_curve_path


def load_data(bins : int):
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)
    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets
    df_wine = pd.concat([X, y], axis=1)
    df_wine_aux = df_wine.copy()

    # drop the features with a high correlation (optional if we want to use all the features)
    # X.drop(['density','free_sulfur_dioxide'], axis = 1, inplace = True)
    # df_wine_aux.drop(['density','free_sulfur_dioxide'], axis = 1, inplace = True)

    # we use just 3 features (the most important ones) for our model, those which need to be discretized
    X_mod = X[['alcohol', 'volatile_acidity', 'total_sulfur_dioxide']]
    df_wine_aux_mod = df_wine_aux[['alcohol', 'volatile_acidity', 'total_sulfur_dioxide']]

    # To use clasification algorithms, we need to discretize the features of the dataset
    enc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
    enc_target = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')

    df_wine_aux_mod['alcohol'] = enc.fit_transform(X_mod[['alcohol']])
    df_wine_aux_mod['volatile_acidity'] = enc.fit_transform(X_mod[['volatile_acidity']])
    df_wine_aux_mod['total_sulfur_dioxide'] = enc.fit_transform(X_mod[['total_sulfur_dioxide']])
    y_enc = enc_target.fit_transform(y)


    return X_mod, y_enc, df_wine_aux_mod

def split_data(X,y,df_wine_aux):

    y = np.ravel(y)  # This converts the target into a 1D array
    # for gausioan naive bayes, we need to use the original dataset (continuous features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # for the rest of the algorithms, we need to use the discretized dataset
    X_train_, X_test_, y_train_, y_test_ = train_test_split(df_wine_aux, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X_train_, X_test_, y_train_, y_test_

if __name__ == "__main__":
    main()